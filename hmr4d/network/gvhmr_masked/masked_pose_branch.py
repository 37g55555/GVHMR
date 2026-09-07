import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from pytorch3d.transforms import matrix_to_axis_angle, rotation_6d_to_matrix

from hmr4d.configs import MainStore, builds
from hmr4d.utils.net_utils import length_to_mask

from .mask_transformer.model.transformer import MaskTransformer
from .mask_transformer.model.smoother import TemporalSmoother
from .tokenization.pose_tokenizer import PoseTokenizer


class MaskedPoseBranch(nn.Module):
    def __init__(
        self, pose_tokenizer, mask_transformer, smoother=None, gumbel=None, lambda_v=None, clip_len=60, overlap_len=2
    ):
        super().__init__()
        self.pose_tokenizer = instantiate(pose_tokenizer)
        self.mask_transformer = MaskTransformer(mask_transformer)
        self.smoother = TemporalSmoother(smoother) if smoother is not None else None
        self.gumbel = gumbel
        self.lambda_v = lambda_v
        self.clip_len = clip_len
        self.clip_overlap_len = overlap_len
        self.mask_transformer.load_and_freeze_token_emb(self.pose_tokenizer.get_codebook())

    def forward(self, inputs, pred_context, global_orient_gv_r6d, local_transl_vel, train=False, step=0):
        B, F = pred_context.shape[:2]
        if train:
            pmask = ~length_to_mask(inputs["length"], F)
            ids = self.pose_tokenizer.encode(inputs["smpl_params_c"]["body_pose"])
            ids = ids.masked_fill(pmask[..., None], self.mask_transformer.mask_id)
            cano_traj_noisy = torch.cat([global_orient_gv_r6d, local_transl_vel], dim=-1)
            batch = {
                "ids": ids,
                "cond": pred_context,
                "cano_traj_noisy": cano_traj_noisy,
                "key_padding_mask": pmask,
            }
            out = self.mask_transformer.training_step(batch, step=step)
            result = {
                "loss": out["ce_loss"] * self.mask_transformer.cfg.loss.lambda_ce,
                "ce_loss": out["ce_loss"],
                "acc": out["acc"],
            }
            if self.smoother is not None:
                ratio = min(step / (self.gumbel.total_steps * self.gumbel.temp_end_ratio), 1.0)
                temperature = self.gumbel.temp_end + 0.5 * (self.gumbel.temp_start - self.gumbel.temp_end) * (
                    1 + math.cos(ratio * math.pi)
                )
                probabilities = F.gumbel_softmax(
                    out["logits"], tau=temperature, hard=self.gumbel.hard, dim=1
                )
                latent = self.pose_tokenizer.probabilities_to_latent(probabilities)
                latent = self.smoother(latent)
                body_pose_r6d = self.pose_tokenizer.decode_latent(latent)
                result["smoothed_body_pose"] = matrix_to_axis_angle(
                    rotation_6d_to_matrix(body_pose_r6d)
                ).flatten(-2)
            return result

        with torch.no_grad():
            window_size = self.clip_overlap_len // 2
            body_pose_list = []
            seq_idx = 0
            while True:
                start = seq_idx * (self.clip_len - self.clip_overlap_len)
                end = start + self.clip_len
                actual_end = min(end, F)
                length = (inputs["length"] - start).clamp(min=0, max=actual_end - start)
                pmask = ~length_to_mask(length, actual_end - start)
                cano_traj_noisy = torch.cat(
                    [global_orient_gv_r6d[:, start:actual_end], local_transl_vel[:, start:actual_end]], dim=-1
                )
                batch = {
                    "cond": pred_context[:, start:actual_end],
                    "cano_traj_noisy": cano_traj_noisy,
                    "key_padding_mask": pmask,
                }
                masked_ids = torch.full(
                    (B, actual_end - start, self.pose_tokenizer.num_tokens),
                    self.mask_transformer.mask_id,
                    dtype=torch.long,
                    device=pred_context.device,
                )
                out = self.mask_transformer.generate(
                    batch,
                    masked_ids,
                    **self.mask_transformer.cfg.test.generate,
                )
                if self.smoother is None:
                    body_pose_r6d = self.pose_tokenizer.decode(out["pred_ids"])
                else:
                    latent = self.pose_tokenizer.ids_to_latent(out["pred_ids"])
                    latent = self.smoother(latent)
                    body_pose_r6d = self.pose_tokenizer.decode_latent(latent)
                body_pose = matrix_to_axis_angle(rotation_6d_to_matrix(body_pose_r6d)).flatten(-2)
                if len(body_pose_list) == 0 or window_size == 0:
                    body_pose_list.append(body_pose)
                else:
                    body_pose_list[-1] = body_pose_list[-1][:, :-window_size]
                    body_pose_list.append(body_pose[:, window_size:])
                seq_idx += 1
                if end >= F:
                    break

            return torch.cat(body_pose_list, dim=1)


pose_tokenizer = builds(
    PoseTokenizer,
    ckpt_path="inputs/checkpoints/tokenhmr/tokenizer.pth",
    populate_full_signature=True,
)
masked_pose_branch = builds(
    MaskedPoseBranch,
    pose_tokenizer=pose_tokenizer,
    mask_transformer={
        "task": "video",
        "num_tokens": 160,
        "num_codes": 2048,
        "dim_token": 256,
        "mask_scheme": [[-1, "inference+random"]],
        "scheme_prob": {
            "inference": 0.4,
            "random": 0.4,
        },
        "mask_tau": 1.0,
        "prob_rid": 0.1,
        "prob_mid": 1.0,
        "full_label": False,
        "dst_former": {
            "num_tokens": "${..num_tokens}",
            "dim_in": "${..dim_token}",
            "dim_out": "${..num_codes}",
            "dim_backbone_feat": "${network.latent_dim}",
            "dim_feat": 1024,
            "dim_cano_traj": 9,
            "mdepth": 4,
            "ddepth": 2,
            "num_heads": 8,
            "mlp_ratio": 4,
            "qkv_bias": True,
            "qk_scale": None,
            "drop_rate": 0.0,
            "attn_drop_rate": "${.drop_rate}",
            "att_fuse": True,
            "attn_len": 60,
        },
        "loss": {
            "lambda_ce": 1.0,
            "focal_gamma": 0.0,
        },
        "test": {
            "generate": {
                "timesteps": 5,
                "cond_scale": 1.0,
                "temperature": 1.0,
                "topk": 1,
                "gsample": False,
            },
        },
    },
    clip_len=60,
    overlap_len=2,
    populate_full_signature=True,
)
MainStore.store(name="tokenhmr_moro", node=masked_pose_branch, group="masked_pose_branch")

masked_pose_branch_smoother = builds(
    MaskedPoseBranch,
    pose_tokenizer=pose_tokenizer,
    mask_transformer=masked_pose_branch().mask_transformer,
    smoother={
        "num_tokens": 160,
        "dim_in": 256,
        "share_weights": True,
        "dim_hidden": 64,
        "kernel_size": 5,
        "num_layers": 2,
    },
    gumbel=None,
    lambda_v=None,
    clip_len=60,
    overlap_len=2,
    populate_full_signature=True,
)
MainStore.store(
    name="tokenhmr_moro_smoother", node=masked_pose_branch_smoother, group="masked_pose_branch"
)
