import torch
import torch.nn as nn
from hydra.utils import instantiate
from pytorch3d.transforms import matrix_to_axis_angle, rotation_6d_to_matrix

from hmr4d.configs import MainStore, builds
from hmr4d.utils.net_utils import length_to_mask

from .mask_transformer.model.transformer import MaskTransformer
from .tokenization.pose_tokenizer import PoseTokenizer


class MaskedPoseBranch(nn.Module):
    def __init__(self, pose_tokenizer, mask_transformer):
        super().__init__()
        self.pose_tokenizer = instantiate(pose_tokenizer)
        self.mask_transformer = MaskTransformer(mask_transformer)
        self.mask_transformer.load_and_freeze_token_emb(self.pose_tokenizer.get_codebook())
    def forward(self, inputs, pred_context, global_orient_gv_r6d, local_transl_vel, train=False, step=0):
        B, F = pred_context.shape[:2]
        pmask = ~length_to_mask(inputs["length"], F)
        if train:
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
            return {
                "loss": out["ce_loss"] * self.mask_transformer.cfg.loss.lambda_ce,
                "ce_loss": out["ce_loss"],
                "acc": out["acc"],
            }

        with torch.no_grad():
            cano_traj_noisy = torch.cat([global_orient_gv_r6d, local_transl_vel], dim=-1)
            batch = {
                "cond": pred_context,
                "cano_traj_noisy": cano_traj_noisy,
                "key_padding_mask": pmask,
            }
            masked_ids = torch.full(
                (B, F, self.pose_tokenizer.num_tokens),
                self.mask_transformer.mask_id,
                dtype=torch.long,
                device=pred_context.device,
            )
            out = self.mask_transformer.generate(
                batch,
                masked_ids,
                **self.mask_transformer.cfg.test.generate,
            )
            body_pose_r6d = self.pose_tokenizer.decode(out["pred_ids"])
            body_pose = matrix_to_axis_angle(rotation_6d_to_matrix(body_pose_r6d))
            return body_pose.flatten(-2)


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
            "dim_backbone_feat": 512,
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
    populate_full_signature=True,
)
MainStore.store(name="tokenhmr_moro", node=masked_pose_branch, group="masked_pose_branch")

