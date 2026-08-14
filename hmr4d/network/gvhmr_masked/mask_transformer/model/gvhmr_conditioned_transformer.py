import torch
import torch.nn as nn
import einops

from .DSTFormer import DSTFormer, trunc_normal_


class GVHMRConditionedTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.dim_in = cfg.dim_in
        self.dim_out = cfg.dim_out
        self.dim_feat = cfg.dim_feat
        self.dim_backbone_feat = cfg.dim_backbone_feat
        self.dim_cano_traj = cfg.get("dim_cano_traj", 9)

        self.mdepth = cfg.mdepth
        self.ddepth = cfg.ddepth

        self.num_tokens = cfg.num_tokens

        self.m_pos_enc = torch.nn.Parameter(
            torch.zeros(1, 1, self.num_tokens + 1, self.dim_feat)
        )
        trunc_normal_(self.m_pos_enc, std=0.02)

        self.m_token_embed = nn.Linear(self.dim_in, self.dim_feat)
        self.m_transformer = DSTFormer(cfg, self.dim_feat, self.mdepth)
        self.m_traj_embed = nn.Linear(self.dim_cano_traj, self.dim_feat)

        self.img_embed = nn.Linear(self.dim_backbone_feat, self.dim_feat)

        self.d_pos_enc = torch.nn.Parameter(
            torch.zeros(1, 1, self.num_tokens + 2, self.dim_feat)
        )
        trunc_normal_(self.d_pos_enc, std=0.02)

        self.d_transformer = DSTFormer(cfg, self.dim_feat, self.ddepth)
        self.d_pose_regressor = nn.Linear(self.dim_feat, self.dim_out)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def motion_encode(self, tokens, cano_traj_noisy, key_padding_mask=None):
        # embed to same dimension
        tokens = self.m_token_embed(tokens)
        cano_traj_input = self.m_traj_embed(cano_traj_noisy)
        cano_traj_input = cano_traj_input.unsqueeze(2)

        motion_feat = torch.cat([tokens, cano_traj_input], dim=2)

        # learnable spatial PE
        motion_feat = motion_feat + self.m_pos_enc

        motion_feat = self.m_transformer(motion_feat, key_padding_mask=key_padding_mask)

        local_pose_feat = motion_feat[:, :, : self.num_tokens, :]
        cano_traj_feat = motion_feat[:, :, -1:, :]

        return local_pose_feat, cano_traj_feat

    def decode(self, feat, key_padding_mask=None):
        # Decode token logits from motion + video features

        # learnable spatial PE
        feat[:, :, :self.num_tokens] = feat[:, :, :self.num_tokens] + self.d_pos_enc[:, :, :self.num_tokens]
        feat[:, :, self.num_tokens:-1] = feat[:, :, self.num_tokens:-1] + self.d_pos_enc[:, :, -2:-1]
        feat[:, :, -1:] = feat[:, :, -1:] + self.d_pos_enc[:, :, -1:]

        feat = self.d_transformer(feat, key_padding_mask=key_padding_mask)

        local_pose_feat = feat[:, :, : self.num_tokens, :]
        local_pose_logits = self.d_pose_regressor(local_pose_feat)
        local_pose_logits = einops.rearrange(local_pose_logits, "b f j c -> b c f j")

        return local_pose_logits

    def forward(self, tokens, batch, cond_out=None):
        """
        tokens: [B, F, J, C]
        """
        cano_traj_noisy = batch["cano_traj_noisy"]
        key_padding_mask = batch.get("key_padding_mask", None)
        local_pose_feat, cano_traj_feat = self.motion_encode(
            tokens, cano_traj_noisy, key_padding_mask=key_padding_mask
        )

        img_feat = self.img_embed(batch["cond"]).unsqueeze(2)
        if cond_out is not None:
            img_feat = torch.zeros_like(cond_out["img_feat"])

        feat = torch.cat(
            [local_pose_feat, img_feat, cano_traj_feat], dim=2
        )
        local_pose_logits = self.decode(feat, key_padding_mask=key_padding_mask)

        out = {
            "logits": local_pose_logits,
            "img_feat": img_feat,
        }
        return out

    def inference(self, tokens, batch, last_output=None):
        return self(tokens, batch)

