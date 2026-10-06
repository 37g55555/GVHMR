import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from hmr4d.network.base_arch.embeddings.rotary_embedding import ROPE
from hmr4d.utils.net_utils import length_to_mask


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        cfg,
        dim,
        st_mode="temporal",
    ):
        super().__init__()

        num_heads = cfg.num_heads
        qkv_bias = cfg.get("qkv_bias", True)
        qk_scale = cfg.get("qk_scale", None)
        proj_drop = cfg.get("drop_rate", 0.0)
        attn_drop = cfg.get("attn_drop_rate", 0.0)

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim**-0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.mode = st_mode
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.mode == "temporal":
            self.rope = ROPE(head_dim)
            self.attn_len = cfg.get("attn_len", 60)

    def forward(self, x, valid_mask):
        B, F, J, C = x.shape

        if self.mode == "temporal":
            x = x.permute(0, 2, 1, 3).reshape(B * J, F, C)
            qkv = self.qkv(x).reshape(-1, F, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = (
                qkv[0],
                qkv[1],
                qkv[2],
            )

            q = self.rope.rotate_queries_or_keys(q)
            k = self.rope.rotate_queries_or_keys(k)

            key_mask = valid_mask[:, None].expand(-1, J, -1).reshape(B * J, F)
            x = self.forward_attention(q, k, v, key_mask)
            x = x.reshape(B, J, F, C).permute(0, 2, 1, 3)
        elif self.mode == "spatial":
            x = x.reshape(B * F, J, C)
            qkv = self.qkv(x).reshape(-1, J, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = (
                qkv[0],
                qkv[1],
                qkv[2],
            )
            x = self.forward_attention(q, k, v)
            x = x.reshape(B, F, J, C)
        else:
            raise NotImplementedError(self.mode)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_attention(self, q, k, v, key_mask=None):
        B, H, N, C = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.mode == "temporal":
            attn_mask = torch.zeros(N, N, dtype=torch.bool, device=q.device)
            for i in range(N):
                start = max(0, i - self.attn_len)
                end = min(N, i + self.attn_len + 1)
                attn_mask[i, start:end] = True

            attn_mask = attn_mask.reshape(1, 1, N, N).expand(B, H, -1, -1)
            attn = attn.masked_fill(attn_mask.logical_not(), float("-inf"))
            attn = attn.masked_fill(~key_mask[:, None, None], float("-inf"))

        if self.mode == "temporal":
            row_valid = (attn_mask & key_mask[:, None, None]).any(dim=-1, keepdim=True)
            attn = attn.masked_fill(~row_valid, 0.0)

        attn = attn.softmax(dim=-1)

        if self.mode == "temporal":
            attn = attn.masked_fill(~row_valid, 0.0)

        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C * self.num_heads)
        return x


class Block(nn.Module):
    def __init__(
        self,
        cfg,
        dim,
        drop_path=0.0,
        mlp_out_ratio=1.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        st_mode="stage_st",
        att_fuse=False,
    ):
        super().__init__()
        self.cfg = cfg
        mlp_ratio = cfg.get("mlp_ratio", 4)
        drop = cfg.get("drop_rate", 0.0)

        self.st_mode = st_mode

        self.norm1_s = norm_layer(dim)
        self.norm1_t = norm_layer(dim)
        self.attn_s = Attention(
            cfg,
            dim,
            st_mode="spatial",
        )
        self.attn_t = Attention(
            cfg,
            dim,
            st_mode="temporal",
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2_s = norm_layer(dim)
        self.norm2_t = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_out_dim = int(dim * mlp_out_ratio)
        self.mlp_s = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=mlp_out_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.mlp_t = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=mlp_out_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.att_fuse = att_fuse
        if self.att_fuse:
            self.ts_attn = nn.Linear(dim * 2, dim * 2)

    def forward(self, x, valid_mask):
        if self.st_mode == "stage_st":
            x = x + self.drop_path(self.attn_s(self.norm1_s(x), valid_mask))
            x = x + self.drop_path(self.mlp_s(self.norm2_s(x)))
            x = x + self.drop_path(self.attn_t(self.norm1_t(x), valid_mask))
            x = x + self.drop_path(self.mlp_t(self.norm2_t(x)))
        elif self.st_mode == "stage_ts":
            x = x + self.drop_path(self.attn_t(self.norm1_t(x), valid_mask))
            x = x + self.drop_path(self.mlp_t(self.norm2_t(x)))
            x = x + self.drop_path(self.attn_s(self.norm1_s(x), valid_mask))
            x = x + self.drop_path(self.mlp_s(self.norm2_s(x)))
        elif self.st_mode == "stage_para":
            x_t = x + self.drop_path(self.attn_t(self.norm1_t(x), valid_mask))
            x_t = x_t + self.drop_path(self.mlp_t(self.norm2_t(x_t)))
            x_s = x + self.drop_path(self.attn_s(self.norm1_s(x), valid_mask))
            x_s = x_s + self.drop_path(self.mlp_s(self.norm2_s(x_s)))
            if self.att_fuse:
                B, F, J, C = x_s.shape
                alpha = torch.cat([x_s, x_t], dim=-1)
                alpha = self.ts_attn(alpha).reshape(B, F, J, C, 2)
                alpha = alpha.softmax(dim=-1)
                x = x_t * alpha[..., 1] + x_s * alpha[..., 0]
            else:
                x = (x_s + x_t) * 0.5
        else:
            raise NotImplementedError(self.st_mode)
        return x


class DSTFormer(nn.Module):
    def __init__(self, cfg, dim_feat, depth):
        super().__init__()
        self.cfg = cfg

        self.dim_feat = dim_feat
        self.depth = depth

        self.num_heads = cfg.num_heads
        self.mlp_ratio = cfg.mlp_ratio
        self.qkv_bias = cfg.get("qkv_bias", True)
        self.qk_scale = cfg.get("qk_scale", None)
        self.drop_rate = cfg.get("drop_rate", 0.0)
        self.attn_drop_rate = cfg.get("attn_drop_rate", 0.0)

        self.att_fuse = cfg.get("att_fuse", True)

        self.blocks_st = nn.ModuleList(
            [
                Block(
                    self.cfg,
                    dim=self.dim_feat,
                    st_mode="stage_st",
                )
                for i in range(self.depth)
            ]
        )
        self.blocks_ts = nn.ModuleList(
            [
                Block(
                    self.cfg,
                    dim=self.dim_feat,
                    st_mode="stage_ts",
                )
                for i in range(self.depth)
            ]
        )

        if self.att_fuse:
            self.ts_attn = nn.ModuleList([nn.Linear(self.dim_feat * 2, 2) for i in range(self.depth)])
            for i in range(self.depth):
                self.ts_attn[i].weight.data.fill_(0)
                self.ts_attn[i].bias.data.fill_(0.5)

    def forward(self, x, valid_mask):
        frame_mask = valid_mask[:, :, None, None]
        for idx, (blk_st, blk_ts) in enumerate(zip(self.blocks_st, self.blocks_ts)):
            x_st = blk_st(x, valid_mask)
            x_ts = blk_ts(x, valid_mask)
            if self.att_fuse:
                att = self.ts_attn[idx]
                alpha = torch.cat([x_st, x_ts], dim=-1)
                alpha = att(alpha)
                alpha = alpha.softmax(dim=-1)
                x = x_st * alpha[..., 0:1] + x_ts * alpha[..., 1:2]
            else:
                x = (x_st + x_ts) * 0.5
            x = x * frame_mask
        return x


def uniform(shape, device=None):
    return torch.zeros(shape, device=device).float().uniform_(0, 1)


def get_mask_subset_prob(mask, prob):
    subset_mask = torch.bernoulli(mask, p=prob) & mask
    return subset_mask


def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)


class MaskedPoseModel(nn.Module):
    def __init__(
        self,
        num_tokens,
        codebook_size,
        code_dim,
        context_dim=512,
        dim=1024,
        motion_depth=4,
        decoder_depth=2,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.0,
        attn_len=60,
        mask_modes=("random", "joint", "frame"),
        mask_mode_weights=(0.4, 0.2, 0.2),
        mask_tau=1.0,
        random_replace_prob=0.1,
        mask_replace_prob=1.0,
        generation_steps=5,
        generation_temperature=1.0,
        temp_start=1.0,
        temp_end=0.01,
        temp_end_ratio=1.0,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.codebook_size = codebook_size
        self.mask_id = codebook_size
        self.mask_modes = tuple(mask_modes)
        self.register_buffer("mask_mode_weights", torch.tensor(mask_mode_weights), persistent=False)
        self.mask_tau = mask_tau
        self.random_replace_prob = random_replace_prob
        self.mask_replace_prob = mask_replace_prob
        self.generation_steps = generation_steps
        self.generation_temperature = generation_temperature
        self.temp_start = temp_start
        self.temp_end = temp_end
        self.temp_end_ratio = temp_end_ratio

        transformer_cfg = OmegaConf.create(
            {
                "num_heads": num_heads,
                "mlp_ratio": mlp_ratio,
                "qkv_bias": True,
                "qk_scale": None,
                "drop_rate": dropout,
                "attn_drop_rate": dropout,
                "att_fuse": True,
                "attn_len": attn_len,
            }
        )

        self.register_buffer("mask_token", torch.zeros(code_dim), persistent=False)
        self.m_token_embed = nn.Linear(code_dim, dim)
        self.m_traj_embed = nn.Linear(9, dim)
        self.m_pos_enc = nn.Parameter(torch.zeros(1, 1, num_tokens + 1, dim))
        trunc_normal_(self.m_pos_enc, std=0.02)
        self.m_transformer = DSTFormer(transformer_cfg, dim, motion_depth)

        self.context_projection = nn.Linear(context_dim, dim)
        self.d_pos_enc = nn.Parameter(torch.zeros(1, 1, num_tokens + 2, dim))
        trunc_normal_(self.d_pos_enc, std=0.02)
        self.d_transformer = DSTFormer(transformer_cfg, dim, decoder_depth)
        self.d_pose_regressor = nn.Linear(dim, codebook_size)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def embed_tokens(self, token_ids, codebook):
        code_ids = token_ids.clamp_max(self.codebook_size - 1)
        embeddings = F.embedding(code_ids, codebook)
        mask_token = self.mask_token.to(embeddings)
        return torch.where((token_ids == self.mask_id)[..., None], mask_token, embeddings)

    def motion_encode(self, token_ids, gamma_gv, root_velocity, valid_mask, codebook):
        tokens = self.m_token_embed(self.embed_tokens(token_ids, codebook))
        traj_input = self.m_traj_embed(torch.cat([gamma_gv, root_velocity], dim=-1)).unsqueeze(2)
        motion_feat = torch.cat([tokens, traj_input], dim=2)
        motion_feat = motion_feat + self.m_pos_enc
        motion_feat = self.m_transformer(motion_feat, valid_mask)
        return motion_feat

    def decode(self, motion_feat, visual_context, valid_mask):
        local_pose_feat = motion_feat[:, :, : self.num_tokens]
        traj_feat = motion_feat[:, :, -1:]
        visual_feat = self.context_projection(visual_context).unsqueeze(2)
        feat = torch.cat([local_pose_feat, visual_feat, traj_feat], dim=2)
        feat = feat + self.d_pos_enc
        feat = self.d_transformer(feat, valid_mask)
        local_pose_feat = feat[:, :, : self.num_tokens]
        return self.d_pose_regressor(local_pose_feat)

    def forward(self, token_ids, gamma_gv, root_velocity, visual_context, length, codebook):
        valid_mask = length_to_mask(length, token_ids.shape[1])
        motion_feat = self.motion_encode(token_ids, gamma_gv, root_velocity, valid_mask, codebook)
        return self.decode(motion_feat, visual_context, valid_mask)

    def get_random_mask(self, valid_tokens, ratios):
        B = valid_tokens.shape[0]
        flat_valid = valid_tokens.flatten(1)
        random_order = torch.rand(flat_valid.shape, device=flat_valid.device).masked_fill(~flat_valid, 2.0)
        ranks = random_order.argsort(dim=-1).argsort(dim=-1)
        num_token_masked = (flat_valid.sum(dim=-1) * ratios).round().long().clamp_min(1)
        return ((ranks < num_token_masked[:, None]) & flat_valid).reshape_as(valid_tokens)

    def get_frame_mask(self, valid_tokens, length, ratios):
        B, F, J = valid_tokens.shape
        mask_len = (length * ratios).round().long().clamp_min(1)
        start = (length - mask_len) * torch.rand_like(ratios)
        start = start.round().int()
        end = (start + mask_len).int()
        mask = torch.zeros((B, F, J), device=valid_tokens.device, dtype=torch.bool)
        for b in range(B):
            mask[b, start[b] : end[b], :] = True
        return mask & valid_tokens

    def get_joint_mask(self, valid_tokens, ratios):
        B, F, J = valid_tokens.shape
        num_token_masked = (J * ratios).round().long().clamp_min(1)
        batch_randperm = torch.rand((B, J), device=valid_tokens.device).argsort(dim=-1)
        mask = batch_randperm < num_token_masked.unsqueeze(-1)
        mask = mask.unsqueeze(1).expand(B, F, J)
        return mask & valid_tokens

    def get_mask(self, token_ids, length):
        B, F, J = token_ids.shape
        valid_frames = length_to_mask(length, F)
        valid_tokens = valid_frames[:, :, None].expand(B, F, J)
        rand_time = uniform((B,), device=token_ids.device) * self.mask_tau
        ratios = cosine_schedule(rand_time)
        mode_index = int(torch.multinomial(self.mask_mode_weights, 1))
        mode = self.mask_modes[mode_index]
        if mode == "random":
            return self.get_random_mask(valid_tokens, ratios)
        if mode == "frame":
            return self.get_frame_mask(valid_tokens, length, ratios)
        return self.get_joint_mask(valid_tokens, ratios)

    def corrupt(self, token_ids, mask):
        x_ids = token_ids.clone()
        mask_rid = get_mask_subset_prob(mask, self.random_replace_prob)
        rand_id = torch.randint_like(x_ids, high=self.codebook_size)
        x_ids = torch.where(mask_rid, rand_id, x_ids)
        mask_mid = get_mask_subset_prob(mask & ~mask_rid, self.mask_replace_prob)
        return torch.where(mask_mid, self.mask_id, x_ids)

    def forward_train(self, token_ids, gamma_gv, root_velocity, visual_context, length, codebook):
        mask = self.get_mask(token_ids, length)
        input_ids = self.corrupt(token_ids, mask)
        logits = self(input_ids, gamma_gv, root_velocity, visual_context, length, codebook)
        return logits, mask

    def get_training_temperature(self, progress):
        ratio = min(progress / self.temp_end_ratio, 1.0)
        return self.temp_end + 0.5 * (self.temp_start - self.temp_end) * (1 + math.cos(ratio * math.pi))

    @torch.no_grad()
    def generate(self, gamma_gv, root_velocity, visual_context, length, codebook):
        B, F = visual_context.shape[:2]
        valid_tokens = length_to_mask(length, F)[:, :, None].expand(B, F, self.num_tokens)
        ids = torch.zeros(valid_tokens.shape, dtype=torch.long, device=visual_context.device)
        ids[valid_tokens] = self.mask_id
        num_masked_init = valid_tokens.flatten(1).sum(dim=-1)

        for timestep in range(self.generation_steps):
            is_mask = ids == self.mask_id
            logits = self(ids, gamma_gv, root_velocity, visual_context, length, codebook)
            probs = nn.functional.softmax(logits / self.generation_temperature, dim=-1)
            pred_ids = probs.argmax(dim=-1)
            ids = torch.where(is_mask, pred_ids, ids)

            probs_without_temperature = logits.softmax(dim=-1)
            scores = probs_without_temperature.gather(-1, pred_ids.unsqueeze(dim=-1)).squeeze(-1)
            scores = scores.masked_fill(~is_mask, 1e5)

            if timestep + 1 == self.generation_steps:
                break
            rand_mask_prob = (torch.cos(torch.tensor((timestep + 1) / self.generation_steps * math.pi)) + 1) * 0.5
            num_token_masked = torch.round(rand_mask_prob.to(ids.device) * num_masked_init).clamp(min=1)
            sorted_indices = scores.flatten(1).argsort(dim=1)
            ranks = sorted_indices.argsort(dim=1)
            is_mask = (ranks < num_token_masked.unsqueeze(-1)).reshape_as(ids) & valid_tokens
            ids = torch.where(is_mask, self.mask_id, ids)

        return ids
