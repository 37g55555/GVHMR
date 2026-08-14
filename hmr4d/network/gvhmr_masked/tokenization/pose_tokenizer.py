import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d

from .models.vanilla_pose_vqvae import DecodeTokens, EncodeTokens


class PoseTokenizer(nn.Module):
    def __init__(self, ckpt_path):
        super().__init__()
        self.encoder = EncodeTokens(ckpt_path)
        self.decoder = DecodeTokens(ckpt_path)

        ckpt = torch.load(ckpt_path, map_location='cpu')
        arch = ckpt['hparams'].ARCH
        self.num_codes = arch.NB_CODE
        num_tokens = getattr(arch, 'NUM_TOKENS', None)
        if num_tokens is None or int(num_tokens) <= 0:
            num_tokens = int(((21//10)*10) * (2**(arch.TOKEN_SIZE_MUL)) / (2**arch.DOWN_T))
        self.num_tokens = int(num_tokens)
        self.code_dim = arch.CODE_DIM

    @torch.no_grad()
    def encode(self, body_pose):
        B, L = body_pose.shape[:2]
        body_pose = body_pose.reshape(B * L, 21, 3)
        body_pose = matrix_to_rotation_6d(axis_angle_to_matrix(body_pose))
        ids = self.encoder(body_pose)
        return ids.reshape(B, L, self.num_tokens)

    @torch.no_grad()
    def decode(self, ids):
        B, L = ids.shape[:2]
        ids = ids.reshape(B * L, self.num_tokens)
        logits = F.one_hot(ids, self.num_codes).float()
        body_pose_r6d = self.decoder(logits)
        return body_pose_r6d.reshape(B, L, 21, 6)

    def get_codebook(self):
        return self.encoder.quantizer.codebook

