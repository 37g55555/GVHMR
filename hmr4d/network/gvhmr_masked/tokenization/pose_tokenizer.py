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

        self.num_joints = self.decoder.decoder.num_joints
        self.num_codes = self.encoder.quantizer.nb_code
        self.num_tokens = self.decoder.num_tokens
        self.code_dim = self.encoder.quantizer.code_dim

    @torch.no_grad()
    def encode(self, body_pose):
        B, L = body_pose.shape[:2]
        body_pose = body_pose.reshape(B * L, self.num_joints, 3)
        body_pose = matrix_to_rotation_6d(axis_angle_to_matrix(body_pose))
        ids = self.encoder(body_pose)
        return ids.reshape(B, L, self.num_tokens)

    @torch.no_grad()
    def decode(self, ids):
        B, L = ids.shape[:2]
        ids = ids.reshape(B * L, self.num_tokens)
        logits = F.one_hot(ids, self.num_codes).float()
        body_pose_r6d = self.decoder(logits)
        return body_pose_r6d.reshape(B, L, self.num_joints, 6)

    def decode_latent(self, latent):
        B, L = latent.shape[:2]
        latent = latent.reshape(B * L, self.num_tokens, self.code_dim)
        body_pose_r6d = self.decoder.decode_latent(latent)
        return body_pose_r6d.reshape(B, L, self.num_joints, 6)

    def get_codebook(self):
        return self.encoder.quantizer.codebook
