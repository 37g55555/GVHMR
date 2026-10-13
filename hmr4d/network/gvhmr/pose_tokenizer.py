import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)


class nonlinearity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # swish
        return x * torch.sigmoid(x)


class ResConv1DBlock(nn.Module):
    def __init__(self, n_in, n_state, dilation=1, activation='silu', norm=None, dropout=None):
        super().__init__()
        padding = dilation
        self.norm = norm
        if norm == "LN":
            self.norm1 = nn.LayerNorm(n_in)
            self.norm2 = nn.LayerNorm(n_in)
        elif norm == "GN":
            self.norm1 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
        elif norm == "BN":
            self.norm1 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)

        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        if activation == "relu":
            self.activation1 = nn.ReLU()
            self.activation2 = nn.ReLU()

        elif activation == "silu":
            self.activation1 = nonlinearity()
            self.activation2 = nonlinearity()

        elif activation == "gelu":
            self.activation1 = nn.GELU()
            self.activation2 = nn.GELU()



        self.conv1 = nn.Conv1d(n_in, n_state, 3, 1, padding, dilation)
        self.conv2 = nn.Conv1d(n_state, n_in, 1, 1, 0,)


    def forward(self, x):
        x_orig = x
        if self.norm == "LN":
            x = self.norm1(x.transpose(-2, -1))
            x = self.activation1(x.transpose(-2, -1))
        else:
            x = self.norm1(x)
            x = self.activation1(x)

        x = self.conv1(x)

        if self.norm == "LN":
            x = self.norm2(x.transpose(-2, -1))
            x = self.activation2(x.transpose(-2, -1))
        else:
            x = self.norm2(x)
            x = self.activation2(x)

        x = self.conv2(x)
        x = x + x_orig
        return x


class Resnet1D(nn.Module):
    def __init__(self, n_in, n_depth, dilation_growth_rate=1, reverse_dilation=True, activation='relu', norm=None):
        super().__init__()

        blocks = [ResConv1DBlock(n_in, n_in, dilation=dilation_growth_rate ** depth, activation=activation, norm=norm) for depth in range(n_depth)]
        if reverse_dilation:
            blocks = blocks[::-1]

        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)

class PoseSPEncoderV1(nn.Module):
    def __init__(self,
                 rot_type = 'rotmat',
                 output_emb_width = 512,
                 down_t = 1,
                 stride_t = 2,
                 token_size_mul = 1,
                 width = 512,
                 depth = 2,
                 input_dim = 9,
                 dilation_growth_rate = 3,
                 inp_preprocess = True,
                 target_tokens = None):
        super(PoseSPEncoderV1, self).__init__()

        encoder_layers = []
        num_joints = 21
        filter_t, pad_t = stride_t * 2, stride_t // 2
        self.inp_preprocess = inp_preprocess
        # If set, we will interpolate the final latent sequence to exactly target_tokens (enables arbitrary token counts)
        self.target_tokens = target_tokens
        encoder_layers.append(nn.Conv1d(input_dim, width, 3, 1, 1))
        encoder_layers.append(nn.ReLU())

        # Make num of tokens in multiple of 10
        encoder_layers.append(nn.Upsample(((num_joints*2)//10)*10))
        encoder_layers.append(nn.Conv1d(width, width, 3, 1, 1))
        encoder_layers.append(nn.ReLU())

        for _ in range(token_size_mul-1):
            encoder_layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            encoder_layers.append(nn.Conv1d(width, width, 3, 1, 1))
            encoder_layers.append(nn.ReLU())

        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation='relu', norm=False),
            )
            encoder_layers.append(block)

        encoder_layers.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.encoder = nn.Sequential(*encoder_layers)

    def preprocess(self, x):
        # (bs, num_joints, 3, 3) -> (bs, num_joints, 9) -> (bs, 9, num_joints)
        x = x.view(x.shape[0], x.shape[1], -1)
        x = x.permute(0,2,1)
        return x

    def forward(self, x, global_step=None):
        if self.inp_preprocess:
            x = self.preprocess(x)
        x = self.encoder(x)
        # Resample to arbitrary token length if requested (allows overriding default formula-based length)
        if self.target_tokens is not None and x.shape[-1] != self.target_tokens:
            # Use linear interpolation for smoother feature resizing
            x = F.interpolate(x, size=int(self.target_tokens), mode='linear', align_corners=False)
        return x


class PoseSPDecoderV1(nn.Module):
    def __init__(self,
                 rot_type='rotmat',
                 output_emb_width = 512,
                 down_t = 1,
                 width = 512,
                 depth = 2,
                 token_size_div = 1,
                 num_tokens = 10,
                 dilation_growth_rate = 3,
                 num_joints=21,
                 output_dim = 6,
                 out_postprocess = True):
        super(PoseSPDecoderV1, self).__init__()

        decoder_layers = []
        self.rot_type = rot_type
        self.num_joints = num_joints
        self.out_postprocess = out_postprocess

        decoder_layers.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        decoder_layers.append(nn.ReLU())

        for i in list(np.linspace(self.num_joints, num_tokens, token_size_div, endpoint=False, dtype=int)[::-1]):
            decoder_layers.append(nn.Upsample(i))
            decoder_layers.append(nn.Conv1d(width, width, 3, 1, 1))
            decoder_layers.append(nn.ReLU())

        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation='relu', norm=False),
                nn.Conv1d(width, out_dim, 3, 1, 1)
            )
            decoder_layers.append(block)

        decoder_layers.append(nn.Conv1d(width, output_dim, 3, 1, 1))

        self.decoder = nn.Sequential(*decoder_layers)

    def postprocess(self, x):
        # (bs, 6, num_joints) -> (bs, num_joints, 6)
        x = x.permute(0,2,1)
        return x

    def forward(self, x):
        output = {}
        batch_size = x.shape[0]

        x = self.decoder(x)

        if not self.out_postprocess:
            return x
        pred_pose = self.postprocess(x)
        pred_pose_6d, pred_pose_rotmat = None, None
        if self.rot_type == 'rot6d':
            pred_pose_6d = pred_pose
            pred_pose_rotmat = rotation_6d_to_matrix(pred_pose.reshape(-1, 6)).view(batch_size, self.num_joints, 3, 3)
        elif self.rot_type == 'rotmat':
            NotImplementedError()

        output.update({
            'pred_pose_body_6d': pred_pose_6d,
            'pred_pose_body_rotmat': pred_pose_rotmat,
        })

        return output


class PoseTokenizer(nn.Module):
    def __init__(self, checkpoint_path):
        super().__init__()
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        arch = checkpoint["hparams"].ARCH

        num_joints = 21
        self.code_dim = int(arch.CODE_DIM)
        self.codebook_size = int(arch.NB_CODE)
        num_tokens = getattr(arch, "NUM_TOKENS", None)
        if num_tokens is None or int(num_tokens) <= 0:
            self.num_tokens = int(((num_joints // 10) * 10) * (2 ** int(arch.TOKEN_SIZE_MUL)) / (2 ** int(arch.DOWN_T)))
            target_tokens = None
        else:
            self.num_tokens = int(num_tokens)
            target_tokens = self.num_tokens

        self.encoder = PoseSPEncoderV1(
            rot_type=arch.ROT_TYPE,
            input_dim=6,
            output_emb_width=self.code_dim,
            down_t=int(arch.DOWN_T),
            width=int(arch.WIDTH),
            depth=int(arch.DEPTH),
            token_size_mul=int(arch.TOKEN_SIZE_MUL),
            dilation_growth_rate=int(arch.DILATION_RATE),
            target_tokens=target_tokens,
        )
        self.decoder = PoseSPDecoderV1(
            rot_type=arch.ROT_TYPE,
            output_dim=6,
            output_emb_width=self.code_dim,
            down_t=int(arch.DOWN_T),
            width=int(arch.WIDTH),
            depth=int(arch.DEPTH),
            token_size_div=int(arch.TOKEN_SIZE_DIV),
            num_tokens=self.num_tokens,
            dilation_growth_rate=int(arch.DILATION_RATE),
            num_joints=num_joints,
        )
        state_dict = checkpoint["net"]
        encoder_state = {
            key.removeprefix("encoder."): value for key, value in state_dict.items() if key.startswith("encoder.")
        }
        decoder_state = {
            key.removeprefix("decoder."): value for key, value in state_dict.items() if key.startswith("decoder.")
        }
        self.encoder.load_state_dict(encoder_state, strict=True)
        self.decoder.load_state_dict(decoder_state, strict=True)
        self.register_buffer("codebook", state_dict["quantizer.codebook"].clone())
        self.requires_grad_(False)

    @torch.no_grad()
    def encode(self, body_pose):
        batch_size, num_frames = body_pose.shape[:2]
        rotations = axis_angle_to_matrix(body_pose.reshape(-1, 21, 3))
        rotations = matrix_to_rotation_6d(rotations)
        x_encoder = self.encoder(rotations)
        x_encoder = x_encoder.permute(0, 2, 1).contiguous()
        x_encoder = x_encoder.view(-1, x_encoder.shape[-1])
        k_w = self.codebook.t()
        distance = (
            torch.sum(x_encoder**2, dim=-1, keepdim=True)
            - 2 * torch.matmul(x_encoder, k_w)
            + torch.sum(k_w**2, dim=0, keepdim=True)
        )
        _, code_idx = torch.min(distance, dim=-1)
        return code_idx.reshape(batch_size, num_frames, self.num_tokens)

    def decode_features(self, decode_feat, batch_size, num_frames):
        decode_feat = decode_feat.reshape(-1, self.num_tokens, self.code_dim)
        pose_out = self.decoder(decode_feat.permute(0, 2, 1))
        rotations = pose_out["pred_pose_body_rotmat"]
        rotations = matrix_to_axis_angle(rotations)
        return rotations.reshape(batch_size, num_frames, 63)

    def decode_probabilities(self, probabilities):
        batch_size, num_frames = probabilities.shape[:2]
        decode_feat = torch.matmul(probabilities, self.codebook)
        return self.decode_features(decode_feat, batch_size, num_frames)

    def decode_ids(self, token_ids):
        batch_size, num_frames = token_ids.shape[:2]
        decode_feat = F.embedding(token_ids, self.codebook)
        return self.decode_features(decode_feat, batch_size, num_frames)
