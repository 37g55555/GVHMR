import re
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from .resnet import Resnet1D
from .quantize_cnn import QuantizeEMAReset
from pytorch3d.transforms import rotation_6d_to_matrix


def prepare_statedict(model, full_state_dict, partname, ignore_partname=' '):
    part_statedict = {}
    new_part_statedict = OrderedDict()

    # Load only the part given by sel_partname
    for key in full_state_dict.keys():
        if key.startswith(f'{partname}') and ignore_partname not in key:
            part_statedict[key] = full_state_dict[key]

    # Replace mismatch names
    for name, param in part_statedict.items():
        if re.match(f'^{partname}', name):
            name = name.replace(f'{partname}.', '', 1)
        new_part_statedict[name] = param

    model.load_state_dict(new_part_statedict, strict=True)
    return model

class PoseSPEncoderV1(nn.Module):
    def __init__(self,
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

    def forward(self, x):
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


class DecodeTokens(nn.Module):
    def __init__(self,
                 ckpt_path=''):
        super(DecodeTokens, self).__init__()

        num_joints = 21
        ckpt = torch.load(ckpt_path, map_location='cpu')
        pretrained_hparams = ckpt['hparams']
        arch = pretrained_hparams.ARCH
        rot_type = arch.ROT_TYPE
        code_dim = arch.CODE_DIM
        nb_code = arch.NB_CODE
        output_emb_width = code_dim
        down_t = arch.DOWN_T
        width = arch.WIDTH
        depth = arch.DEPTH
        dilation_growth_rate = arch.DILATION_RATE
        token_size_div = arch.TOKEN_SIZE_DIV
        token_size_mul = arch.TOKEN_SIZE_MUL
        # Allow reading optional NUM_TOKENS from checkpoint hparams (backward compatible)
        num_tokens = getattr(arch, 'NUM_TOKENS', None)
        if num_tokens is None or int(num_tokens) <= 0:
            num_tokens = int(((num_joints//10)*10) * (2**(token_size_mul)) / (2**down_t))
        else:
            num_tokens = int(num_tokens)
        self.num_tokens = num_tokens

        self.decoder = PoseSPDecoderV1(rot_type=rot_type,
                                       output_dim=6,
                                       output_emb_width=output_emb_width,
                                       down_t=down_t,
                                       width=width,
                                       depth=depth,
                                       token_size_div=token_size_div,
                                       num_tokens=num_tokens,
                                       dilation_growth_rate=dilation_growth_rate,
                                       num_joints=num_joints)
        self.quantizer = QuantizeEMAReset(nb_code, code_dim)
        self.load_weights(ckpt)

    def forward(self, logits):
        decode_feat = self.quantizer.dequantize_logits(logits)
        return self.decode_latent(decode_feat)

    def decode_latent(self, decode_feat):
        pose_out = self.decoder(decode_feat.permute(0,2,1))
        return pose_out['pred_pose_body_6d']

    def load_weights(self, ckpt):
        prepare_statedict(self.decoder, ckpt['net'], 'decoder', 'body_model')
        prepare_statedict(self.quantizer, ckpt['net'], 'quantizer', 'body_model')


class EncodeTokens(nn.Module):
    def __init__(self,
                 ckpt_path=''):
        super(EncodeTokens, self).__init__()

        ckpt = torch.load(ckpt_path, map_location='cpu')
        pretrained_hparams = ckpt['hparams']
        arch = pretrained_hparams.ARCH
        code_dim = arch.CODE_DIM
        nb_code = arch.NB_CODE
        output_emb_width = code_dim
        down_t = arch.DOWN_T
        width = arch.WIDTH
        depth = arch.DEPTH
        dilation_growth_rate = arch.DILATION_RATE
        token_size_mul = arch.TOKEN_SIZE_MUL
        # Mirror num_tokens logic for encoder interpolation if explicit override exists in checkpoint.
        explicit_num_tokens = getattr(arch, 'NUM_TOKENS', None)
        self.encoder = PoseSPEncoderV1(input_dim=6,
                                       output_emb_width=output_emb_width,
                                       down_t=down_t,
                                       width=width,
                                       depth=depth,
                                       token_size_mul=token_size_mul,
                                       dilation_growth_rate=dilation_growth_rate,
                                       target_tokens=int(explicit_num_tokens) if (explicit_num_tokens is not None and int(explicit_num_tokens) > 0) else None)
        self.quantizer = QuantizeEMAReset(nb_code, code_dim)

        self.load_weights(ckpt)

    def forward(self, x):
        # Encoder
        x_encoder = self.encoder(x)

        # Quantize
        x_encoder = self.quantizer.preprocess(x_encoder)
        code_idx = self.quantizer.quantize(x_encoder)

        return code_idx

    def load_weights(self, ckpt):
        prepare_statedict(self.encoder, ckpt['net'], 'encoder')
        prepare_statedict(self.quantizer, ckpt['net'], 'quantizer')
