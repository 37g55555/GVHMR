import torch
import pytorch_lightning as pl
from hydra.utils import instantiate
from hmr4d.utils.pylogger import Log
from hmr4d.configs import MainStore, builds

from hmr4d.utils.geo.hmr_cam import normalize_kp2d


class DemoPL(pl.LightningModule):
    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = instantiate(pipeline, _recursive_=False)

    @torch.no_grad()
    def predict(self, data, static_cam=False):
        """auto add batch dim
        data: {
            "length": int, or Torch.Tensor,
            "kp2d": (F, 3)
            "bbx_xys": (F, 3)
            "K_fullimg": (F, 3, 3)
            "cam_angvel": (F, 3)
            "f_imgseq": (F, 3, 256, 256)
        }

        """
        # ROPE inference
        batch = {
            "length": data["length"][None],
            "obs": normalize_kp2d(data["kp2d"], data["bbx_xys"])[None],
            "bbx_xys": data["bbx_xys"][None],
            "K_fullimg": data["K_fullimg"][None],
            "cam_angvel": data["cam_angvel"][None],
            "f_imgseq": data["f_imgseq"][None],
        }
        batch = {k: v.cuda() for k, v in batch.items()}
        outputs = self.pipeline.forward(batch, train=False, postproc=True, static_cam=static_cam)

        pred = {
            "smpl_params_global": {k: v[0] for k, v in outputs["pred_smpl_params_global"].items()},
            "smpl_params_incam": {k: v[0] for k, v in outputs["pred_smpl_params_incam"].items()},
            "K_fullimg": data["K_fullimg"],
            "net_outputs": outputs,  # intermediate outputs
        }
        return pred

    def load_pretrained_model(self, ckpt_path):
        """Load pretrained checkpoint, and assign each weight to the corresponding part."""
        Log.info(f"[PL-Trainer] Loading ckpt type: {ckpt_path}")

        state_dict = torch.load(ckpt_path, "cpu")["state_dict"]
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if len(missing) > 0:
            Log.warn(f"Missing keys: {missing}")
        if len(unexpected) > 0:
            Log.warn(f"Unexpected keys: {unexpected}")

    def load_masked_pretrained_model(self, ckpt_path):
        """Load a masked-pose checkpoint with the training checkpoint semantics."""
        Log.info(f"[PL-Trainer] Loading masked-pose ckpt: {ckpt_path}")

        if self.pipeline.masked_pose_branch is None:
            raise RuntimeError("Masked-pose checkpoint requested without MaskedPoseBranch")

        state_dict = torch.load(ckpt_path, "cpu")["state_dict"]
        masked_pose_prefix = "pipeline.masked_pose_branch."
        if not any(k.startswith(masked_pose_prefix) for k in state_dict):
            raise RuntimeError(f"MaskedPoseBranch state not found in checkpoint: {ckpt_path}")

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        real_missing = [k for k in missing if not k.startswith("pipeline.endecoder")]
        if len(real_missing) > 0 or len(unexpected) > 0:
            raise RuntimeError(
                f"Incompatible masked-pose checkpoint. Missing keys: {real_missing}; unexpected keys: {unexpected}"
            )


MainStore.store(name="gvhmr_pl_demo", node=builds(DemoPL, pipeline="${pipeline}"), group="model/gvhmr")
