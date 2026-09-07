import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from hmr4d.model.gvhmr.gvhmr_pl import GvhmrPL
from hmr4d.configs import MainStore, builds
from hmr4d.utils.pylogger import Log

from ..utils.optim_utils import parse_scheduler


OmegaConf.register_new_resolver("eval", eval)


class MaskTransformerModule(GvhmrPL):
    def __init__(
        self,
        pipeline,
        optim,
        ignored_weights_prefix=["smplx", "pipeline.endecoder"],
    ):
        super().__init__(
            pipeline=pipeline,
            optimizer=None,
            scheduler_cfg=None,
            ignored_weights_prefix=ignored_weights_prefix,
        )
        self.optim = optim

        self._freeze_stages()

    def _freeze_stages(self):
        self.pipeline.denoiser3d.eval()
        for param in self.pipeline.denoiser3d.parameters():
            param.requires_grad = False

        self.pipeline.masked_pose_branch.pose_tokenizer.eval()
        for param in self.pipeline.masked_pose_branch.pose_tokenizer.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        return self

    def configure_optimizers(self):
        params = []
        for v in self.pipeline.parameters():
            if v.requires_grad:
                params.append(v)
        optimizer = torch.optim.AdamW(params, lr=self.optim.lr, weight_decay=self.optim.weight_decay)
        scheduler = parse_scheduler(self.optim.scheduler, optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def load_pretrained_model(self, ckpt_path):
        if self.pipeline.masked_pose_branch.smoother is None:
            return super().load_pretrained_model(ckpt_path)

        Log.info(f"[PL-Trainer] Loading smoother-stage initialization: {ckpt_path}")
        state_dict = torch.load(ckpt_path, "cpu")["state_dict"]
        missing, unexpected = self.load_state_dict(state_dict, strict=False)

        smoother_prefix = "pipeline.masked_pose_branch.smoother."
        real_missing = []
        for key in missing:
            ignored_when_saving = any(
                key.startswith(ignored_prefix) for ignored_prefix in self.ignored_weights_prefix
            )
            if not ignored_when_saving and not key.startswith(smoother_prefix):
                real_missing.append(key)

        if real_missing or unexpected:
            raise RuntimeError(
                f"Incompatible smoother-stage initialization. "
                f"Missing keys: {real_missing}; unexpected keys: {unexpected}"
            )


mask_transformer_module = builds(
    MaskTransformerModule,
    pipeline="${pipeline}",
    optim={
        "lr": 1e-5,
        "lr_ratio": 0.1,
        "weight_decay": 1e-4,
        "warmup_steps": 2_000,
        "scheduler": {
            "name": "SequentialLR",
            "interval": "step",
            "milestones": ["${model.optim.warmup_steps}"],
            "schedulers": [
                {
                    "name": "LinearLR",
                    "args": {
                        "start_factor": 0.01,
                        "end_factor": 1.0,
                        "total_iters": "${model.optim.warmup_steps}",
                    },
                },
                {
                    "name": "CosineAnnealingLR",
                    "args": {
                        "T_max": "${eval:${pl_trainer.max_steps} - ${model.optim.warmup_steps}}",
                        "eta_min": "${eval:${model.optim.lr} * ${model.optim.lr_ratio}}",
                    },
                },
            ],
        },
    },
    populate_full_signature=True,
)
MainStore.store(name="mask_transformer_module", node=mask_transformer_module, group="model/gvhmr")

model_checkpoint = builds(
    ModelCheckpoint,
    dirpath="${output_dir}/checkpoints",
    every_n_train_steps=1_000,
    save_last=True,
    save_weights_only=False,
    populate_full_signature=True,
)
MainStore.store(name="every1000s", node=model_checkpoint, group="callbacks/model_checkpoint")
