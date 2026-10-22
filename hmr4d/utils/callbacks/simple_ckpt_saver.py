from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks.checkpoint import Checkpoint
from pytorch_lightning.utilities import rank_zero_only

from hmr4d.utils.pylogger import Log
from hmr4d.configs import MainStore, builds


class SimpleCkptSaver(Checkpoint):
    """
    This callback runs at the end of each training epoch.
    Check {every_n_epochs} and save at most {save_top_k} model if it is time.
    """

    def __init__(
        self,
        output_dir,
        filename="e{epoch:03d}-s{step:06d}.ckpt",
        save_top_k=1,
        every_n_epochs=1,
        save_last=None,
        save_weights_only=True,
    ):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.filename = filename
        self.save_top_k = save_top_k
        self.every_n_epochs = every_n_epochs
        self.save_last = save_last
        self.save_weights_only = save_weights_only

        # Setup output dir
        if rank_zero_only.rank == 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            Log.info(f"[Simple Ckpt Saver]: Save to `{self.output_dir}'")

    def on_train_epoch_end(self, trainer, pl_module):
        """Save a checkpoint at the end of the training epoch."""
        if self.every_n_epochs >= 1 and (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            if self.save_top_k == 0:
                return

            # Current saved ckpts in the output_dir
            model_paths = []
            for p in sorted(list(self.output_dir.glob("*.ckpt"))):
                model_paths.append(p)
            model_to_remove = model_paths[0] if len(model_paths) >= self.save_top_k else None
            model_to_remove = trainer.strategy.broadcast(model_to_remove)

            # Save cureent checkpoint
            filepath = self.output_dir / self.filename.format(epoch=trainer.current_epoch, step=trainer.global_step)
            trainer.save_checkpoint(filepath, weights_only=self.save_weights_only)

            # Remove the earliest checkpoint
            if model_to_remove:
                trainer.strategy.remove_checkpoint(model_to_remove)


group_name = "callbacks/simple_ckpt_saver"
base = builds(SimpleCkptSaver, output_dir="${output_dir}/checkpoints/", populate_full_signature=True)
MainStore.store(name="base", node=base, group=group_name)
MainStore.store(name="every1e", node=base, group=group_name)
MainStore.store(name="every2e", node=base(every_n_epochs=2), group=group_name)
MainStore.store(name="every5e", node=base(every_n_epochs=5), group=group_name)
MainStore.store(name="every5e_top100", node=base(every_n_epochs=5, save_top_k=100), group=group_name)
MainStore.store(name="every10e", node=base(every_n_epochs=10), group=group_name)
MainStore.store(name="every10e_top100", node=base(every_n_epochs=10, save_top_k=100), group=group_name)
