import warnings
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import os

class FixedModelCheckpoint(ModelCheckpoint):
    def __init__(
        self,
        *args,
        save_torchscript: bool = False,
        **kwargs
    ):
        """
        Args:
            save_torchscript: If True, also exports the model as a TorchScript
                file (.pth) that can be loaded without the original class.
        """
        super().__init__(*args, **kwargs)
        self.save_torchscript = save_torchscript

    def _save_checkpoint(self, trainer, filepath: str) -> None:
        """Override checkpoint saving to optionally export a TorchScript model."""

        # First, create the checkpoint dict (not yet written to disk)
        checkpoint = trainer._checkpoint_connector.dump_checkpoint(
            weights_only=self.save_weights_only
        )

        # Fix hyper_parameters for Lightning 2.x compatibility
        if "hyper_parameters" in checkpoint:
            hparams = checkpoint["hyper_parameters"]
            if "_class_path" in hparams:
                hparams["class_path"] = hparams.pop("_class_path")

            if len(hparams) > 2:
                init_args = {
                    k: v
                    for k, v in hparams.items()
                    if k not in ["class_path", "_instantiator"]
                }
                checkpoint["hyper_parameters"] = {
                    "class_path": hparams["class_path"],
                    "init_args": init_args,
                    "_instantiator": hparams["_instantiator"],
                }

        # Save the corrected checkpoint
        trainer.save_checkpoint(filepath, weights_only=self.save_weights_only)
        torch.save(checkpoint, filepath)

        # ✅ Save TorchScript version if requested
        if self.save_torchscript:
            weights_filepath = filepath.rsplit(".", 1)[0] + ".pth"
            model = trainer.lightning_module

            try:
                scripted = torch.jit.script(model)
            except Exception:
                # Fallback to tracing if scripting fails
                example_input = getattr(model, "example_input_array", None)
                if example_input is None:
                    raise RuntimeError(
                        "TorchScript export failed: model has no example_input_array for tracing."
                    )
                example_input = example_input.to(model.device)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    scripted = torch.jit.trace(model, example_input)

            scripted.save(weights_filepath)

        # Update internal bookkeeping
        self._last_global_step_saved = trainer.global_step
        self._last_checkpoint_saved = filepath

        # Clean up old TorchScript files, if using save_top_k
        if self.save_torchscript and self.save_top_k > 0:
            self._remove_old_torchscript_files()

        # Notify loggers
        if trainer.is_global_zero:
            for logger in trainer.loggers:
                logger.after_save_checkpoint(self)

    def _remove_old_torchscript_files(self):
        """Remove TorchScript files that no longer have a matching .ckpt."""
        existing_ckpts = set(self.best_k_models.keys())
        dirpath = self.dirpath

        for filename in os.listdir(dirpath):
            if filename.endswith(".pth"):
                ckpt_equiv = filename.rsplit(".", 1)[0] + ".ckpt"
                ckpt_path = os.path.join(dirpath, ckpt_equiv)
                pth_path = os.path.join(dirpath, filename)
                if ckpt_path not in existing_ckpts and not os.path.exists(ckpt_path):
                    try:
                        os.remove(pth_path)
                    except OSError:
                        pass
