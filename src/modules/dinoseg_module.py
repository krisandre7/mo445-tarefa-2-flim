import os
from pathlib import Path
import gdown
import zipfile
import torch
import torch.nn as nn
import pytorch_lightning as L
from typing import Any

from torchmetrics import MeanMetric, MaxMetric
from torchmetrics.classification import JaccardIndex
from torchmetrics.segmentation import DiceScore

from src.models.dpt import DPT


class SegDINOModule(L.LightningModule):
    def __init__(
        self,
        ckpt_dir: str = "checkpoints/dinov3",
        ckpt_google_drive_id: str = "1dxA94EdTadIlhkJxLZFQeq8yqfgAfC2r",
        dino_size: str = "base",
        num_classes: int = 1,
        task: str = "binary",  # added to support torchmetrics setup
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        # --- Load Backbone ---
        self.example_input_array = torch.randn(1, 3, 256, 256, device=self.device)

        # baixa e extrai se necessário
        if not os.path.exists(self.hparams.ckpt_dir) or not os.listdir(self.hparams.ckpt_dir):
            downloaded = False
            filename = f"{Path(self.hparams.ckpt_dir).name}.zip"
            if not os.path.exists(filename):
                print(f"Downloading DINOv3 checkpoints to {self.hparams.ckpt_dir}...")

                if self.hparams.ckpt_google_drive_id is None or self.hparams.ckpt_google_drive_id == "":
                    raise ValueError("ckpt_google_drive_id must be provided to download the dataset.")

                filename = gdown.download(id=self.hparams.ckpt_google_drive_id, quiet=False)
                downloaded = True
            else:
                print(f"Found existing zip file {filename}, skipping download.")

            os.makedirs(os.path.dirname(self.hparams.ckpt_dir), exist_ok=True)

            with zipfile.ZipFile(filename, "r") as zip_ref:
                zip_ref.extractall(Path(self.hparams.ckpt_dir).parent)
                if downloaded:
                    os.remove(filename)

        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        if dino_size == "base":
            checkpoint_name = "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
            ckpt_path = os.path.join(ckpt_dir, checkpoint_name)
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

            backbone = torch.hub.load(
                "facebookresearch/dinov3",
                "dinov3_vitb16",
                source="github",
                pretrained=True,
                weights=ckpt_path,
            )
        elif dino_size == "small":
            checkpoint_name = "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
            ckpt_path = os.path.join(ckpt_dir, checkpoint_name)
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

            backbone = torch.hub.load(
                "facebookresearch/dinov3",
                "dinov3_vits16",
                source="github",
                weights=ckpt_path,
                pretrained=True,
            )
        else:
            raise ValueError(f"Unsupported DINO size: {dino_size}")

        # --- Initialize DPT Head ---
        self.model = DPT(nclass=num_classes, backbone=backbone, freeze_backbone=freeze_backbone)

        # --- Loss Function ---
        self.criterion = nn.BCEWithLogitsLoss()

        # --- TorchMetrics additions (from SegmentationModule) ---
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.train_iou = JaccardIndex(task=task, num_classes=num_classes)
        self.val_iou = JaccardIndex(task=task, num_classes=num_classes)

        self.train_dice = DiceScore(num_classes=num_classes)
        self.val_dice = DiceScore(num_classes=num_classes)

        self.val_iou_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def _shared_step(self, batch: Any):
        x = batch['image']
        y = batch['label']
        
        if x.max() > 1.0:
            y = y / 255.0  # normalize to [0, 1] if not already

        logits = self(x)

        if self.hparams.num_classes == 1:
            y = y.float()
        else:
            y = y.long()
            
        if y.ndim == logits.ndim - 1:
            y = y.unsqueeze(1)

        loss = self.criterion(logits, y)
        return loss, logits, y

    # ---------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        loss, logits, y = self._shared_step(batch)

        preds = torch.sigmoid(logits) if self.hparams.num_classes == 1 else torch.softmax(logits, dim=1)
        preds = (preds > 0.5).long() if self.hparams.num_classes == 1 else torch.argmax(preds, dim=1)

        # Metrics update
        self.train_loss(loss)
        self.train_iou(preds, y.long())
        self.train_dice(preds, y.long())

        # Logging
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/iou", self.train_iou, on_step=False, on_epoch=True)
        self.log("train/dice", self.train_dice, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, y = self._shared_step(batch)

        preds = torch.sigmoid(logits) if self.hparams.num_classes == 1 else torch.softmax(logits, dim=1)
        preds = (preds > 0.5).long() if self.hparams.num_classes == 1 else torch.argmax(preds, dim=1)

        # Metric updates
        self.val_loss(loss)
        self.val_iou(preds, y.long())
        self.val_dice(preds, y.long())

        self.log("val/loss", self.val_loss, on_epoch=True, prog_bar=True)
        self.log("val/iou", self.val_iou, on_epoch=True)
        self.log("val/dice", self.val_dice, on_epoch=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        loss, logits, y = self._shared_step(batch)

        preds = torch.sigmoid(logits) if self.hparams.num_classes == 1 else torch.softmax(logits, dim=1)
        preds = (preds > 0.5).long() if self.hparams.num_classes == 1 else torch.argmax(preds, dim=1)

        # Metric updates
        self.val_loss(loss)
        self.val_iou(preds, y.long())
        self.val_dice(preds, y.long())

        self.log("test/loss", self.val_loss, on_epoch=True, prog_bar=True)
        self.log("test/iou", self.val_iou, on_epoch=True)
        self.log("test/dice", self.val_dice, on_epoch=True)

        return loss
