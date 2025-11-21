from pathlib import Path
import secrets
from pyflim import layers
import torch
import torch.nn as nn
import numpy as np
import pyift.pyift as ift
from pytorch_lightning import LightningModule
from skimage.morphology import binary_dilation, binary_erosion, disk

from torchmetrics import MaxMetric
from src.modules.utils import f_beta_score, labnorm
from src.models.graph_weight_estimator import GraphWeightEstimator
from pytorch_lightning.trainer.states import RunningStage

class SchistoSegmentationModule(LightningModule):
    def __init__(
        self,
        use_gt: bool = False,
        arcw_id: int = 1,
        split: int = 3,
        graph_weight_estimator: dict = None,
        triplet_margin: float = 0.2,
        l2_reg_weight: float = 1e-5,
        dynamic_trees: dict = None,
        flim_model_path: str = None,
        flim_decoder: dict = None,
        seed_selection: dict = None,
        f_beta: float = 0.3,
        seed: int = 2021,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.rng = np.random.default_rng(seed)
        
        # Initialize model
        self.model = GraphWeightEstimator(
            **graph_weight_estimator
        )
        
        # Loss function
        self.loss_fn = nn.TripletMarginLoss(margin=triplet_margin)
        
        # Load FLIM model
        if flim_model_path is None:
            flim_model_path = f'models/flim/split{split}/flim_encoder_split{split}.pth'
        
        if not Path(flim_model_path).exists():
            raise FileNotFoundError(
                f"FLIM model not found at {flim_model_path}. Please provide a valid path."
            )
        
        self.flim_model = None 
        self.flim_model_path = flim_model_path
        
        self.flim_model = torch.load(
            self.flim_model_path, 
            weights_only=False,
        )
        
        self.flim_model.decoder = layers.FLIMAdaptiveDecoderLayer(
            1,
            device=str(self.device),
            **self.hparams.flim_decoder
        )
        self.flim_model.eval()

        # --- METRIC STORAGE ---
        # We initialize lists to store scores for std calculation
        self.val_mlp_scores = []
        self.test_mlp_scores = []
        self.test_lab_scores = []
        
        # import maxmetric from pytorch_lightning.metrics
        self.val_best_fbeta_mlp = MaxMetric()
        self.test_best_fbeta_mlp = MaxMetric()

    def _get_tags_and_run_name(self):
        # ... (Your existing code for tags remains unchanged) ...
        hparams = getattr(self, "hparams", None)
        if hparams is None:
            return [], "unnamed_run"

        tags = []
        run_name_parts = []
        
        stage = self.trainer.state.stage
        
        if stage == RunningStage.TRAINING:
            tags.append("train")
            run_name_parts.append("train")
        elif stage == RunningStage.VALIDATING:
            tags.append("val")
            run_name_parts.append("val")
        elif stage == RunningStage.TESTING:
            tags.append("test")
            run_name_parts.append("test")
        elif stage == RunningStage.PREDICTING:
            tags.append("predict")
            run_name_parts.append("predict")

        run_name_parts.append("schist_seg")
        tags.append("schisto_seg")

        if hparams.use_gt:
            tags.append("use_gt")
            run_name_parts.append("gt")
        else:
            tags.append("use_flim")
            run_name_parts.append("flim")

        tags.append(f"split_{hparams.split}")
        run_name_parts.append(f"s{hparams.split}")

        tags.append(f"arcw_{hparams.arcw_id}")
        run_name_parts.append(f"arcw{hparams.arcw_id}")

        gcfg = hparams.graph_weight_estimator
        tags.append(f"embed{gcfg.get('embed_dim', 'na')}")
        run_name_parts.append(f"e{gcfg.get('embed_dim', 'na')}")

        flim_decoder = hparams.flim_decoder
        if flim_decoder and "decoder_type" in flim_decoder:
            dec_type = flim_decoder["decoder_type"]
            tags.append(dec_type)
            run_name_parts.append(dec_type)

        tags.append(f"triplet_margin_{hparams.triplet_margin}")
        tags.append(f"l2_{hparams.l2_reg_weight}")
        run_name_parts.append(f"m{hparams.triplet_margin}")

        if "segmentation" in hparams.dynamic_trees:
            seg_cfg = hparams.dynamic_trees["segmentation"]
            seg_border = seg_cfg.get("border", 1)
            tags.append(f"border_{seg_border}")
            run_name_parts.append(f"b{seg_border}")

        optimizer_cfg = getattr(hparams, "optimizer", None)
        if optimizer_cfg and "class_path" in optimizer_cfg:
            opt_name = optimizer_cfg["class_path"].split(".")[-1].lower()
            tags.append(opt_name)
            run_name_parts.append(opt_name)
            opt_args = optimizer_cfg.get("init_args", {})
            if "lr" in opt_args:
                tags.append(f"lr_{opt_args['lr']}")
                run_name_parts.append(f"lr{opt_args['lr']}")
            if "weight_decay" in opt_args:
                tags.append(f"wd_{opt_args['weight_decay']}")

        scheduler_cfg = getattr(hparams, "scheduler", None)
        if scheduler_cfg and "class_path" in scheduler_cfg:
            sched_name = scheduler_cfg["class_path"].split(".")[-1].lower()
            tags.append(sched_name)
            run_name_parts.append(sched_name)

        run_name = "_".join(run_name_parts)

        return tags, run_name
    
    def setup(self, stage):
        tags, run_name = self._get_tags_and_run_name()
        run_name += f"_{secrets.randbits(24)}"
        if hasattr(self.logger, 'experiment'):
            self.trainer.logger.experiment.tags = tuple(set(self.trainer.logger.experiment.tags).union(set(tags)))
            self.trainer.logger.experiment.name = run_name
        
    def forward(self, x):
        return self.model(x)
    
    def random_select(self, mask, percentage=0.3):
        mask = mask.astype(bool)
        indices = np.flatnonzero(mask)
        num_select = int(len(indices) * percentage)
        selected = self.rng.choice(indices, num_select, replace=False)
        new_mask = np.zeros_like(mask, dtype=np.uint8)
        new_mask.flat[selected] = 1
        return new_mask
    
    def prepare_seeds(self, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        seed_config = self.hparams.seed_selection

        eroded = binary_erosion(
            mask, footprint=disk(seed_config["erosion_disk_radius"])
        )
        dilated = binary_dilation(
            mask, footprint=disk(seed_config["dilation_disk_radius"])
        )

        seeds_in = self.random_select(
            eroded, percentage=seed_config["in_percentage"]
        )
        seeds_out = self.random_select(
            1 - dilated, percentage=seed_config["out_percentage"]
        )

        return seeds_in.astype(np.int32), seeds_out.astype(np.int32)

    def training_step(self, batch, batch_idx):
        # ... (Your existing training logic) ...
        x = batch["image"]
        y = batch["label"]
        y[y > 0] = 1

        if y.max() == 0:
            return None

        y = y[0].cpu().numpy()
        representation = self.model(x)
        seeds_in, seeds_out = self.prepare_seeds(y)

        seeds_in_mimg = ift.CreateImageFromNumPy(seeds_in, is3D=False)
        seeds_out_mimg = ift.CreateImageFromNumPy(seeds_out, is3D=False)
        res_mimg = ift.CreateMImageFromNumPy(
            np.ascontiguousarray(
                representation.detach().cpu().numpy().astype(np.float32)
            )
        )

        if self.hparams.arcw_id <= 0:
            res = ift.DynamicTrees(res_mimg, seeds_in_mimg, seeds_out_mimg)
        else:
            res = ift.myDynamicTreeInOut(
                res_mimg, seeds_in_mimg, seeds_out_mimg, self.hparams.arcw_id
            )
        res = res.AsNumPy()

        x_correct_1 = np.argwhere((res == 1) & (y == 1))
        x_correct_0 = np.argwhere((res == 0) & (y == 0))

        if len(x_correct_1) == 0 or len(x_correct_0) == 0:
            return None

        self.rng.shuffle(x_correct_0)
        x_correct_0 = x_correct_0[: x_correct_1.shape[0], :]

        self.rng.shuffle(x_correct_1)
        half = x_correct_1.shape[0] // 2

        if half == 0:
            return None

        anchors_idx = x_correct_1[:half]
        positives_idx = x_correct_1[half : 2 * half]

        anchor = representation[anchors_idx[:, 0], anchors_idx[:, 1]]
        positive = representation[positives_idx[:, 0], positives_idx[:, 1]]
        negative = representation[x_correct_0[:half, 0], x_correct_0[:half, 1]]

        loss = self.loss_fn(anchor, positive, negative)

        l2_reg = sum(torch.norm(p, 2) for p in self.model.fc.parameters())
        loss += self.hparams.l2_reg_weight * l2_reg
        
        batch_size = x.shape[0]
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss
    
    def test_lab(self, batch):
        x = batch["image"]
        y = batch["label"]
        y[y > 0] = 1
        y = y[0].cpu().numpy()
        
        saliency = self.get_saliency(x, y)
        
        beta = self.hparams.f_beta
        seg_config = self.hparams.dynamic_trees["segmentation"]
        
        orig_mimg = ift.CreateMImageFromNumPy(
            np.ascontiguousarray(
                x.squeeze()
                .permute((1, 2, 0))
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        )
        if self.hparams.arcw_id <= 0:
            res1 = ift.SMansoniDelineation(
                orig_mimg,
                saliency,
                seg_config["border"],
                seg_config["min_comp_area"],
                seg_config["max_comp_area"],
                seg_config["saliency_erosion"],
                seg_config["saliency_dilation"]
            )
        else:
            res1 = ift.mySMansoniDelineation(
                orig_mimg,
                saliency,
                seg_config["border"],
                seg_config["min_comp_area"],
                seg_config["max_comp_area"],
                seg_config["saliency_erosion"],
                seg_config["saliency_dilation"],
                self.hparams.arcw_id,
            )
        res1 = res1.AsNumPy()
        loss_lab = f_beta_score(y, res1, beta2=beta)
        
        if self.trainer.state.stage == RunningStage.VALIDATING:
            stage = 'val'
        else:
            stage = 'test'
            # Accumulate Test LAB score
            self.test_lab_scores.append(loss_lab)

        batch_size = x.shape[0]
        self.log(f'{stage}/fbeta_lab', loss_lab, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)

    def get_saliency(self, x, y):
        if not self.hparams.use_gt:
            x_t = labnorm(x)
            y_t = self.flim_model.forward(x_t, decoder_layer=0)
            y_t = y_t.squeeze().detach().cpu().numpy()
            saliency = ift.CreateImageFromNumPy(
                y_t.astype(np.int32), is3D=False
            )
        else:
            saliency = ift.CreateImageFromNumPy(
                (y * 255).astype(np.int32), is3D=False
            )
        return saliency
    
    def validation_step(self, batch, batch_idx):
        x = batch["image"].to(self.device)
        y = batch["label"]
        y[y > 0] = 1
        y = y[0].cpu().numpy()

        representation = self.model(x)
        saliency = self.get_saliency(x, y)

        res_mimg = ift.CreateMImageFromNumPy(
            np.ascontiguousarray(
                representation.detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        )
        
        seg_config = self.hparams.dynamic_trees["segmentation"]

        if self.hparams.arcw_id <= 0:
            res = ift.SMansoniDelineation(
                res_mimg,
                saliency,
                seg_config["border"],
                seg_config["min_comp_area"],
                seg_config["max_comp_area"],
                seg_config["saliency_erosion"],
                seg_config["saliency_dilation"]
            )
        else:
            res = ift.mySMansoniDelineation(
                res_mimg,
                saliency,
                seg_config["border"],
                seg_config["min_comp_area"],
                seg_config["max_comp_area"],
                seg_config["saliency_erosion"],
                seg_config["saliency_dilation"],
                self.hparams.arcw_id,
            )

        res = res.AsNumPy()
        
        loss_mlp = f_beta_score(y, res, beta2=self.hparams.f_beta)
        
        if self.trainer.state.stage == RunningStage.VALIDATING:
            stage = 'val'
            # Accumulate Validation score
            self.val_mlp_scores.append(loss_mlp)
        else:
            stage = 'test'
            # Accumulate Test MLP score
            self.test_mlp_scores.append(loss_mlp)
        
        # This logs the MEAN
        batch_size = x.shape[0]
        self.log(f'{stage}/fbeta_mlp', loss_mlp, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        
        return loss_mlp

    def on_validation_epoch_end(self):
        # Calculate and log Standard Deviation for Validation
        if len(self.val_mlp_scores) > 1:
            scores_tensor = torch.tensor(self.val_mlp_scores, dtype=torch.float)
            std_dev = scores_tensor.std()
            self.log('val/fbeta_mlp_std', std_dev, prog_bar=True)
        
        self.val_best_fbeta_mlp.update(torch.tensor(np.mean(self.val_mlp_scores), dtype=torch.float))
        self.log('val/best_fbeta_mlp', self.val_best_fbeta_mlp.compute(), prog_bar=True)
        
        # Clear list for next epoch
        self.val_mlp_scores.clear()

    def on_test_epoch_end(self):
        # Calculate and log Standard Deviation for Test MLP
        if len(self.test_mlp_scores) > 1:
            mlp_tensor = torch.tensor(self.test_mlp_scores, dtype=torch.float)
            mlp_std = mlp_tensor.std()
            self.log('test/fbeta_mlp_std', mlp_std, prog_bar=True)
            
        # Calculate and log Standard Deviation for Test LAB
        if len(self.test_lab_scores) > 1:
            lab_tensor = torch.tensor(self.test_lab_scores, dtype=torch.float)
            lab_std = lab_tensor.std()
            self.log('test/fbeta_lab_std', lab_std, prog_bar=True)

        # Clear lists
        self.test_mlp_scores.clear()
        self.test_lab_scores.clear()
        
        self.test_best_fbeta_mlp.update(torch.tensor(np.mean(self.test_mlp_scores), dtype=torch.float))
        self.log('test/best_fbeta_mlp', self.test_best_fbeta_mlp.compute(), prog_bar=True)
        
    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)
        self.test_lab(batch)