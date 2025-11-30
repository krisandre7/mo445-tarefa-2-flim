import os
import pickle
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import pytorch_lightning as pl
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

# Constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PATCH_SIZE = 16
IMAGE_SIZE = 1024

def lab_to_rgb_tensor(lab_image: torch.Tensor) -> torch.Tensor:
    """Convert LAB tensor to RGB tensor.
    
    Args:
        lab_image: Tensor of shape (C, H, W) or (B, C, H, W) in LAB format
                   (Assumes tensor is on CPU or will be moved to CPU for cv2 conversion)
    Returns:
        RGB tensor in the same shape (B, C, H, W), normalized to [0, 1]
    """
    # Handle missing batch dim
    squeeze_dim = False
    if lab_image.ndim == 3:
        lab_image = lab_image.unsqueeze(0)
        squeeze_dim = True
    
    batch_size = lab_image.shape[0]
    rgb_batch = []
    
    # We must iterate because cv2 is CPU/numpy only and doesn't handle N-dim batches natively for color conversion
    # However, this is done per batch, so it's reasonably efficient.
    lab_cpu = lab_image.cpu()
    
    for i in range(batch_size):
        # (C, H, W) -> (H, W, C)
        lab_np = lab_cpu[i].permute(1, 2, 0).numpy()
        
        # Convert LAB to RGB using OpenCV (handles the specific LAB scaling used in the dataset)
        rgb_np = cv2.cvtColor(lab_np, cv2.COLOR_LAB2RGB)
        
        # (H, W, C) -> (C, H, W)
        rgb_tensor = torch.from_numpy(rgb_np).float().permute(2, 0, 1)
        rgb_batch.append(rgb_tensor)
    
    result = torch.stack(rgb_batch)
    
    return result.squeeze(0) if squeeze_dim else result

MODEL_DINOV3_VITS = "dinov3_vits16"
MODEL_DINOV3_VITSP = "dinov3_vits16plus"
MODEL_DINOV3_VITB = "dinov3_vitb16"
MODEL_DINOV3_VITL = "dinov3_vitl16"
MODEL_DINOV3_VITHP = "dinov3_vith16plus"
MODEL_DINOV3_VIT7B = "dinov3_vit7b16"
class DinoForegroundSegmentor(pl.LightningModule):
    def __init__(
        self,
        dinov3_location: str = "facebookresearch/dinov3",
        model_name: str = "dinov3_vitl16",
        classifier_path: str = None,
        reg_c: float = 1e-3,
        threshold: float = 0.3,
        image_size: int = 400,
    ):
        """
        Args:
            dinov3_location: Path to DINOv3 repo or 'facebookresearch/dinov3'.
            model_name: Name of the DINOv3 model architecture.
            classifier_path: Path to a pickled LogisticRegression model.
            reg_c: Regularization strength for Logistic Regression.
            threshold: Probability threshold for binary segmentation.
            image_size: Input image size.
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.dinov3_location = dinov3_location
        self.model_name = model_name
        self.classifier_path = classifier_path
        self.reg_c = reg_c
        self.threshold = threshold
        self.image_size = image_size
        
        # Load DINOv3 Backbone
        self.feature_extractor = self._load_dinov3()
        self.feature_extractor.eval()
        
        self.n_layers = self._get_num_layers(model_name)
        
        # Classifier
        self.clf = None
        self._try_load_classifier()

        # Patch quantization filter (conv2d acts as a sliding window average)
        # We use this to downsample the high-res ground truth mask to the DINO patch grid
        self.patch_quant_filter = nn.Conv2d(
            1, 1, PATCH_SIZE, stride=PATCH_SIZE, bias=False
        )
        self.patch_quant_filter.weight.data.fill_(1.0 / (PATCH_SIZE * PATCH_SIZE))
        self.patch_quant_filter.requires_grad_(False)

    def _load_dinov3(self):
        WEIGHTS_MAP = {
            MODEL_DINOV3_VITS: "checkpoints/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
            MODEL_DINOV3_VITL: "checkpoints/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
            MODEL_DINOV3_VITB: "checkpoints/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
        }
        source = "local" if os.path.exists(self.dinov3_location) and "github" not in self.dinov3_location else "github"
        
        weights = None
        if source == "local" and self.model_name in WEIGHTS_MAP:
            if os.path.exists(WEIGHTS_MAP[self.model_name]):
                weights = WEIGHTS_MAP[self.model_name]

        model = torch.hub.load(
            repo_or_dir=self.dinov3_location,
            model=self.model_name,
            source=source,
            pretrained=True if weights is None else False,
        )
        
        if weights is not None:
             state_dict = torch.load(weights, map_location="cpu")
             model.load_state_dict(state_dict)
             
        return model

    def _get_num_layers(self, model_name):
        MODEL_TO_NUM_LAYERS = {
            MODEL_DINOV3_VITS: 12,
            MODEL_DINOV3_VITSP: 12,
            MODEL_DINOV3_VITB: 12,
            MODEL_DINOV3_VITL: 24,
            MODEL_DINOV3_VITHP: 32,
            MODEL_DINOV3_VIT7B: 40,
        }

        return MODEL_TO_NUM_LAYERS.get(model_name, 12)

    def _try_load_classifier(self):
        if self.classifier_path and os.path.exists(self.classifier_path):
            print(f"Loading Logistic Regressor from {self.classifier_path}")
            with open(self.classifier_path, "rb") as f:
                self.clf = pickle.load(f)
        else:
            print("No classifier found. It will be trained upon calling trainer.fit()")
            self.clf = None
            
        # create automatic name if not provided
        if self.classifier_path is None:
            self.classifier_path = f"checkpoints/dinov3_foreground_segmentor/logreg_{self.model_name}_c{self.reg_c:.0e}_is{self.image_size}.pkl"

    def forward(self, x_rgb):
        """
        Args:
            x_rgb: (B, 3, H, W) RGB tensor, normalized [0,1].
        Returns:
            prob_mask: (B, H, W) Probability map (0-1).
        """
        B, C, H, W = x_rgb.shape
        
        # 1. Normalize and Resize for DINO (Batch operation)
        x_norm = TF.normalize(x_rgb, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        x_norm = TF.resize(x_norm, [self.image_size, self.image_size], antialias=True)
        
        # 2. Extract features
        with torch.no_grad():
            # feats: List of tensors. We take the last one.
            # Shape: (B, dim, h_patch, w_patch)
            feats = self.feature_extractor.get_intermediate_layers(
                x_norm, n=range(self.n_layers), reshape=True, norm=True
            )[-1]
            
            dim = feats.shape[1]
            h_patch, w_patch = feats.shape[2], feats.shape[3]
            
            # Rearrange to (B * total_patches, dim) for the scikit-learn classifier
            features_flat = feats.permute(0, 2, 3, 1).reshape(-1, dim)
            features_cpu = features_flat.cpu().numpy()

        if self.clf is None:
            raise RuntimeError("Classifier not trained. Run trainer.fit() first.")

        # 3. Classifier Inference (CPU)
        # output: (N_samples, 2). We take column 1 (positive class).
        probs = self.clf.predict_proba(features_cpu)[:, 1]
        
        # 4. Reshape and Upsample (GPU)
        # Reshape back to batch: (B, h_patch, w_patch)
        probs_batch = probs.reshape(B, h_patch, w_patch)
        
        # Move back to GPU for interpolation
        probs_tensor = torch.from_numpy(probs_batch).unsqueeze(1).float().to(self.device) # (B, 1, h_p, w_p)
        
        # Bilinear upsampling to original image size
        probs_full = torch.nn.functional.interpolate(
            probs_tensor, size=(H, W), mode='bilinear', align_corners=False
        )
        
        return probs_full.squeeze(1) # (B, H, W)

    def on_fit_start(self):
        """
        Collects training data features and trains the Logistic Regressor if not present.
        Handles batch processing for faster feature extraction.
        """
        if self.clf is not None:
            print("Classifier already loaded. Skipping training.")
            return

        print("Starting Logistic Regression training...")
        
        datamodule = self.trainer.datamodule
        if datamodule is None:
            raise ValueError("SchistoDataModule must be attached to the Trainer.")
            
        train_loader = datamodule.train_dataloader()
        
        xs = []
        ys = []

        self.feature_extractor.to(self.device)

        with torch.no_grad():
            for batch in tqdm(train_loader, desc="Extracting DINO features"):
                # 1. Prepare Data
                # image shape: (B, C, H, W) in LAB
                # mask shape: (B, H, W) or (B, 1, H, W)
                images_lab = batch["image"]
                masks = batch["label"]

                if masks.ndim == 3:
                    masks = masks.unsqueeze(1) # Ensure (B, 1, H, W)

                # Move masks to GPU immediately
                masks = masks.to(self.device)

                # 2. Process Masks (Batch)
                # Resize masks to training resolution
                masks_resized = TF.resize(masks, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.NEAREST)
                masks_norm = masks_resized.float() / 255.0
                
                # Quantize masks to patch grid: (B, 1, H, W) -> (B, 1, H_p, W_p)
                masks_quantized = self.patch_quant_filter(masks_norm)
                # Flatten: (B, 1, H_p, W_p) -> (B * H_p * W_p)
                ys.append(masks_quantized.view(-1).cpu())

                # 3. Process Images (Batch)
                # Convert LAB to RGB (CPU op inside function, returns tensor)
                images_rgb = lab_to_rgb_tensor(images_lab).to(self.device)
                
                # Normalize and Resize
                images_norm = TF.normalize(images_rgb, mean=IMAGENET_MEAN, std=IMAGENET_STD)
                images_norm = TF.resize(images_norm, [self.image_size, self.image_size], antialias=True)
                
                # 4. Extract Features
                # feats: (B, dim, H_p, W_p)
                feats = self.feature_extractor.get_intermediate_layers(
                    images_norm, n=range(self.n_layers), reshape=True, norm=True
                )[-1]
                
                dim = feats.shape[1]
                # Flatten: (B, dim, H_p, W_p) -> (B, H_p, W_p, dim) -> (B*H_p*W_p, dim)
                feat_vec = feats.permute(0, 2, 3, 1).reshape(-1, dim)
                xs.append(feat_vec.cpu())

        # Concatenate all batches
        xs = torch.cat(xs)
        ys = torch.cat(ys)
        
        # Filter training samples (confident background or foreground only)
        # This keeps the design matrix smaller and cleaner
        idx = (ys < 0.01) | (ys > 0.99)
        xs = xs[idx]
        ys = ys[idx]
        
        print(f"Training LR on {len(ys)} patches. Design matrix: {xs.shape}")
        
        self.clf = LogisticRegression(
            random_state=0, C=self.reg_c, max_iter=1000, verbose=1, n_jobs=-1
        )
        self.clf.fit(xs.numpy(), (ys > 0.5).long().numpy())
        
        if self.classifier_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.classifier_path)), exist_ok=True)
            with open(self.classifier_path, "wb") as f:
                pickle.dump(self.clf, f)
            print(f"Classifier saved to {self.classifier_path}")

    def training_step(self, batch, batch_idx):
        return None

    def configure_optimizers(self):
        return []

    def test_step(self, batch, batch_idx):
        """
        Performs inference on a batch and logs metrics.
        """
        images_lab = batch["image"] # (B, 3, H, W)
        masks_gt = batch["label"]   # (B, H, W) or (B, 1, H, W)
        
        if masks_gt.ndim == 4:
            masks_gt = masks_gt.squeeze(1) # Ensure (B, H, W)
            
        # 1. Convert to RGB (Batch)
        images_rgb = lab_to_rgb_tensor(images_lab).to(self.device)
        
        # 2. Forward Pass (Batch) -> Returns (B, H, W) probabilities
        pred_probs = self(images_rgb)
        
        # 3. Metrics (Batch-wise vectorization)
        pred_bin = (pred_probs >= self.threshold).float()
        masks_gt_bin = (masks_gt > 0).float()
        
        # Compute Dice per image in batch to average correctly
        # flatten H,W dimensions -> (B, -1)
        pred_flat = pred_bin.view(pred_bin.shape[0], -1)
        gt_flat = masks_gt_bin.view(masks_gt_bin.shape[0], -1)
        
        intersection = (pred_flat * gt_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + gt_flat.sum(dim=1)
        
        # Handle division by zero for empty masks
        dice_scores = (2. * intersection) / (union + 1e-8)
        mean_dice = dice_scores.mean()
        
        self.log("test/dice", mean_dice, on_step=True, on_epoch=True, batch_size=images_lab.shape[0])
        return {"test_loss": 1 - mean_dice}

    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx)