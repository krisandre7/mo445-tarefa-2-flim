import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
import pickle
from typing import Tuple, Union
from pathlib import Path

# ===========================
# Constants
# ===========================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

MODEL_DINOV3_VITS = "dinov3_vits16"
MODEL_DINOV3_VITB = "dinov3_vitb16"
MODEL_DINOV3_VITL = "dinov3_vitl16"

# You might need to adjust paths if running from different directories,
# or pass the dinov3_repo_location to the loader.
WEIGHTS_MAP = {
    MODEL_DINOV3_VITS: "checkpoints/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    MODEL_DINOV3_VITL: "checkpoints/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    MODEL_DINOV3_VITB: "checkpoints/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
}

MODEL_TO_NUM_LAYERS = {
    MODEL_DINOV3_VITS: 12,
    MODEL_DINOV3_VITB: 12,
    MODEL_DINOV3_VITL: 24,
}

# ===========================
# Helpers
# ===========================

def lab_to_rgb_tensor(lab_image: torch.Tensor) -> torch.Tensor:
    """Convert LAB tensor to RGB tensor."""
    squeeze_dim = False
    if lab_image.ndim == 3:
        lab_image = lab_image.unsqueeze(0)
        squeeze_dim = True
    
    batch_size = lab_image.shape[0]
    rgb_batch = []
    
    for i in range(batch_size):
        lab_np = lab_image[i].permute(1, 2, 0).cpu().numpy()
        rgb_np = cv2.cvtColor(lab_np, cv2.COLOR_LAB2RGB)
        rgb_tensor = torch.from_numpy(rgb_np).float()
        rgb_batch.append(rgb_tensor.permute(2, 0, 1))
    
    result = torch.stack(rgb_batch)
    return result.squeeze(0) if squeeze_dim else result

def load_dinov3_model(
    model_name: str, 
    repo_or_dir: str = "facebookresearch/dinov3", 
    device: str = "cuda"
):
    """Helper to load DINOv3 model using torch.hub."""
    source = "local" if repo_or_dir != "facebookresearch/dinov3" else "github"
    
    # Handle the case where weights map might not exist in context, or fallback
    weights = WEIGHTS_MAP.get(model_name)
    
    model = torch.hub.load(
        repo_or_dir=repo_or_dir,
        model=model_name,
        source=source,
        weights=weights,
    )
    model.to(device)
    model.eval()
    return model

# ===========================
# Pipeline Class
# ===========================

class ForegroundSegmentationPipeline:
    """Encapsulated inference pipeline for foreground segmentation."""
    
    def __init__(
        self,
        model,
        classifier,
        image_size: int = 400,
        patch_size: int = 16,
        n_layers: int = 12,
        feat_layer: int = -1,
        threshold: float = 0.4,
        min_component_size: int = 400,
        dilate_radius: int = 80,
        low_threshold: float = 0.25,
        device: str = "cuda"
    ):
        self.model = model
        self.classifier = classifier
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.feat_layer = feat_layer
        self.threshold = threshold
        self.min_component_size = min_component_size
        self.dilate_radius = dilate_radius
        self.low_threshold = low_threshold
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        classifier_path: Union[str, Path],
        dinov3_location: str = "facebookresearch/dinov3",
        device: str = "cuda",
        **kwargs
    ):
        """
        Factory method to instantiate the pipeline from names and paths.
        
        Args:
            model_name: Name of the DINOv3 model (e.g. 'dinov3_vits16')
            classifier_path: Path to the pickled logistic regression model
            dinov3_location: Local path or 'facebookresearch/dinov3'
            device: 'cuda' or 'cpu'
            **kwargs: Overrides for pipeline params (threshold, image_size, etc.)
        """
        # 1. Load the Backbone Model
        print(f"Loading DINOv3 model: {model_name}...")
        model = load_dinov3_model(model_name, repo_or_dir=dinov3_location, device=device)
        
        # 2. Determine layers based on model name
        n_layers = MODEL_TO_NUM_LAYERS.get(model_name, 12)
        
        # 3. Load the Classifier
        print(f"Loading classifier from {classifier_path}...")
        with open(classifier_path, "rb") as f:
            classifier = pickle.load(f)
            
        # 4. Return instance
        return cls(
            model=model,
            classifier=classifier,
            n_layers=n_layers,
            device=device,
            **kwargs
        )
    
    def __call__(
        self, image: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """Run inference on a single image."""
        
        # Convert and normalize
        image_rgb = lab_to_rgb_tensor(image)
        image_rgb = TF.resize(image_rgb, [self.image_size, self.image_size]).squeeze(0)
        image_normalized = TF.normalize(
            image_rgb, mean=IMAGENET_MEAN, std=IMAGENET_STD
        )
        
        # Extract features
        with torch.inference_mode():
            # Ensure input is on the correct device
            inp = image_normalized.unsqueeze(0).to(self.device)
            
            with torch.autocast(device_type=self.device, dtype=torch.float32):
                feats = self.model.get_intermediate_layers(
                    inp,
                    n=range(self.n_layers),
                    reshape=True,
                    norm=True,
                )
                x = feats[self.feat_layer].squeeze().detach().cpu()
        
        dim = x.shape[0]
        x = x.view(dim, -1).permute(1, 0)
        h_patches = self.image_size // self.patch_size
        w_patches = self.image_size // self.patch_size
        
        # Predict foreground score        
        fg_score = self.classifier.predict_proba(x)[:, 1].reshape(h_patches, w_patches)
        fg_score = cv2.resize(
            fg_score, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR
        )
        fg_score_bin = (fg_score >= self.threshold).astype(np.uint8)
        
        # Find best connected component
        num_labels, labels_im = cv2.connectedComponents(fg_score_bin)
        best_label = 0
        max_fg_score_sum = 0
        
        for label in range(1, num_labels):
            component_mask = labels_im == label
            
            # Skip components touching borders
            if (
                component_mask[0, :].any()
                or component_mask[-1, :].any()
                or component_mask[:, 0].any()
                or component_mask[:, -1].any()
            ):
                continue
            
            # Skip small components
            if component_mask.sum() < self.min_component_size:
                continue
            
            fg_score_sum = fg_score[component_mask].sum()
            if fg_score_sum > max_fg_score_sum:
                max_fg_score_sum = fg_score_sum
                best_label = label
        
        # If no label found, return flag to skip
        if best_label == 0:
            return fg_score, fg_score_bin, np.zeros_like(fg_score_bin), False
        
        pred_mask = (labels_im == best_label).astype(np.uint8)
        
        # Post-processing: expand mask
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.dilate_radius, self.dilate_radius)
        )
        dilated_mask = cv2.dilate(pred_mask, kernel)
        fg_low = (fg_score >= self.low_threshold).astype(np.uint8)
        expanded_mask = cv2.bitwise_and(fg_low, dilated_mask)
        
        # Keep component with best overlap
        num_labels, labels = cv2.connectedComponents(expanded_mask)
        best_overlap_label = 0
        max_overlap = 0
        for lbl in range(1, num_labels):
            comp = labels == lbl
            overlap = np.sum(comp & pred_mask)
            if overlap > max_overlap:
                max_overlap = overlap
                best_overlap_label = lbl
        
        pred_mask = (labels == best_overlap_label).astype(np.uint8)
        
        return fg_score, fg_score_bin, pred_mask, True