# %% [markdown]
# # Training a Foreground Segmentation Tool with DINOv3
# 
# In this tutorial, we will train a linear foreground segmentation model using DINOv3 features.

# %% [markdown]
# ### Setup
# 
# Let's start by loading some pre-requisites and checking the DINOv3 repository location:

# %%
import io
import os
import pickle
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.linear_model import LogisticRegression
import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm
import pyrootutils

root = pyrootutils.setup_root(
    search_from=Path.cwd().parent,
    indicator=".project-root",
    pythonpath=True,
    dotenv=True,
)

from src.data_modules.schisto import SchistoDataModule

DINOV3_GITHUB_LOCATION = "facebookresearch/dinov3"

if os.getenv("DINOV3_LOCATION") is not None:
    DINOV3_LOCATION = os.getenv("DINOV3_LOCATION")
else:
    DINOV3_LOCATION = DINOV3_GITHUB_LOCATION

print(f"DINOv3 location set to {DINOV3_LOCATION}")

# %% [markdown]
# ### Model
# 
# Let's load the DINOv3 model. For this notebook, we will be using the ViT-L model.

# %%
MODEL_DINOV3_VITS = "dinov3_vits16"
MODEL_DINOV3_VITSP = "dinov3_vits16plus"
MODEL_DINOV3_VITB = "dinov3_vitb16"
MODEL_DINOV3_VITL = "dinov3_vitl16"
MODEL_DINOV3_VITHP = "dinov3_vith16plus"
MODEL_DINOV3_VIT7B = "dinov3_vit7b16"

MODEL_NAME = MODEL_DINOV3_VITL

WEIGHTS_MAP = {
    MODEL_DINOV3_VITS: "../checkpoints/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    MODEL_DINOV3_VITL: "../checkpoints/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    MODEL_DINOV3_VITB: "../checkpoints/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
}

model = torch.hub.load(
    repo_or_dir=DINOV3_LOCATION,
    model=MODEL_NAME,
    source="local" if DINOV3_LOCATION != DINOV3_GITHUB_LOCATION else "github",
    weights=WEIGHTS_MAP[MODEL_NAME],
)
model.cuda()

# %% [markdown]
# ### Data
# Now that we have the model set up, let's load the training data using the datamodule.

# %%
IMAGE_SIZE = 512
PATCH_SIZE = 16

# Initialize datamodule
datamodule = SchistoDataModule(
    batch_size=1,
    use_flim_data=False,
    num_workers=8,
    image_size=IMAGE_SIZE
)
datamodule.prepare_data()
datamodule.setup()
train_dataloader = datamodule.train_dataloader()

print(f"Number of training batches: {len(train_dataloader)}")

# %% [markdown]
# ### Building Per-Patch Label Map
# 
# Since our models run with a patch size of 16, we have to quantize the ground truth to a 16x16 pixels grid.

# %%
# Quantization filter for the given patch size
patch_quant_filter = torch.nn.Conv2d(
    1, 1, PATCH_SIZE, stride=PATCH_SIZE, bias=False
)
patch_quant_filter.weight.data.fill_(1.0 / (PATCH_SIZE * PATCH_SIZE))

def lab_to_rgb_tensor(lab_image: torch.Tensor) -> torch.Tensor:
    """Convert LAB tensor to RGB tensor.
    
    Args:
        lab_image: Tensor of shape (C, H, W) or (B, C, H, W) in LAB format
    
    Returns:
        RGB tensor in the same shape, normalized to [0, 1]
    """
    # Handle batch dimension
    squeeze_dim = False
    if lab_image.ndim == 3:
        lab_image = lab_image.unsqueeze(0)
        squeeze_dim = True
    
    batch_size = lab_image.shape[0]
    rgb_batch = []
    
    for i in range(batch_size):
        # Convert to numpy (H, W, C)
        lab_np = lab_image[i].permute(1, 2, 0).cpu().numpy()
        # Convert LAB to RGB using OpenCV
        rgb_np = cv2.cvtColor(lab_np.astype(np.uint8), cv2.COLOR_LAB2RGB)
        # Convert back to tensor and normalize to [0, 1]
        rgb_tensor = torch.from_numpy(rgb_np).float() / 255.0
        rgb_batch.append(rgb_tensor.permute(2, 0, 1))
    
    result = torch.stack(rgb_batch)
    return result.squeeze(0) if squeeze_dim else result

# %% [markdown]
# Let's visualize a sample image and mask:

# %%
sample_item = next(iter(train_dataloader))
sample_image_lab = sample_item["image"][0]  # (C, H, W)
sample_mask = sample_item["label"][0]  # (H, W) or (1, H, W)

# Convert LAB to RGB
sample_image_rgb = lab_to_rgb_tensor(sample_image_lab)

# Handle mask shape
if sample_mask.ndim == 3:
    sample_mask = sample_mask.squeeze(0)

# Normalize mask to [0, 1] range (already 0-255)
sample_mask_normalized = sample_mask.float() / 255.0

plt.figure(figsize=(12, 4), dpi=150)
plt.subplot(1, 3, 1)
plt.imshow(sample_image_rgb.permute(1, 2, 0))
plt.axis('off')
plt.title("RGB Image")
plt.subplot(1, 3, 2)
plt.imshow(sample_mask_normalized, cmap='gray')
plt.axis('off')
plt.title("Mask")
plt.subplot(1, 3, 3)
# Overlay mask on image
overlay = sample_image_rgb.permute(1, 2, 0).clone()
overlay[sample_mask_normalized > 0.5] = torch.tensor([1.0, 0.0, 0.0])
plt.imshow(overlay)
plt.axis('off')
plt.title("Overlay")
plt.show()

# %% [markdown]
# ### Extracting Features and Labels for All the Images

# %%
xs = []
ys = []
image_index = []

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MODEL_TO_NUM_LAYERS = {
    MODEL_DINOV3_VITS: 12,
    MODEL_DINOV3_VITSP: 12,
    MODEL_DINOV3_VITB: 12,
    MODEL_DINOV3_VITL: 24,
    MODEL_DINOV3_VITHP: 32,
    MODEL_DINOV3_VIT7B: 40,
}

n_layers = MODEL_TO_NUM_LAYERS[MODEL_NAME]

with torch.inference_mode():
    with torch.autocast(device_type='cuda', dtype=torch.float32):
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Processing images")):
            # Get image and mask
            image_lab = batch["image"][0]  # (C, H, W)
            mask = batch["label"][0]  # (H, W) or (1, H, W)
            
            # Handle mask shape
            if mask.ndim == 3:
                mask = mask.squeeze(0)
            
            # Normalize mask to [0, 1] (already 0-255)
            mask_normalized = mask.float() / 255.0
            
            # Quantize mask to patch grid
            mask_resized = mask_normalized.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            mask_quantized = patch_quant_filter(mask_resized).squeeze().view(-1)
            ys.append(mask_quantized.detach().cpu())
            
            # Convert LAB to RGB
            image_rgb = lab_to_rgb_tensor(image_lab)  # (C, H, W)
            
            # Normalize with ImageNet statistics
            image_normalized = TF.normalize(image_rgb, mean=IMAGENET_MEAN, std=IMAGENET_STD)
            image_batch = image_normalized.unsqueeze(0).cuda()
            
            # Extract features
            feats = model.get_intermediate_layers(
                image_batch, n=range(n_layers), reshape=True, norm=True
            )
            dim = feats[-1].shape[1]
            xs.append(feats[-1].squeeze().view(dim, -1).permute(1, 0).detach().cpu())
            
            image_index.append(batch_idx * torch.ones(ys[-1].shape))

# Concatenate all lists into torch tensors 
xs = torch.cat(xs)
ys = torch.cat(ys)
image_index = torch.cat(image_index)

# Keep only the patches that have clear positive or negative label
idx = (ys < 0.01) | (ys > 0.99)
xs = xs[idx]
ys = ys[idx]
image_index = image_index[idx]

print("Design matrix size:", xs.shape)
print("Label matrix size:", ys.shape)

# %% [markdown]
# ### Training a Classifier and Model Selection
# We'll train with leave-one-out cross-validation and try different regularization values.

# %%
n_images = len(train_dataloader)
cs = np.logspace(-7, 0, 8)
scores = np.zeros((n_images, len(cs)))

for i in range(n_images):
    print(f'Validation using image {i+1}/{n_images}')
    train_selection = image_index != float(i)
    fold_x = xs[train_selection].numpy()
    fold_y = (ys[train_selection] > 0.5).long().numpy()
    val_x = xs[~train_selection].numpy()
    val_y = (ys[~train_selection] > 0.5).long().numpy()

    plt.figure(figsize=(8, 6))
    for j, c in enumerate(cs):
        print(f"  Training logistic regression with C={c:.2e}")
        clf = LogisticRegression(random_state=0, C=c, max_iter=10000).fit(fold_x, fold_y)
        output = clf.predict_proba(val_x)
        precision, recall, thresholds = precision_recall_curve(val_y, output[:, 1])
        s = average_precision_score(val_y, output[:, 1])
        scores[i, j] = s
        plt.plot(recall, precision, label=f'C={c:.1e} AP={s*100:.1f}')

    plt.grid()
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Image {i+1}/{n_images}')
    plt.axis([0, 1, 0, 1])
    plt.legend()
    plt.show()

# %% [markdown]
# ### Choosing the Best C

# %%
plt.figure(figsize=(8, 6))
plt.plot(scores.mean(axis=0), marker='o')
plt.xticks(np.arange(len(cs)), [f"{c:.0e}" for c in cs], rotation=45)
plt.xlabel('Regularization C')
plt.ylabel('Average AP')
plt.title('Average Precision across validation folds')
plt.grid()
plt.tight_layout()
plt.show()

best_c_idx = scores.mean(axis=0).argmax()
best_c = cs[best_c_idx]
print(f"Best C: {best_c:.2e} with average AP: {scores.mean(axis=0)[best_c_idx]:.3f}")

# %% [markdown]
# ### Retraining with the optimal regularization

# %%
clf = LogisticRegression(
    random_state=0, C=best_c, max_iter=100000, verbose=2
).fit(xs.numpy(), (ys > 0.5).long().numpy())

# %% [markdown]
# ### Test Images and Inference

# %%
# Get test dataloader
test_dataloader = datamodule.test_dataloader()
test_item = next(iter(test_dataloader))

test_image_lab = test_item["image"][0]
test_image_rgb = lab_to_rgb_tensor(test_image_lab)
test_image_normalized = TF.normalize(
    test_image_rgb, mean=IMAGENET_MEAN, std=IMAGENET_STD
)

with torch.inference_mode():
    with torch.autocast(device_type='cuda', dtype=torch.float32):
        feats = model.get_intermediate_layers(
            test_image_normalized.unsqueeze(0).cuda(),
            n=range(n_layers),
            reshape=True,
            norm=True
        )
        x = feats[-1].squeeze().detach().cpu()
        dim = x.shape[0]
        x = x.view(dim, -1).permute(1, 0)

h_patches = test_image_rgb.shape[1] // PATCH_SIZE
w_patches = test_image_rgb.shape[2] // PATCH_SIZE

fg_score = clf.predict_proba(x)[:, 1].reshape(h_patches, w_patches)
fg_score_mf = torch.from_numpy(signal.medfilt2d(fg_score, kernel_size=3))

plt.figure(figsize=(12, 4), dpi=150)
plt.subplot(1, 3, 1)
plt.axis('off')
plt.imshow(test_image_rgb.permute(1, 2, 0))
plt.title('Input Image')
plt.subplot(1, 3, 2)
plt.axis('off')
plt.imshow(fg_score, cmap='viridis')
plt.colorbar()
plt.title('Foreground Score')
plt.subplot(1, 3, 3)
plt.axis('off')
plt.imshow(fg_score_mf, cmap='viridis')
plt.colorbar()
plt.title('+ Median Filter')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Saving the Model for Future Use

# %%
save_root = '.'
model_path = os.path.join(save_root, "fg_classifier.pkl")
with open(model_path, "wb") as f:
    pickle.dump(clf, f)

print(f"Model saved to {model_path}")