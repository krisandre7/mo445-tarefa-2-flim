#!/usr/bin/env python3
"""
Foreground Segmentation using DINOv3 features and Logistic Regression.
"""

import argparse
import os
import pickle
import shutil
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import random

# Import your custom modules
import pyrootutils

root = pyrootutils.setup_root(
    search_from=Path.cwd(),
    indicator=".project-root",
    pythonpath=True,
    dotenv=True,
)

# IMPORTS CHANGED HERE: Importing constants and the loading helper
from src.models.dinov3_fg_segmentation import (  # noqa: E402
    ForegroundSegmentationPipeline,
    lab_to_rgb_tensor,
    load_dinov3_model,
    MODEL_DINOV3_VITS,
    MODEL_DINOV3_VITB,
    MODEL_DINOV3_VITL,
    MODEL_TO_NUM_LAYERS,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from src.data_modules.schisto import SchistoDataModule  # noqa: E402

# ===========================
# Helper Functions
# ===========================


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ... [Keep dice_coefficient, sensitivity, specificity, f_beta_score as they were] ...
def dice_coefficient(pred_bin: np.ndarray, gt: np.ndarray) -> float:
    if pred_bin.max() == 0 and gt.max() == 0:
        return 1.0
    TP = np.sum((pred_bin == 1) & (gt == 1))
    FP = np.sum((pred_bin == 1) & (gt == 0))
    FN = np.sum((pred_bin == 0) & (gt == 1))
    if TP == 0:
        return 0.0
    return float((2 * TP) / (2 * TP + FP + FN + 1e-8))


def sensitivity(pred_bin: np.ndarray, gt: np.ndarray) -> float:
    if pred_bin.max() == 0 and gt.max() == 0:
        return 1.0
    TP = np.sum((pred_bin == 1) & (gt == 1))
    FN = np.sum((pred_bin == 0) & (gt == 1))
    if TP + FN == 0:
        return 0.0
    return float(TP / (TP + FN + 1e-8))


def specificity(pred_bin: np.ndarray, gt: np.ndarray) -> float:
    if pred_bin.max() == 0 and gt.max() == 0:
        return 1.0
    TN = np.sum((pred_bin == 0) & (gt == 0))
    FP = np.sum((pred_bin == 1) & (gt == 0))
    if TN + FP == 0:
        return 0.0
    return float(TN / (TN + FP + 1e-8))


def f_beta_score(pred_bin: np.ndarray, gt: np.ndarray, beta2: float = 0.3) -> float:
    if pred_bin.max() == 0 and gt.max() == 0:
        return 1.0
    TP = np.sum((pred_bin == 1) & (gt == 1))
    FP = np.sum((pred_bin == 1) & (gt == 0))
    FN = np.sum((pred_bin == 0) & (gt == 1))
    if TP == 0:
        return 0.0
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    return float(
        ((1 + beta2) * precision * recall) / (beta2 * precision + recall + 1e-8)
    )


# ===========================
# Feature Extraction
# ===========================


def extract_features(
    model,
    dataloader,
    image_size: int,
    patch_size: int,
    n_layers: int,
    feat_layer: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # ... [Implementation remains exactly the same as provided in prompt] ...
    patch_quant_filter = torch.nn.Conv2d(
        1, 1, patch_size, stride=patch_size, bias=False
    )
    patch_quant_filter.weight.data.fill_(1.0 / (patch_size * patch_size))

    xs, ys, image_index = [], [], []

    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            for batch_idx, batch in enumerate(
                tqdm(dataloader, desc="Extracting features")
            ):
                image_lab = batch["image"][0]
                mask = batch["label"][0]
                mask = TF.resize(mask.unsqueeze(0), [image_size, image_size]).squeeze(0)

                if mask.ndim == 3:
                    mask = mask.squeeze(0)

                mask_normalized = mask.float() / 255.0
                mask_resized = mask_normalized.unsqueeze(0).unsqueeze(0)
                mask_quantized = patch_quant_filter(mask_resized).squeeze().view(-1)
                ys.append(mask_quantized.detach().cpu())

                image_rgb = lab_to_rgb_tensor(image_lab)
                image_normalized = TF.normalize(
                    image_rgb, mean=IMAGENET_MEAN, std=IMAGENET_STD
                )
                image_normalized = TF.resize(image_normalized, [image_size, image_size])
                image_batch = image_normalized.unsqueeze(0).cuda()

                feats = model.get_intermediate_layers(
                    image_batch, n=range(n_layers), reshape=True, norm=True
                )
                dim = feats[feat_layer].shape[1]
                xs.append(
                    feats[feat_layer]
                    .squeeze()
                    .view(dim, -1)
                    .permute(1, 0)
                    .detach()
                    .cpu()
                )
                image_index.append(batch_idx * torch.ones(ys[-1].shape))

    xs = torch.cat(xs)
    ys = torch.cat(ys)
    image_index = torch.cat(image_index)
    idx = (ys < 0.01) | (ys > 0.99)
    return xs[idx], ys[idx], image_index[idx]


# ===========================
# Training & Evaluation Wrappers
# ===========================


def train_classifier(
    xs: torch.Tensor, ys: torch.Tensor, regularization_c: float = 1e-3
) -> LogisticRegression:
    print("Training classifier...")
    clf = LogisticRegression(
        random_state=0, C=regularization_c, max_iter=100000, verbose=1
    )
    clf.fit(xs.numpy(), (ys > 0.5).long().numpy())
    return clf


def evaluate_pipeline(
    pipeline, dataloader, save_failures=False, failure_dir=Path("low_sensitivity_cases")
):
    # ... [Implementation remains exactly the same as provided in prompt] ...
    # Just ensure to use pipeline(image) call which we haven't changed the signature of
    if save_failures and failure_dir.exists():
        shutil.rmtree(failure_dir)
    if save_failures:
        failure_dir.mkdir(exist_ok=True)

    dice_scores, sensitivity_scores, specificity_scores, f_beta_scores = [], [], [], []

    for item in tqdm(dataloader, desc="Evaluating"):
        image = item["image"][0]
        mask = item["label"][0]
        # item_id = Path(item["image_path"][0]).stem

        test_mask = TF.resize(
            mask.unsqueeze(0), [pipeline.image_size, pipeline.image_size]
        )
        gt_mask = (test_mask.squeeze().numpy() > 0).astype(np.uint8)

        # Inference
        fg_score, fg_score_bin, pred_mask, valid_candidate = pipeline(image)

        if not valid_candidate:
            continue

        dice_scores.append(dice_coefficient(pred_mask, gt_mask))
        sensitivity_scores.append(sensitivity(pred_mask, gt_mask))
        specificity_scores.append(specificity(pred_mask, gt_mask))
        f_beta_scores.append(f_beta_score(pred_mask, gt_mask, beta2=0.3))

        # Save low-sensitivity cases
        if save_failures and sensitivity_scores[-1] * 100 < 55:
            # ... [Save logic remains same] ...
            pass  # (Truncated for brevity, paste original saving logic here)

    return {
        "dice_mean": np.mean(dice_scores),
        "dice_std": np.std(dice_scores),
        "sensitivity_mean": np.mean(sensitivity_scores),
        "sensitivity_std": np.std(sensitivity_scores),
        "specificity_mean": np.mean(specificity_scores),
        "specificity_std": np.std(specificity_scores),
        "f_beta_mean": np.mean(f_beta_scores),
        "f_beta_std": np.std(f_beta_scores),
    }


# ===========================
# Main CLI
# ===========================


def main():
    seed_everything()
    parser = argparse.ArgumentParser()
    # ... [Keep your existing arguments] ...
    parser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_DINOV3_VITS,
        choices=[MODEL_DINOV3_VITS, MODEL_DINOV3_VITB, MODEL_DINOV3_VITL],
    )
    parser.add_argument(
        "--dinov3-location", type=str, default="facebookresearch/dinov3"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--image_size", type=int, default=400
    )  # Check argument name consistency (image-size vs image_size)
    parser.add_argument("--train-split-ratio", type=float, default=0.5)
    parser.add_argument("--classifier-path", type=str, default=None)
    parser.add_argument("--regularization-c", type=float, default=1e-3)
    parser.add_argument("--output-path", type=str, default="fg_classifier.pkl")
    parser.add_argument(
        "--eval-split", type=str, default="test", choices=["test", "val"]
    )
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--save-failures", action="store_true")
    parser.add_argument("--failure-dir", type=str, default="low_sensitivity_cases")
    parser.add_argument(
        "--mode",
        type=str,
        default="train_and_eval",
        choices=["train", "eval", "train_and_eval"],
    )

    args = parser.parse_args()

    # 1. Load model (Simplified using the new helper)
    print(f"Loading DINOv3 model: {args.model_name}")
    model = load_dinov3_model(
        model_name=args.model_name, repo_or_dir=args.dinov3_location
    )

    n_layers = MODEL_TO_NUM_LAYERS[args.model_name]

    # 2. Initialize datamodule
    print("Preparing data...")
    datamodule = SchistoDataModule(
        batch_size=args.batch_size,
        use_flim_data=False,
        num_workers=args.num_workers,
        image_size=args.image_size,
        train_split_ratio=args.train_split_ratio,
    )
    datamodule.prepare_data()
    datamodule.setup()

    # 3. Training Logic
    clf = None
    if args.mode in ["train", "train_and_eval"]:
        if args.classifier_path is not None:
            print(f"Loading classifier from {args.classifier_path}")
            with open(args.classifier_path, "rb") as f:
                clf = pickle.load(f)
        else:
            print("Training new classifier...")
            train_dataloader = datamodule.train_dataloader()
            xs, ys, _ = extract_features(
                model, train_dataloader, args.image_size, 16, n_layers
            )
            clf = train_classifier(xs, ys, regularization_c=args.regularization_c)
            with open(args.output_path, "wb") as f:
                pickle.dump(clf, f)
    else:
        # Eval only
        if args.classifier_path is None:
            raise ValueError("Provide classifier path for eval")
        with open(args.classifier_path, "rb") as f:
            clf = pickle.load(f)

    # 4. Evaluation Logic
    if args.mode in ["eval", "train_and_eval"]:
        eval_dataloader = (
            datamodule.test_dataloader()
            if args.eval_split == "test"
            else datamodule.val_dataloader()
        )

        # Dependency Injection style (Standard for training script)
        pipeline = ForegroundSegmentationPipeline(
            model=model,
            classifier=clf,
            image_size=args.image_size,
            n_layers=n_layers,
            threshold=args.threshold,
        )

        metrics = evaluate_pipeline(
            pipeline, eval_dataloader, args.save_failures, Path(args.failure_dir)
        )

        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        print(f"Dice Score:     {metrics['dice_mean']:.4f} ± {metrics['dice_std']:.4f}")
        print(
            f"Sensitivity:    {metrics['sensitivity_mean']:.4f} ± {metrics['sensitivity_std']:.4f}"
        )
        print(
            f"Specificity:    {metrics['specificity_mean']:.4f} ± {metrics['specificity_std']:.4f}"
        )
        print(
            f"F-beta Score:   {metrics['f_beta_mean']:.4f} ± {metrics['f_beta_std']:.4f}"
        )
        print("=" * 50)

        if args.save_failures:
            num_failures = len(list(Path(args.failure_dir).iterdir()))
            print(f"Saved {num_failures} low-sensitivity cases to {args.failure_dir}")


if __name__ == "__main__":
    main()
