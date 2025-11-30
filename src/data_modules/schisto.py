import os
from typing import Any, Optional
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from pyflim import data
import albumentations as A

class SchistoDataModule(LightningDataModule):
    def __init__(
        self,
        split: int = 3,
        home: str = "/home/kris/projects/mo445-analise-de-imagem/tarefa_2",
        train_split_ratio: float = 0.5,
        use_flim_data: bool = True,
        batch_size: int = 1,
        num_workers: int = 0,
        seed: int = 2021,
        transforms: Optional[dict[str, list[dict[str, Any]]]] = None,
        image_size: int = 400,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.orig_folder = f"{home}/datasets/schisto/images/"
        self.label_folder = f"{home}/datasets/schisto/truelabels/"
        self.marker_folder = f"{home}/data/schisto/split{split}/markers/"
        self.train_split_file = f"{home}/data/schisto/split{split}/train{split}.csv"
        self.test_split_file = f"{home}/data/schisto/split{split}/test{split}.csv"
        
        # --- transforms ---
        self.transforms_cfg = transforms or {}
        self.transforms_train: Optional[A.Compose] = None
        self.transforms_val: Optional[A.Compose] = None
        self.transforms_test: Optional[A.Compose] = None
        
        self.rng = np.random.default_rng(seed)
        
        if transforms:
            self._build_transforms()
    
    # =========================================================================
    # TRANSFORMS LOGIC
    # =========================================================================
    def _build_transforms(self):
        """Build albumentations transforms for train/val/test splits."""
        def build_transform_list(transforms: list[dict]):
            out = []
            for transform in transforms:
                cls_name = transform["class_path"]
                init_args = transform.get("init_args", {})
                if hasattr(A, cls_name):
                    transform_cls = getattr(A, cls_name)
                    out.append(transform_cls(**init_args))
                else:
                    raise ValueError(f"Albumentations has no transform '{cls_name}'")
            return A.Compose(out)

        if "train" in self.transforms_cfg:
            self.transforms_train = build_transform_list(self.transforms_cfg["train"])
        if "val" in self.transforms_cfg:
            self.transforms_val = build_transform_list(self.transforms_cfg["val"])
        if "test" in self.transforms_cfg:
            self.transforms_test = build_transform_list(self.transforms_cfg["test"])

    def setup(self, stage: Optional[str] = None):
        # Get image lists for FLIM training (images with markers)
        images_list_flim = [
            i.replace("-seeds.txt", "") 
            for i in os.listdir(self.marker_folder)
        ]
        
        # Get image lists for GWE training
        train_df = pd.read_csv(self.train_split_file, header=None)
        images_list_gwe_all = [
            i.replace('images/', '').replace('.png', '') 
            for i in train_df[0]
            if i not in images_list_flim and 
            np.array(Image.open(
                self.label_folder + i.replace('images/', '')
            )).max() > 0
        ]
        
        # Split GWE images into train/val
        split_idx = int(self.hparams.train_split_ratio * len(images_list_gwe_all))
        
        # if train_split_ratio is 1, then use all images for training
        if split_idx == len(images_list_gwe_all):
            images_list_gwe_train = images_list_gwe_all
            images_list_gwe_val = images_list_gwe_all
        else:
            images_list_gwe_train = images_list_gwe_all[:split_idx]
            images_list_gwe_val = images_list_gwe_all[split_idx:]
        
        # Test images
        test_df = pd.read_csv(self.test_split_file, header=None)
        images_list_test = [
            i.replace('images/', '').replace('.png', '') 
            for i in test_df[0]
        ]
        
        # print a table summary
        print("Dataset summary:")
        print(f"  FLIM train images: {len(images_list_flim)}")
        print(f"  GWE train images: {len(images_list_gwe_train)}")
        print(f"  GWE val images: {len(images_list_gwe_val)}")
        print(f"  Test images: {len(images_list_test)}")
        
        if stage == "fit" or stage is None:
            self.train_dataset_flim = data.FLIMData(
                orig_folder=self.orig_folder,
                marker_folder=self.marker_folder,
                images_list=images_list_flim,
                label_folder=self.label_folder,
                orig_ext=".png",
                marker_ext="-seeds.txt",
                label_ext=".png",
                transform=self.transforms_train,
                lab_norm=False
            )
            
            self.train_dataset_gwe = data.FLIMData(
                orig_folder=self.orig_folder,
                images_list=images_list_gwe_train,
                label_folder=self.label_folder,
                orig_ext=".png",
                label_ext=".png",
                transform=self.transforms_train,
                lab_norm=False
            )
            
            self.val_dataset = data.FLIMData(
                orig_folder=self.orig_folder,
                images_list=images_list_gwe_val,
                label_folder=self.label_folder,
                orig_ext=".png",
                label_ext=".png",
                transform=self.transforms_val,
                lab_norm=False
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = data.FLIMData(
                orig_folder=self.orig_folder,
                images_list=images_list_test,
                label_folder=self.label_folder,
                orig_ext=".png",
                label_ext=".png",
                transform=self.transforms_test,
                lab_norm=False
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset_flim if self.hparams.use_flim_data else self.train_dataset_gwe,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers
        )
    
    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers
        )
