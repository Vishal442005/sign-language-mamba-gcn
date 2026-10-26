"""
dataset.py
==========
Dual-modal dataset loader for Sign Language Recognition.
Loads RGB image and cached 21-landmark MediaPipe skeletal keypoints (.npy).
"""

import os
import glob
from typing import Tuple, Optional, Callable

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class SignLanguageDualModalDataset(Dataset):
    """
    Dual-modal dataset returning:
        - img: Tensor of shape (C, H, W)
        - kpts: Tensor of shape (21, 3) (normalized hand keypoints)
        - label: int class index
    """
    def __init__(
        self,
        root_dir: str,
        is_train: bool = True,
        img_size: int = 224,
        transform: Optional[Callable] = None,
        samples_per_class: Optional[int] = None
    ):
        self.root_dir = root_dir
        self.is_train = is_train
        self.img_size = img_size
        
        # Discover classes
        self.classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            cls_idx = self.class_to_idx[cls_name]
            
            # Find all images
            imgs = [
                os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
                if f.lower().endswith(image_extensions)
            ]
            imgs.sort()
            if samples_per_class is not None and samples_per_class > 0:
                imgs = imgs[:samples_per_class]
                
            for img_path in imgs:
                npy_path = os.path.splitext(img_path)[0] + '.npy'
                self.samples.append((img_path, npy_path, cls_idx))
                
        # Setup image transformations
        if transform is not None:
            self.transform = transform
        else:
            if self.is_train:
                self.transform = transforms.Compose([
                    transforms.Resize((img_size, img_size)),
                    transforms.RandomHorizontalFlip(p=0.3),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        img_path, npy_path, label = self.samples[idx]
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)
        
        # Load cached keypoints or zero fallback
        if os.path.exists(npy_path):
            try:
                kpts_arr = np.load(npy_path)
                if kpts_arr.shape != (21, 3):
                    kpts_arr = np.zeros((21, 3), dtype=np.float32)
            except Exception:
                kpts_arr = np.zeros((21, 3), dtype=np.float32)
        else:
            kpts_arr = np.zeros((21, 3), dtype=np.float32)
            
        kpts_tensor = torch.from_numpy(kpts_arr).float()
        return img_tensor, kpts_tensor, label
