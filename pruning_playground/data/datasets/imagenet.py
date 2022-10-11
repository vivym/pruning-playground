from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import datasets, transforms as T
from torchvision.transforms.functional import InterpolationMode

from ..transforms import RandomMixup, RandomCutmix
from ..presets import ClassificationPresetTrain, ClassificationPresetEval


class ImageNet(pl.LightningDataModule):
    def __init__(
        self,
        root_path: str,
        train_batch_size: int = 256,
        val_batch_size: int = 256,
        test_batch_size: int = 256,
        num_workers: int = 16,
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
        train_crop_size: int = 224,
        val_crop_size: int = 224,
        val_resize_size: int = 256,
        interpolation: str = "bilinear",
        auto_augment_policy: Optional[str] = None,
        random_erase_prob: float = 0.0,
        ra_magnitude: int = 9,
        augmix_severity: int = 3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.root_path = Path(root_path)
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.train_crop_size = train_crop_size
        self.val_crop_size = val_crop_size
        self.val_resize_size = val_resize_size
        self.interpolation = InterpolationMode(interpolation)
        self.auto_augment_policy = auto_augment_policy
        self.random_erase_prob = random_erase_prob
        self.ra_magnitude = ra_magnitude
        self.augmix_severity = augmix_severity

    def train_dataloader(self):
        collate_fn = None
        mixup_transforms = []

        mixup_transforms = []
        if self.mixup_alpha > 0.0:
            mixup_transforms.append(RandomMixup(num_classes=1000, p=1.0, alpha=self.mixup_alpha))
        if self.cutmix_alpha > 0.0:
            mixup_transforms.append(RandomCutmix(num_classes=1000, p=1.0, alpha=self.cutmix_alpha))
        if mixup_transforms:
            mixupcutmix = T.RandomChoice(mixup_transforms)

            def collate_fn(batch):
                return mixupcutmix(*default_collate(batch))

        dataset = datasets.ImageFolder(
            self.root_path / "train",
            transform=ClassificationPresetTrain(
                crop_size=self.train_crop_size,
                interpolation=self.interpolation,
                auto_augment_policy=self.auto_augment_policy,
                random_erase_prob=self.random_erase_prob,
                ra_magnitude=self.ra_magnitude,
                augmix_severity=self.augmix_severity,
            )
        )

        return DataLoader(
            dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        dataset = datasets.ImageFolder(
            self.root_path / "val",
            transform=ClassificationPresetEval(
                crop_size=self.val_crop_size,
                resize_size=self.val_resize_size,
                interpolation=self.interpolation,
            )
        )

        return DataLoader(
            dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
