"""Module that contains all data related functions and classes"""

import random
from pathlib import Path
import torch
import pytorch_lightning as pl
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from cvops.configs import BBC05v1DataConfigs


class CellDataset(Dataset):
    """Cell torch dataset"""

    def __init__(self, images_directory, masks_directory, filenames, transform=None):
        self.images_directory = images_directory
        self.masks_directory = masks_directory
        self.filenames = filenames
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = cv2.imread(str(self.images_directory / filename))
        if image is None or image.size == 0:
            while True:
                newidx = random.randint(0, len(self.filenames) - 1)
                filename = self.filenames[newidx]
                image = cv2.imread(str(self.images_directory / filename))
                if image is not None:
                    break
                if image.size != 0:
                    break

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        mask = cv2.imread(str(self.masks_directory / filename), -1)
        mask = mask.astype(np.float32)
        mask /= 255.0

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
            mask = torch.unsqueeze(mask, 0)

        return image, mask


class BBC5v1DataModule(pl.LightningDataModule):
    """Data module for pytorch lightning"""

    def __init__(self, batch_size=50, img_resize_size=512, **kwargs):
        """
        Initialization of inherited lightning data module
        """
        super().__init__()
        self.img_resize_size = img_resize_size
        self.data_path = Path("data/datasets/")
        self.download_path = self.data_path / "BBBC05.tar.gz"
        self.train_image_dir = BBC05v1DataConfigs.TRAIN_IMAGE_DIR
        self.train_mask_dir = BBC05v1DataConfigs.TRAIN_MASK_DIR
        self.valid_image_dir = BBC05v1DataConfigs.VALID_IMAGE_DIR
        self.valid_mask_dir = BBC05v1DataConfigs.VALID_MASK_DIR
        self.test_image_dir = BBC05v1DataConfigs.TEST_IMAGE_DIR
        self.test_mask_dir = BBC05v1DataConfigs.TEST_MASK_DIR
        self.filenames_valid = []
        self.filenames_train = []
        self.dataset_train = None
        self.dataset_valid = None
        self.df_train = None
        self.df_val = None
        self.df_test = None
        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None
        self.args = kwargs

        self.filenames = None
        self.batch_size = batch_size
        self.num_workers = kwargs["num_workers"] if "num_workers" in kwargs else 0

        self.transform_train = A.Compose(
            [
                A.Resize(self.img_resize_size, self.img_resize_size, always_apply=True),
                A.VerticalFlip(p=0.2),
                A.Blur(p=0.2),
                A.RandomBrightnessContrast(p=0.2),
                A.RandomSunFlare(p=0.2, src_radius=200),
                A.RandomShadow(p=0.2),
                A.RandomFog(p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                A.pytorch.ToTensorV2(transpose_mask=True),
            ]
        )

        self.transform_valid = A.Compose(
            [
                A.Resize(self.img_resize_size, self.img_resize_size, always_apply=True),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(transpose_mask=True),
            ]
        )

        self.transform_predict = A.Compose(
            [
                A.Resize(self.img_resize_size, self.img_resize_size, always_apply=True),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    def prepare_data(self) -> None:
        self.filenames_train = [path.name for path in self.train_image_dir.iterdir()]
        self.filenames_valid = [path.name for path in self.valid_image_dir.iterdir()]
        print(
            f"Using {len(self.filenames_train)} training examples and {len(self.filenames_valid)} validation examples"
        )

    def setup(self, stage: str):
        """
        Create train and valid datasets
        """

        print(f"In CellDataModule.setup; stage = {stage}")
        if stage == "fit":
            self.dataset_train = CellDataset(
                self.train_image_dir,
                self.train_mask_dir,
                self.filenames_train,
                transform=self.transform_train,
            )

            self.dataset_valid = CellDataset(
                self.valid_image_dir,
                self.valid_mask_dir,
                self.filenames_valid,
                transform=self.transform_valid,
            )

        if stage == "test":
            raise NotImplementedError

        if stage == "predict":
            raise NotImplementedError

    def create_data_loader(self, dataset: Dataset):
        """
        Generic data loader function
        """
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
            worker_init_fn=self.worker_init,
        )

    def train_dataloader(self):
        """
        :return: output - Train data loader for the given input
        """
        return self.create_data_loader(self.dataset_train)

    def val_dataloader(self):
        """
        :return: output - Validation data loader for the given input
        """
        return self.create_data_loader(self.dataset_valid)

    def test_dataloader(self):
        """
        :return: output - Test data loader for the given input
        """
        raise NotImplementedError

    @staticmethod
    def worker_init(worker_id):
        """Init random module in worker"""
        np.random.seed(42 + worker_id)
