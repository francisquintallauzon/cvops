"""Module that contains all data related functions and classes"""

import urllib
import random
import tarfile
from pathlib import Path
import torch
import pytorch_lightning as pl
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset


class CellDataset(Dataset):
    """Cell torch dataset"""

    def __init__(
        self, images_directory, masks_directory, mask_filenames, transform=None
    ):
        self.images_directory = images_directory
        self.masks_directory = masks_directory
        self.filenames = mask_filenames
        self.transform = transform  # test de sauve

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
        # mask=np.expand_dims(mask,axis=0)

        return image, mask


class CellDataModule(pl.LightningDataModule):
    """Data module for pytorch lightning"""

    def __init__(self, batch_size=10, img_resize_size=512, max_images=None, **kwargs):
        """
        Initialization of inherited lightning data module
        """
        super().__init__()
        self.img_resize_size = img_resize_size
        self.data_path = Path("data/datasets/")
        self.download_path = self.data_path / "BBBC05.tar.gz"
        self.image_dir = self.data_path / "BBBC005_v1_images"
        self.mask_dir = self.data_path / "BBBC005_v1_ground_truth"
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
        self.max_images = max_images

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

    def prepare_data(self):

        # dset_url = "https://www.kaggle.com/datasets/vbookshelf/synthetic-cell-images-and-masks-bbbc005-v1"
        dset_url = "https://www.googleapis.com/drive/v3/files/1UJMyQmI8bZCWB2UKWqa0F00I7f_nz-2q?alt=media&key=AIzaSyDhmuR1Oj_myOqYXEXBQ0J3FN1-cwvR9zI"

        if not self.image_dir.exists() or not self.mask_dir.exists():
            if not self.data_path.exists():
                self.data_path.mkdir()
            if not self.download_path.exists():
                urllib.request.urlretrieve(dset_url, filename=self.download_path)
            with tarfile.open(self.download_path, "r") as zip_ref:
                zip_ref.extractall(self.data_path)

        fnames = []
        fnames_nofilter = []
        widths = []
        count = 0
        for mask_fullpath in self.mask_dir.iterdir():
            if mask_fullpath.suffix.casefold() == ".tif":
                fnames_nofilter.append(mask_fullpath)
                image_fullpath = self.image_dir / mask_fullpath.name
                image = cv2.imread(str(image_fullpath))
                if (
                    image is not None
                    and image.size != 0
                    and 0.9 <= image.shape[1] / image.shape[0] < 1.4
                ):
                    fnames.append(image_fullpath.name)
                    widths.append(image.shape[1])
                    count += 1
                    if self.max_images:
                        if count > self.max_images:
                            break
        print(f"Initial images: {len(fnames_nofilter)}, keeping {len(fnames)}")

        self.filenames = fnames  # list(sorted(fnames))

        random.seed(43)
        random.shuffle(self.filenames)
        n_val = int(len(self.filenames) * 0.2)
        self.filenames_valid = self.filenames[:n_val]
        self.filenames_train = self.filenames[n_val:]
        print(
            f"{len(self.filenames_train)} train, {len(self.filenames_valid)} validation"
        )

    def setup(self, stage: str):
        """
        Create train and valid datasets
        """

        print(f"In CellDataModule.setup; stage = {stage}")
        if stage == "fit":
            self.dataset_train = CellDataset(
                self.image_dir,
                self.mask_dir,
                self.filenames_train,
                transform=self.transform_train,
            )

            self.dataset_valid = CellDataset(
                self.image_dir,
                self.mask_dir,
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
