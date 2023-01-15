from pathlib import Path

DATA_PATH = Path("data/datasets/")
STAGING_PATH = Path("data/staging/")


class BBC05v1DataConfigs:
    # dset_url = "https://www.kaggle.com/datasets/vbookshelf/synthetic-cell-images-and-masks-bbbc005-v1"
    # dataset_url = "https://data.broadinstitute.org/bbbc/BBBC005/BBBC005_v1_images.zip"
    DATASET_URL = "https://www.googleapis.com/drive/v3/files/1UJMyQmI8bZCWB2UKWqa0F00I7f_nz-2q?alt=media&key=AIzaSyDhmuR1Oj_myOqYXEXBQ0J3FN1-cwvR9zI"

    DOWNLOAD_PATH = STAGING_PATH / "BBBC005v1.gz"
    STAGING_IMAGE_DIR = STAGING_PATH / "BBBC005_v1_images"
    STAGING_MASK_DIR = STAGING_PATH / "BBBC005_v1_ground_truth"

    DATASET_PATH = DATA_PATH / "BBC005_v1"
    TRAIN_IMAGE_DIR = DATASET_PATH / "train" / "images"
    TRAIN_MASK_DIR = DATASET_PATH / "train" / "masks"
    VALID_IMAGE_DIR = DATASET_PATH / "valid" / "images"
    VALID_MASK_DIR = DATASET_PATH / "valid" / "masks"
    TEST_IMAGE_DIR = DATASET_PATH / "test" / "images"
    TEST_MASK_DIR = DATASET_PATH / "test" / "masks"
