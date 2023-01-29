import urllib.request
import tarfile
import shutil
from pathlib import Path
import cv2
from cvops.configs import DATA_PATH, STAGING_PATH, BBC05v1DataConfigs


class BBC05v1Importer:
    """
    Import data from original data source and organize it into train, valid and test sets.
    """

    def __init__(
        self,
        data_path=DATA_PATH,
        staging_path=STAGING_PATH,
        staging_image_dir=BBC05v1DataConfigs.STAGING_IMAGE_DIR,
        staging_mask_dir=BBC05v1DataConfigs.STAGING_MASK_DIR,
        download_path=BBC05v1DataConfigs.DOWNLOAD_PATH,
        train_image_dir=BBC05v1DataConfigs.TRAIN_IMAGE_DIR,
        train_mask_dir=BBC05v1DataConfigs.TRAIN_MASK_DIR,
        valid_image_dir=BBC05v1DataConfigs.VALID_IMAGE_DIR,
        valid_mask_dir=BBC05v1DataConfigs.VALID_MASK_DIR,
        test_image_dir=BBC05v1DataConfigs.TEST_IMAGE_DIR,
        test_mask_dir=BBC05v1DataConfigs.TEST_MASK_DIR,
        dataset_url=BBC05v1DataConfigs.DATASET_URL,
        train=0.8,
        valid=0.1,
        dry_run=False,
        clean_staging=True,
    ):
        self.data_path = data_path
        self.dataset_url = dataset_url
        self.staging_path = staging_path
        self.staging_image_dir = staging_image_dir
        self.staging_mask_dir = staging_mask_dir
        self.download_path = download_path
        self.train_image_dir = train_image_dir
        self.train_mask_dir = train_mask_dir
        self.valid_image_dir = valid_image_dir
        self.valid_mask_dir = valid_mask_dir
        self.test_image_dir = test_image_dir
        self.test_mask_dir = test_mask_dir

        self._download_data()
        staging_examples = self._get_valid_file_pairs()
        staging_train_examples, staging_valid_examples, staging_test_examples = self._split_files(
            staging_examples, train, valid
        )
        if dry_run is False:
            self._move_examples_to_foler(staging_train_examples, self.train_image_dir, self.train_mask_dir)
            self._move_examples_to_foler(staging_valid_examples, self.valid_image_dir, self.valid_mask_dir)
            self._move_examples_to_foler(staging_test_examples, self.test_image_dir, self.test_mask_dir)

        if clean_staging is True:
            shutil.rmtree(self.staging_path)

        self._nb_examples = len(staging_examples)
        self._nb_examples_train = len(staging_train_examples)
        self._nb_examples_valid = len(staging_valid_examples)
        self._nb_examples_test = len(staging_test_examples)

    @property
    def size(self):
        return self._nb_examples

    @property
    def train_size(self):
        return self._nb_examples_train

    @property
    def valid_size(self):
        return self._nb_examples_valid

    @property
    def test_size(self):
        return self._nb_examples_test

    def __str__(self) -> str:
        return f"{self.__class__.__name__} object with {self.size} examples; {self.train_size} test examples, {self.valid_size} valid examples and {self.test_size} test examples."

    def _download_data(self):
        """
        Download and extract dataset if staging data if not present
        """

        if not self.staging_mask_dir.exists() or not self.staging_image_dir.exists():
            if not self.staging_path.exists():
                self.staging_path.mkdir(parents=True)
            if not self.download_path.exists():
                urllib.request.urlretrieve(self.dataset_url, filename=self.download_path)
            with tarfile.open(self.download_path, "r") as zip_ref:
                zip_ref.extractall(self.staging_path)

    def _get_valid_file_pairs(self):

        """
        Filter valid files.  Valid files are
            - Files that have a size ratio beween 0.9 and 1.4
            - Files with non-zero file size
            - Files for wich mask and images have a corresponding name

        Parameters:
            None

        Returns:
            example_pairs: list of dict(image, masks) of Path
        """

        example_pairs = []

        number_of_files = 0

        for image_path in self.staging_image_dir.iterdir():

            if image_path.suffix.casefold() != ".tif":
                continue

            number_of_files += 1

            if not (self.staging_mask_dir / image_path.name).exists():
                continue

            image = cv2.imread(str(image_path))
            if image is None or image.size == 0 or not 0.9 <= image.shape[1] / image.shape[0] < 1.4:
                continue

            example_pairs.append({"image": image_path, "mask": self.staging_mask_dir / image_path.name})

        print(f"Found: {number_of_files} images, keeping {len(example_pairs)}")

        return example_pairs

    def _split_files(self, examples, train=0.8, valid=0.1):
        """
        Split list of examples into train, valid and test lists according to
        split ratios given by train and valid value.  Test ratio is (1-train-valid)

        Parameters:
            examples : list of dict(image, masks) of Path
            train: float value between 0 and 1.  Correspond to the porportion of examples that will be assigned to the train set.
            valid: float value between 0 and 1.  Correspond to the porportion of examples that will be assigned to the valid set.

        Returns:
            train_examples: list of dict(image, masks) of Path
            valid_exampes: list of dict(image, masks) of Path
            test_examples: list of dict(image, masks) of Path
        """

        n_train = int(len(examples) * train)
        n_valid = int(len(examples) * valid)
        n_test = len(examples) - n_train - n_valid

        train_examples = examples[:n_train]
        valid_exampes = examples[n_train : n_train + n_valid]
        test_examples = examples[-n_test:]

        return train_examples, valid_exampes, test_examples

    def _move_examples_to_foler(self, examples, image_dir: Path, mask_dir: Path):
        """
        Move a set of examples to a destination folder

        Parameters:
            examples : list of dict(image, masks) of Path
            image_dir : Path
            mask_dir : Path
        """

        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        for ex in examples:
            ex["image"].rename(image_dir / ex["image"].name)
            ex["mask"].rename(mask_dir / ex["mask"].name)

    def get_number_of_example(self):
        pass


if __name__ == "__main__":
    importer = BBC05v1Importer()
    print(importer)
