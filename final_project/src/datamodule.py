import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import lightning as L


class ImageDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 64,
        dataset_name: str = "mnist",
        train_proportion: float = 0.9,
        num_workers: int = 2,
        random_rotation_test: bool = False,
        random_rotation_train: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.train_proportion = train_proportion
        self.num_workers = num_workers
        self.random_rotation_test = random_rotation_test
        self.random_rotation_train = random_rotation_train
        self.seed = seed
        if dataset_name not in ["mnist", "cifar10", "cifar100", "fashion_mnist"]:
            raise ValueError(f"Dataset {dataset_name} not supported")

        self.train_transform: transforms.Compose = None
        self.test_transform: transforms.Compose = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _calculate_train_val_lengths(self, total_train_size: int) -> tuple[int, int]:
        """Calculate train and validation lengths based on train_proportion."""
        train_size = int(total_train_size * self.train_proportion)
        val_size = total_train_size - train_size
        return train_size, val_size

    def _create_transforms(self):
        """Create data transforms based on dataset and rotation settings."""
        # Dataset normalization parameters
        normalize_params = {
            "mnist": {"mean": (0.1307,), "std": (0.3081,)},
            "fashion_mnist": {"mean": (0.2860,), "std": (0.3530,)},
            "cifar10": {
                "mean": (0.4914, 0.4822, 0.4465),
                "std": (0.2023, 0.1994, 0.2010),
            },
            "cifar100": {
                "mean": (0.5071, 0.4867, 0.4408),
                "std": (0.2675, 0.2565, 0.2761),
            },
        }

        if self.dataset_name not in normalize_params:
            raise ValueError(f"Dataset {self.dataset_name} not supported")

        # Define rotation transform for train and test
        rotation_transform_train = (
            transforms.RandomRotation(
                degrees=[0, 360],
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=0,
            )
            if self.random_rotation_train
            else None
        )

        rotation_transform_test = (
            transforms.RandomRotation(
                degrees=[0, 360],
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=0,
            )
            if self.random_rotation_test
            else None
        )

        # Create transforms using predefined normalization parameters
        base_train_transforms = [transforms.ToTensor()]
        base_test_transforms = [transforms.ToTensor()]

        if self.random_rotation_train:
            base_train_transforms.append(rotation_transform_train)
        if self.random_rotation_test:
            base_test_transforms.append(rotation_transform_test)

        base_train_transforms.append(
            transforms.Normalize(**normalize_params[self.dataset_name])
        )
        base_test_transforms.append(
            transforms.Normalize(**normalize_params[self.dataset_name])
        )

        self.train_transform = transforms.Compose(base_train_transforms)
        self.test_transform = transforms.Compose(base_test_transforms)

    def prepare_data(self):
        """Download datasets if not already present."""
        dataset_map = {
            "mnist": datasets.MNIST,
            "fashion_mnist": datasets.FashionMNIST,
            "cifar10": datasets.CIFAR10,
            "cifar100": datasets.CIFAR100,
        }

        dataset_class = dataset_map[self.dataset_name]

        # Download train and test sets
        dataset_class(self.data_dir, train=True, download=True)
        dataset_class(self.data_dir, train=False, download=True)

    def setup(self, stage: str = None):
        """Set up train, validation, and test datasets."""
        self._create_transforms()

        dataset_map = {
            "mnist": datasets.MNIST,
            "fashion_mnist": datasets.FashionMNIST,
            "cifar10": datasets.CIFAR10,
            "cifar100": datasets.CIFAR100,
        }

        dataset_class = dataset_map[self.dataset_name]

        if stage == "fit" or stage is None:
            # Load full training dataset
            full_train_dataset = dataset_class(
                self.data_dir, train=True, transform=self.train_transform
            )

            # Calculate split lengths based on proportion
            total_train_size = len(full_train_dataset)
            train_size, val_size = self._calculate_train_val_lengths(total_train_size)

            # Split into train and validation
            self.train_dataset, self.val_dataset = random_split(
                full_train_dataset,
                lengths=[train_size, val_size],
                generator=torch.Generator().manual_seed(self.seed),
            )

        if stage == "test" or stage is None:
            self.test_dataset = dataset_class(
                self.data_dir, train=False, transform=self.test_transform
            )

    def train_dataloader(self):
        """Create training data loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )

    def val_dataloader(self):
        """Create validation data loader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )

    def test_dataloader(self):
        """Create test data loader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )
