import torch

from torchvision import datasets, transforms
from torch.utils.data import random_split

class Dataset:
    def __init__(self, dataset_name, root="./data", seed=42):
        """
        Classe per caricare MNIST o FashionMNIST, con split train/val/test.

        Args:
            dataset_name (str): 'mnist' o 'fashion'
            root (str): directory dove salvare i dati
            seed (int): seed per random_split
        """
        self.name = dataset_name
        self.root = root
        self.seed = seed

        if self.name == 'mnist':
            self.train_dataset, self.val_dataset, self.test_dataset = self._mnist()
        elif self.name == 'fashion':
            self.train_dataset, self.val_dataset, self.test_dataset = self._fashion()
        else:
            raise ValueError(f"Dataset {dataset_name} non supportato. Usa 'mnist' o 'fashion'.")

    def _mnist(self):

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        full_train = datasets.MNIST(
            root=self.root,
            train=True,
            download=True,
            transform=transform
        )
        test_dataset = datasets.MNIST(
            root=self.root,
            train=False,
            download=True,
            transform=transform
        )

        train_size = int(0.8 * len(full_train))
        val_size = len(full_train) - train_size

        # Split riproducibile
        generator = torch.Generator().manual_seed(self.seed)
        train_dataset, val_dataset = random_split(full_train, [train_size, val_size], generator=generator)
        return train_dataset, val_dataset, test_dataset

    def _fashion(self):

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        full_train = datasets.FashionMNIST(
            root=self.root,
            train=True,
            download=True,
            transform=transform
        )
        test_dataset = datasets.FashionMNIST(
            root=self.root,
            train=False,
            download=True,
            transform=transform
        )

        train_size = int(0.8 * len(full_train))
        val_size = len(full_train) - train_size

        generator = torch.Generator().manual_seed(self.seed)
        train_dataset, val_dataset = random_split(full_train, [train_size, val_size], generator=generator)
        return train_dataset, val_dataset, test_dataset

    def get_datasets(self):
        """
        Restituisce i dataset: train, val, test
        """
        return self.train_dataset, self.val_dataset, self.test_dataset
