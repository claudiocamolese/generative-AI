import torch
import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import random_split

class MyDataloader:
    def __init__(self, dataset_name, root="./data", seed=42):
        """
        Args:
            dataset_name (str): dataset name in torchvision.datasets
            root (str): path to save files
            seed (int): seed for random_split
        """
        self.name = dataset_name
        self.root = root
        self.seed = seed

        if self.name == 'mnist':
            self.train_loader, self.val_loader, self.test_loader = self._mnist()
        elif self.name == 'fashion':
            self.train_loader, self.val_loader, self.test_loader = self._fashion()
        elif self.name == 'cifar10':
            os.makedirs(f"data/{self.name}/", exist_ok= True)
            self.train_loader, self.val_loader, self.test_loader = self._cifar10()
        elif self.name == 'cifar100':
            os.makedirs(f"data/{self.name}/", exist_ok= True)
            self.train_loader, self.val_loader, self.test_loader = self._cifar100()
        else:
            raise ValueError(f"Dataset {dataset_name} not supported.")

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
        
        train_loader = DataLoader(train_dataset, batch_size= 64, shuffle=True)
        val_loader   = DataLoader(val_dataset, batch_size= 64, shuffle=False)
        test_loader  = DataLoader(test_dataset, batch_size= 64, shuffle=False)
        
        return train_loader, val_loader, test_loader

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
        
        train_loader = DataLoader(train_dataset, batch_size= 64, shuffle=True)
        val_loader   = DataLoader(val_dataset, batch_size= 64, shuffle=False)
        test_loader  = DataLoader(test_dataset, batch_size= 64, shuffle=False)

        return train_loader, val_loader, test_loader
    
    def _cifar10(self):

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])])

        full_train = datasets.CIFAR10(
            root=f"{self.root}/{self.name}/",
            train=True,
            download=True,
            transform=transform
        )
        test_dataset = datasets.CIFAR10(
            root=f"{self.root}/{self.name}/",
            train=False,
            download=True,
            transform=transform
        )

        train_size = int(0.8 * len(full_train))
        val_size = len(full_train) - train_size

        generator = torch.Generator().manual_seed(self.seed)
        train_dataset, val_dataset = random_split(full_train, [train_size, val_size], generator=generator)
        
        train_loader = DataLoader(train_dataset, batch_size= 64, shuffle=True)
        val_loader   = DataLoader(val_dataset, batch_size= 64, shuffle=False)
        test_loader  = DataLoader(test_dataset, batch_size= 64, shuffle=False)

        return train_loader, val_loader, test_loader
    

    def _cifar100(self):

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])])

        full_train = datasets.CIFAR10(
            root=f"{self.root}/{self.name}/",
            train=True,
            download=True,
            transform=transform
        )
        test_dataset = datasets.CIFAR10(
            root=f"{self.root}/{self.name}/",
            train=False,
            download=True,
            transform=transform
        )

        train_size = int(0.8 * len(full_train))
        val_size = len(full_train) - train_size

        generator = torch.Generator().manual_seed(self.seed)
        train_dataset, val_dataset = random_split(full_train, [train_size, val_size], generator=generator)
        
        train_loader = DataLoader(train_dataset, batch_size= 64, shuffle=True)
        val_loader   = DataLoader(val_dataset, batch_size= 64, shuffle=False)
        test_loader  = DataLoader(test_dataset, batch_size= 64, shuffle=False)

        return train_loader, val_loader, test_loader

    def get_dataloader(self):
        """
            Returns the dataloader: train, val, test
        """
        return self.train_loader, self.val_loader, self.test_loader
