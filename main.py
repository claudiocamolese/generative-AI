import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import argparse
import sys

def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="model to be trained",
                        choices=["mog"])
    parser.add_argument("--train", action="store_true",
                        help="training mode")
    parser.add_argument("--test", action="store_true",
                        help="testing mode")
    parser.add_argument("--gan", action="store_true",
                        help="use GAN during training")
    parser.add_argument("--dm", action= "store_true",
                        help="use Diffusion models during training")
    return parser.parse_args(args)

def main(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


    full_train = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )


    train_size = int(0.8 * len(full_train))   # 48,000
    val_size = len(full_train) - train_size   # 12,000

    train_dataset, val_dataset = random_split(full_train, [train_size, val_size])


    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

    if args.train :
        print("train")
        if args.gan:
            print("gan")
        elif args.dm:
            print(len(train_loader))
            print(len(val_loader))
            print(len(test_loader))
    else:
        pass

if __name__ == "__main__":
    args = parse_args()
    main(args)