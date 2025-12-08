import argparse
import sys

def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", action="store_true", help="training mode")
    parser.add_argument("--test", action="store_true", help="testing mode")
    parser.add_argument("--track", action="store_true", help="Track the training of the model")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dm", action="store_true", help="Run diffusion model")
    group.add_argument("--gan", action="store_true", help="Run GAN model")
    group.add_argument("--fb", action="store_true", help="Run flow-based model")
    group.add_argument("--vae", action="store_true", help="Run variational autoencoder model")

    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument("--mnist", action="store_true", help="Select MNIST dataset")
    dataset_group.add_argument("--fashion", action="store_true", help="Select Fashion MNIST dataset")
    dataset_group.add_argument("--cifar10", action="store_true", help="Select CIFAR-10 dataset")

    args = parser.parse_args(args)

    if not (args.train or args.test):
        parser.error("You must specify at least --train or --test")

    # Converti i flag in stringa
    if args.mnist:
        dataset_name = "mnist"
    elif args.fashion:
        dataset_name = "fashion"
    elif args.cifar10:
        dataset_name = "cifar10"
    else:
        dataset_name = None  # non dovrebbe mai succedere

    return args, dataset_name

