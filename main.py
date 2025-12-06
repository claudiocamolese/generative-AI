import comet_ml
from comet_ml import Experiment
import torch
import argparse
import sys
import os
import yaml

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from train import Trainer
from test import Tester
from plotting import PlotModel


from models.diffusion.diffusion import DiffusionModel
from models.diffusion.utils.probability import marginal_prob_std, diffusion_coeff
from models.diffusion.utils.visualizer import Visualizer

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
    parser.add_argument("--track", action="store_true",
                        help="Track the training of the model")
    return parser.parse_args(args)

def main(args):

    with open('config.yaml', 'r') as file:
        config_file = yaml.safe_load(file) 

    if args.track:
        experiment = Experiment(api_key=config_file['comet']['api_key'], 
                                         project_name=config_file['comet']['project_name'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if config_file['dataset']['name'] == 'mnist':
        full_train = datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transform
        )

        test_dataset = datasets.MNIST(
            root="./data",
            train=False,
            download=True,
            transform=transform
        )

    elif config_file['dataset']['name'] == 'fashion':
        full_train = datasets.FashionMNIST(
            root="./data",
            train=True,
            download=True,
            transform=transform
        )
        test_dataset = datasets.FashionMNIST(
            root="./data",
            train=False,
            download=True,
            transform=transform
        )

    train_size = int(0.8 * len(full_train))   # 48,000
    val_size = len(full_train) - train_size   # 12,000

    train_dataset, val_dataset = random_split(full_train, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size= 64, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size= 64, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size= 64, shuffle=False)

    os.makedirs(f"models/diffusion/checkpoints/{config_file['dataset']['name']}/", exist_ok=True)

    if args.train :
        trainer = Trainer(train_loader= train_loader,
                          val_loader= val_loader,
                          device= device,
                          experiment= experiment if args.track else None,
                          track_flag = True if args.track else False,
                          config =config_file)

        if args.gan:
            print("gan")

        elif args.dm:
            epochs = config_file['train']['dm']['epochs']
            lr = float(config_file['train']['dm']['lr'])

            model = DiffusionModel(marginal_prob_std= marginal_prob_std)
            
            plotter = PlotModel(model= model)
            plotter.plot_model(input= plotter.input_diffusion(), path="./figures/diffusion/model/")

            trainer.train_diffusion_model(model= model, epochs= epochs, lr= lr)

    if args.test:
        tester = Tester(test_loader= test_loader, 
                        device= device, 
                        track_flag= True if args.track else False,
                        experiment= experiment if args.track else None)

        if args.gan:
            pass

        if args.dm:
            path = f"./models/diffusion/checkpoints/{config_file['dataset']['name']}/final_model.pth"
            
            diffusion_model = DiffusionModel(marginal_prob_std= marginal_prob_std)
            model = tester.test_diffusion(model= diffusion_model, path= path)

            visualizer = Visualizer(
                sampler= diffusion_model.sampling_technique,  # usa il metodo della classe
                marginal_fn= diffusion_model.marginal_prob_std,
                diffusion_coeff_fn= diffusion_coeff,
                device= device,
                config= config_file
            )

            visualizer.visualize_classes(diffusion_model, sample_batch_size= 16, num_steps= 250)
            visualizer.visualize_steps(diffusion_model, digit= 4, sample_batch_size= 16)


if __name__ == "__main__":
    args = parse_args()
    main(args)