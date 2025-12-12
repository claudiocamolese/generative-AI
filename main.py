import comet_ml
from comet_ml import Experiment
import torch

import os
import yaml

from train import Trainer
from test import Tester
from utils.plotting import PlotModel
from utils.parser import parse_args

# ---------- Diffusion ---------------------
from dataset import MyDataloader
from models.diffusion.diffusion import DiffusionModel
from models.diffusion.utils.probability import marginal_prob_std, diffusion_coeff
from models.diffusion.utils.visualizer import DiffVisualizer
# ------------------------------------------


# ---------- Diffusion ---------------------
from models.flow_based.flow_match import FlowMatching
from models.flow_based.denoiser import Denoiser
from models.flow_based.utils.visualizer import FlowVisualizer
# ------------------------------------------


def main(args, dataset_name):
    
    with open('config.yaml', 'r') as file:
        config_file = yaml.safe_load(file) 

    if args.track:
        experiment = Experiment(api_key=config_file['comet']['api_key'], 
                                         project_name=config_file['comet']['project_name'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataloader = MyDataloader(dataset_name= dataset_name)
    train_loader, val_loader, test_loader = dataloader.get_dataloader()

    images, labels = next(iter(train_loader))
    B, channels, height, width = images.shape

    if args.train :
        trainer = Trainer(train_loader= train_loader,
                          val_loader= val_loader,
                          device= device,
                          experiment= experiment if args.track else None,
                          track_flag = True if args.track else False,
                          dataset_name = dataset_name)

        if args.gan:
            print("gan")

        elif args.dm:
            os.makedirs(f"./models/diffusion/checkpoints/{dataset_name}/", exist_ok=True)
            epochs = config_file['train']['dm']['epochs']
            lr = float(config_file['train']['dm']['lr'])

            model = DiffusionModel(marginal_prob_std= marginal_prob_std, 
                                   channels= [32, 64, 128, 256] if channels==1 else [128, 256, 512, 1024],
                                   embed_dim= 256 if channels==1 else 512,
                                   text_dim= 256 if channels==1 else 512,
                                   in_channels= channels, 
                                   img_size= height)
            
            
            plotter = PlotModel(model= model, device = device, in_channel= channels, img_size= height)
            plotter.plot_model(input= plotter.input_diffusion(), path="./figures/diffusion/model/")

            trainer.train_diffusion_model(model= model, epochs= epochs, lr= lr)

        elif args.fm:
            os.makedirs(f"./models/flow_based/checkpoints/{dataset_name}/", exist_ok=True)
            epochs = config_file['train']['fm']['epochs']
            lr = float(config_file['train']['fm']['lr'])
            
            denoiser = Denoiser(in_channels= channels)
            model = FlowMatching(denoiser= denoiser, img_size= (height, height))

            plotter = PlotModel(model= model, device= device, in_channel= channels, img_size= height)
            plotter.plot_model(input= plotter.input_flow_match(), path= "./figures/flow_based/model/")
            trainer.train_flowbased_model(model = model, epochs = epochs, lr= lr)


    if args.test:
        tester = Tester(test_loader= test_loader, 
                        device= device, 
                        track_flag= True if args.track else False,
                        experiment= experiment if args.track else None)

        if args.gan:
            pass

        if args.dm:
            path = f"./models/diffusion/checkpoints/{dataset_name}/final_model.pth"
            
            diffusion_model = DiffusionModel(marginal_prob_std= marginal_prob_std, 
                                   channels= [32, 64, 128, 256] if channels==1 else [128, 256, 512, 1024],
                                   embed_dim= 256 if channels==1 else 512,
                                   text_dim= 256 if channels==1 else 512,
                                   in_channels= channels, 
                                   img_size= height)
            model = tester.test_diffusion(model= diffusion_model, path= path)

            visualizer = DiffVisualizer(
                sampler= diffusion_model.sampling_technique,
                marginal_fn= diffusion_model.marginal_prob_std,
                diffusion_coeff_fn= diffusion_coeff,
                device= device,
                dataset_name= dataset_name
            )

            visualizer.visualize_classes(diffusion_model, sample_batch_size= 16, num_steps= 250)
            visualizer.visualize_steps(diffusion_model, digit= 4, sample_batch_size= 16)

        if args.fm:
            path = f"./models/flow_based/checkpoints/{dataset_name}/final_model.pth"

            denoiser = Denoiser(in_channels= channels, num_hiddens= [32, 64, 128, 256] if channels==1 else [128, 256, 512, 1024])
            flow_based = FlowMatching(denoiser= denoiser, img_size= (height, height))

            visualizer = FlowVisualizer(model= flow_based, dataset_name= dataset_name, device=device)
    
            model = tester.test_flow_based(model= flow_based, path= path)
            visualizer.visualize_classes(sample_batch_size=16)
            visualizer.visualize_steps(digit=4, sample_batch_size=16)


if __name__ == "__main__":
    args, dataset_name = parse_args()
    main(args, dataset_name)