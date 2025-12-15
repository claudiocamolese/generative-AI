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
from models.diffusion.utils.visualizer import Visualizer
# ------------------------------------------

# ------------ GAN ----------------
from models.gan.gan import ConditionalDiscriminator, ConditionalGenerator
# ---------------------------------

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
    B, channels, high, width = images.shape


    os.makedirs(f"./models/diffusion/checkpoints/{dataset_name}/", exist_ok=True)

    if args.train :
        trainer = Trainer(train_loader= train_loader,
                          val_loader= val_loader,
                          device= device,
                          experiment= experiment if args.track else None,
                          track_flag = True if args.track else False,
                          dataset_name = dataset_name)

        if args.gan:
            epochs = config_file['train']['gan']['epochs']
            g_lr = float(config_file['train']['gan']['g_lr'])
            d_lr = float(config_file['train']['gan']['d_lr'])
            n_critic = int(config_file['train']['gan']['n_critic'])
            gp_lambda = float(config_file['train']['gan']['gp_lambda'])

            latent_size = config_file['model']['gan']['latent_size']
            generator_size = config_file['model']['gan']['generator_size']
            discriminator_size = config_file['model']['gan']['discriminator_size']
            hidden_size = config_file['model']['gan']['hidden_size']
            num_classes = 10

            generator = ConditionalGenerator(num_classes=num_classes, 
                                            generator_size=generator_size, 
                                            latent_size=latent_size)
            
            discriminator = ConditionalDiscriminator(num_classes=num_classes, 
                                                    discriminator_size=discriminator_size, 
                                                    hidden_size=hidden_size)

            trainer.train_gan_model(generator=generator, 
                            discriminator=discriminator, 
                            epochs=epochs, 
                            g_lr=g_lr, 
                            d_lr=d_lr, 
                            n_critic=n_critic, 
                            latent_size=latent_size,
                            num_classes=num_classes,
                            gp_lambda=gp_lambda)

        elif args.dm:
            epochs = config_file['train']['dm']['epochs']
            lr = float(config_file['train']['dm']['lr'])

            model = DiffusionModel(marginal_prob_std= marginal_prob_std, 
                                   channels= [32, 64, 128, 256] if channels==1 else [128, 256, 512, 1024],
                                   embed_dim= 256 if channels==1 else 512,
                                   text_dim= 256 if channels==1 else 512,
                                   in_channels= channels, 
                                   img_size= high)
            
            
            plotter = PlotModel(model= model, device = device, in_channel= channels, img_size= high)
            plotter.plot_model(input= plotter.input_diffusion(), path="./figures/diffusion/model/")

            trainer.train_diffusion_model(model= model, epochs= epochs, lr= lr)

    if args.test:
        tester = Tester(test_loader= test_loader, 
                        device= device, 
                        track_flag= True if args.track else False,
                        experiment= experiment if args.track else None)

        if args.gan:
            gen_path = f"./models/gan/checkpoints/{dataset_name}/generator.pth"
            disc_path = f"./models/gan/checkpoints/{dataset_name}/discriminator.pth"
            
            num_classes = 10
            latent_size = config_file['model']['gan']['latent_size']
            generator_size = config_file['model']['gan']['generator_size']
            channels = channels
            img_size = high
            
            generator_arch_for_test = ConditionalGenerator(num_classes=num_classes, 
                                                        generator_size=generator_size, 
                                                        latent_size=latent_size)
            
            path_to_best_weights = f"./models/gan/checkpoints/{dataset_name}/generator_final_model.pth"
            
            tester.test_gan(
                generator_arch=generator_arch_for_test, 
                path_to_weights=path_to_best_weights, 
                num_classes=num_classes, 
                latent_size=latent_size,
                img_size=img_size,
                channels=channels,
                dataset_name=dataset_name,
                images_per_class=10
            )

        if args.dm:
            path = f"./models/diffusion/checkpoints/{dataset_name}/final_model.pth"
            
            diffusion_model = DiffusionModel(marginal_prob_std= marginal_prob_std, 
                                   channels= [32, 64, 128, 256] if channels==1 else [128, 256, 512, 1024],
                                   embed_dim= 256 if channels==1 else 512,
                                   text_dim= 256 if channels==1 else 512,
                                   in_channels= channels, 
                                   img_size= high)
            model = tester.test_diffusion(model= diffusion_model, path= path)

            visualizer = Visualizer(
                sampler= diffusion_model.sampling_technique,
                marginal_fn= diffusion_model.marginal_prob_std,
                diffusion_coeff_fn= diffusion_coeff,
                device= device,
                dataset_name= dataset_name
            )

            visualizer.visualize_classes(diffusion_model, sample_batch_size= 16, num_steps= 250)
            visualizer.visualize_steps(diffusion_model, digit= 4, sample_batch_size= 16)


if __name__ == "__main__":
    args, dataset_name = parse_args()
    main(args, dataset_name)