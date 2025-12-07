import os
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np


class Visualizer:
    """
        Utility class for visualizing the outputs of diffusion models.

        It provides two visualization modes:
        - visualize_classes(): generate samples for all 10 MNIST classes
        - visualize_steps(): generate samples for a fixed class varying the number of diffusion steps

        Images are automatically saved under:
        - figures/diffusion/stable/
        - figures/diffusion/steps/
    """

    def __init__(self, sampler, marginal_fn, diffusion_coeff_fn, config, device="cuda" if torch.cuda.is_available() else 'cpu'):
        """
            Initialize the Visualizer.

            Args:
                sampler: Sampling function (e.g., Euler-Maruyama, ODE, PC sampler)
                marginal_fn: Function returning marginal probability std
                diffusion_coeff_fn: Diffusion coefficient function
                device: Computation device ("cuda" or "cpu")
        """

        self.sampler = sampler
        self.marginal_fn = marginal_fn
        self.diffusion_coeff_fn = diffusion_coeff_fn
        self.device = device
        self.config = config

        # Create output folders
        os.makedirs(f"./figures/diffusion/{self.config['dataset']['name']}/stable", exist_ok=True)
        os.makedirs(f"./figures/diffusion/{self.config['dataset']['name']}/steps", exist_ok=True)

    # ----------------------------------------------------------------------

    def visualize_classes(self, model, sample_batch_size=16, num_steps=250):
        """
            Generate samples for digits 0-9 with a fixed number of sampling steps.

            Args:
                model: Trained diffusion model (score model)
                sample_batch_size: Number of samples per class
                num_steps: Number of denoising steps

            Saves:
                figures/diffusion/stable/classes_{num_steps}steps.png
        """

        model.eval()
        plt.figure(figsize=(10, 4))

        for i, digit in enumerate(range(10)):

            y = digit * torch.ones(sample_batch_size, dtype= torch.long).to(self.device)

            samples = self.sampler(
                num_steps= num_steps,
                batch_size= sample_batch_size,
                device= self.device,
                y= y
            )

            samples = samples.clamp(0.0, 1.0)
            sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))

            plt.subplot(2, 5, i + 1)
            plt.title(f"Class: {digit}")
            plt.axis("off")
            plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin= 0., vmax= 1.)

        plt.tight_layout()
        path = f"./figures/diffusion/{self.config['dataset']['name']}/stable/classes_{num_steps}_steps.png"
        plt.savefig(path, dpi=300)
        plt.close()

        print(f"Saved figure to: {path}")

    # ----------------------------------------------------------------------

    def visualize_steps(self, model, digit= 4, sample_batch_size= 16, num_steps_list= None):
        """
            Generate samples for a fixed digit while varying the number of diffusion steps.

            Args:
                model: Trained diffusion model
                digit: Digit to condition on (0-9)
                sample_batch_size: Number of samples per setting
                num_steps_list: List of step counts to test. Defaults to:
                                [10, 50, 100, 200, 500, 1000]

            Saves:
                figures/diffusion/steps/digit_{digit}_steps.png
        """

        if num_steps_list is None:
            num_steps_list = [10, 50, 100, 200, 500, 2000]

        model.eval()
        plt.figure(figsize=(3 * len(num_steps_list), 3))

        y = digit * torch.ones(sample_batch_size, dtype=torch.long).to(self.device)

        for i, num_steps in enumerate(num_steps_list):

            samples = self.sampler(
                num_steps= num_steps,
                batch_size= sample_batch_size,
                device= self.device,
                y=y)

            samples = samples.clamp(0.0, 1.0)
            sample_grid = make_grid(samples, nrow= int(np.sqrt(sample_batch_size)))

            plt.subplot(1, len(num_steps_list), i + 1)
            plt.title(f"Steps: {num_steps}")
            plt.axis("off")
            plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin= 0., vmax= 1.)

        plt.tight_layout()
        path = f"./figures/diffusion/{self.config['dataset']['name']}/steps/class_{digit}_steps.png"
        plt.savefig(path, dpi= 300)
        plt.close()

        print(f"Saved figure to: {path}")
