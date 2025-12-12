import os
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np

class FlowVisualizer:
    """
    Visualizer per Flow Matching Model.
    Permette di generare immagini condizionate sulle classi MNIST.
    """

    def __init__(self, model, dataset_name, device=None):
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset_name = dataset_name

        os.makedirs(f"./figures/flow_based/{self.dataset_name}/classes", exist_ok=True)
        os.makedirs(f"./figures/flow_based/{self.dataset_name}/steps", exist_ok=True)

    # --------------------------
    def visualize_classes(self, sample_batch_size=16):
        """Genera immagini per tutte le classi (0-9)"""

        self.model.eval()
        plt.figure(figsize=(10, 4))

        for i, digit in enumerate(range(10)):
            y = digit * torch.ones(sample_batch_size, dtype=torch.long).to(self.device)
            samples = self.model.sample(c=y, img_size=self.model.img_size, num_samples=sample_batch_size)
            samples = samples.clamp(0.0, 1.0)
            sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))

            plt.subplot(2, 5, i + 1)
            plt.title(f"Class: {digit}")
            plt.axis("off")
            plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)

        plt.tight_layout()
        path = f"./figures/flow_based/{self.dataset_name}/classes/all_classes.png"
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"Saved figure to: {path}")

    # --------------------------
    def visualize_steps(self, digit=4, sample_batch_size=16):
        """Genera immagini di un singolo digit (condizionato)"""

        self.model.eval()
        plt.figure(figsize=(4, 4))
        y = digit * torch.ones(sample_batch_size, dtype=torch.long).to(self.device)
        samples = self.model.sample(c=y, img_size=self.model.img_size, num_samples=sample_batch_size)
        samples = samples.clamp(0.0, 1.0)
        sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))

        plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
        plt.axis("off")
        plt.title(f"Digit: {digit}")

        path = f"./figures/flow_based/{self.dataset_name}/steps/digit_{digit}.png"
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"Saved figure to: {path}")
