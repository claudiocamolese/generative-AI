import comet_ml
import torch.nn as nn
import torch

from models.diffusion.utils.probability import marginal_prob_std
from models.diffusion.utils.loss import loss_diffusion
from tqdm import tqdm
from printing import printing_test

class Tester():
    def __init__(self, test_loader, device, track_flag, experiment):
        self.test_loader = test_loader
        self.device = device
        self.track_flag = track_flag
        self.experiment = experiment

    def test_diffusion(self, model, path):
        self.model = model.to(self.device)
        model_name = 'Diffusion_model'
        
        self.model.load_state_dict(torch.load(path, weights_only=True, map_location= self.device))
        self.model.eval()

        printing_test(model_name)

        batch_bar = tqdm(self.test_loader, desc="Test Loss: 0.000")

        for step, (img_batch, label_batch) in enumerate(batch_bar):

            self.img_batch = img_batch.to(self.device)
            self.label_batch = label_batch.to(self.device)

            loss = loss_diffusion(model= self.model, img_batch=img_batch, label_batch=label_batch, marginal_prob_std= marginal_prob_std)

            batch_bar.set_description(f"Test Loss: {loss.item():.4f}")

            if self.track_flag:
                self.experiment.log_metric("test_batch_loss", loss.item(), step=step)

        avg_loss = sum(loss.item() * img_batch.size(0) for img_batch, label_batch in self.test_loader) / len(self.test_loader.dataset)
        print(f"Test loss: {avg_loss}")
        
        if self.track_flag:
            self.experiment.log_metric("test_loss", avg_loss)

        return self.model