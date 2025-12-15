import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd


class Loss(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.model = model

    def loss_diffusion(self, img_batch, label_batch, marginal_prob_std,eps=1e-5, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Computes the loss for a conditional denoising diffusion probabilistic model (DDPM).

        Args:
            model: The neural network model that predicts the score (i.e., the gradient of the log probability).
            img_batch (torch.Tensor): The original data samples (e.g., images) with shape (batch_size, channels, height, width).
            label_batch (torch.Tensor): The conditional information (e.g., class labels or other auimg_batchiliary data).
            marginal_prob_std (function): A function that returns the standard deviation of the noise at a given time step.
            eps (float, optional): A small value to ensure numerical stability. Default is 1e-5.

        Returns:
            torch.Tensor: The computed loss as a scalar tensor.
        """
    
        # Sample a random time step for each sample in the batch. t ∈ [eps, 1]
        random_t = torch.rand(img_batch.shape[0], device= device) * (1. - eps) + eps
        
        # Sample random noise from a standard normal distribution with the same shape as the input.
        z = torch.randn_like(img_batch).to(device= device)
        
        # Compute the standard deviation of the noise at the sampled time step.
        std = marginal_prob_std(random_t).to(device)
        
        # Perturb the input data with the sampled noise, scaled by the computed standard deviation.
        perturbed_img_batch = img_batch + z * std[:, None, None, None]
        
        # Predict the score (denoising direction) using the model.
        # The model takes the perturbed data, the time step, and the conditional information as inputs.
        score = self.model(img_batch= perturbed_img_batch, t = random_t, label_batch =label_batch)
        
        # Compute the loss as the mean squared error between the predicted score and the true noise,
        # weighted by the standard deviation.
        #loss = torch.mean(torch.sum((score * std[:, None, None, None] - z)**2, dim=(1,2,3)))
        loss = F.mse_loss(score * std[:, None, None, None], -z, reduction='mean')
        
        return loss
    
    def generator_loss(self, fake_output):
        # G_loss = -E[D(G(z))] (Objective: max(E[D(G(z))]))
        return -fake_output.mean()

    def discriminator_loss(self, real_images, fake_image, real_output, fake_output, real_class_labels, gp_lambda, batch_size, device):
        d_loss_real = real_output.mean()
        d_loss_fake = fake_output.mean()
        
        alpha = torch.rand((batch_size, 1), device=device)
        alpha = alpha.view(-1, 1, 1, 1)
        
        interpolates = (alpha * real_images + (1. - alpha) * fake_image).requires_grad_(True)
        
        d_interpolates = self.model(interpolates, real_class_labels)
        
        grad_tensor = torch.ones(d_interpolates.size(), device=device)
        
        # Compute gradients of D(x_hat) w.r.t. x_hat
        gradients = autograd.grad(
            outputs=d_interpolates, 
            inputs=interpolates, 
            grad_outputs=grad_tensor, 
            create_graph=True, 
            only_inputs=True
        )[0]
        gradient_penalty = gp_lambda * ((gradients.view(batch_size, -1).norm(dim=1) - 1.) ** 2).mean()
        
        # D_loss = E[D(G(z))] - E[D(x)] + GP (WGAN objective: max(E[D(x)] - E[D(G(z))]))
        return d_loss_fake - d_loss_real + gradient_penalty