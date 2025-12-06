import torch
import numpy as np

def diffusion_coeff(t, sigma = 25., device = 'cuda' if torch.cuda.is_available() else 'cpu'):
    return (sigma**t).to(device)

def marginal_prob_std(t, sigma = 25., device= 'cuda' if torch.cuda.is_available() else 'cpu'):
    t = t.to(device)
    return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))