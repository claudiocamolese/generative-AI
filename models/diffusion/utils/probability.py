import torch
import numpy as np

def diffusion_coeff(t, sigma= 10., device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
        Compute the diffusion coefficient g(t) used in the SDE formulation of 
        score-based diffusion models (Variance-Exploding SDE).

        The VE SDE is defined as:
            dx = g(t) dW
        where     g(t) = sigma^t

        Args:
            t (Tensor): 
                Diffusion time values in the range [0, 1]. Can be a scalar tensor 
                or a batch tensor of shape (B,).
            sigma (float, optional): 
                Growth factor controlling how quickly noise increases during the 
                forward SDE. Higher values increase the spread between early and 
                late noise levels. Default is 25.
            device (str, optional): 
                Device on which to place the returned tensor. Defaults to 'cuda' 
                if available, otherwise 'cpu'.

        Returns:
            Tensor:
                Diffusion coefficient g(t) of shape (B,), same shape as `t`.
    """
    return (sigma**t).to(device)


def marginal_prob_std(t, sigma= 10., device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
        Compute the marginal standard deviation σ(t) of x_t in the forward 
        Variance-Exploding (VE) SDE model.

        For a VE SDE defined as:
            dx = g(t) dW,   with   g(t) = sigma^t,
        the marginal distribution at time t is:
            x_t ~ N(x_0, σ(t)^2 I)

        The closed-form expression of the marginal standard deviation is:
            σ(t) = sqrt( (sigma^(2t) - 1) / (2 log(sigma)) )

        Args:
            t (Tensor): 
                Diffusion time values in the range [0, 1], shape (B,).
            sigma (float, optional): 
                Noise scaling base parameter. Same meaning as in diffusion_coeff.
                Default is 25.
            device (str, optional): 
                Device for the output tensor. Default: CUDA if available.

        Returns:
            Tensor:
                Marginal standard deviation σ(t), same shape as `t`.
    """
    t = t.to(device)
    return torch.sqrt((sigma**(2 * t) - 1.) / (2. * np.log(sigma)))
