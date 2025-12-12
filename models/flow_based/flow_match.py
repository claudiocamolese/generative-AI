import torch.nn as nn
import torch

class FlowMatching(nn.Module):
    def __init__(self, denoiser, prob_uncond=0.2, num_ts = 50, img_size = (28,28), 
                 device = "cuda" if torch.cuda.is_available() else "cpu"):
        
        super().__init__()
        self.denoiser = denoiser
        self.prob_uncond = prob_uncond
        self.num_ts = num_ts
        self.img_size = img_size
        self.device = device

    def forward(self, img_batch, label_batch):
        self.denoiser.train()
        # Randomly sample intermediate noise
        x_0 = torch.randn_like(img_batch).to(self.device)
        target = img_batch - x_0
        t = torch.rand((x_0.shape[0],)).to(self.device)
        x_t = x_0 + t.view(*t.shape, 1, 1, 1)*(img_batch - x_0)

          # Apply mask to true classes with probability p_uncond
        mask = torch.bernoulli((1-self.prob_uncond) * torch.ones((label_batch.shape[0],)))
        mask = mask.to(self.device)
        
        # Compute squared error between true and predicted deviation from sample image 
        output = self.denoiser(x_t, label_batch, t, mask)
        return output, target
    
    @torch.inference_mode()
    def sample(
        self,
        c: torch.Tensor,
        img_size: tuple[int, int],
        num_samples: int,
        guidance_scale: float = 5.0,
        seed: int = 0,
    ):
        """
        Generates images using the flow-matching model with classifier-free guidance.
        """

        self.denoiser.eval()

        # Fix random seed if provided
        if seed:
            torch.manual_seed(seed)

        # Detect correct number of channels from the denoiser itself
        C = self.denoiser.conv1.in_channels

        # Initial noise (num_samples, C, H, W)
        x_t = torch.randn((num_samples, C, img_size[0], img_size[1]),
                        device=self.device)

        # Initial time (all zeros)
        t = torch.zeros((num_samples,), device=self.device)

        # Move class labels to device
        c = c.to(self.device)

        # Step size for straight-line flow
        step_size = 1.0 / self.num_ts

        # Flow matching integration loop
        while t[0].item() < 1.0:

            # Mask for unconditional and conditional passes
            mask_uncond = torch.zeros((c.shape[0],), device=self.device)
            mask_cond   = torch.ones((c.shape[0],), device=self.device)

            # Unconditional and conditional velocities
            u_uncond = self.denoiser(x_t, c, t, mask_uncond)
            u_cond   = self.denoiser(x_t, c, t, mask_cond)

            # Classifier-free guidance
            u = u_uncond + guidance_scale * (u_cond - u_uncond)

            # Euler update
            x_t = x_t + step_size * u
            t   = t   + step_size

        return x_t
