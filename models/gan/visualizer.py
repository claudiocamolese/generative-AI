import torch
import matplotlib.pyplot as plt
import os

def generate_image(generator, class_index, num_images, latent_size, num_classes, device):
        """
        Generate a batch of images refered to a image class
        
        Args:
            generator (nn.Module): trained Conditional Generator model
            class_index (int): index class to generate.
            num_images (int): number of images to generate.
            latent_size (int): noise vector dimension.
            num_classes (int): Total number of classes.
            
        Returns:
            torch.Tensor: tensor of generated images
        """
        generator.eval()

        all_labels = torch.eye(num_classes, dtype=torch.float32, device=device)
        
        condition_vector = all_labels[class_index].repeat(num_images, 1).to(device)
        
        noise = torch.randn((num_images, latent_size), device=device)
        
        with torch.no_grad():
            fake_images = generator(noise, condition_vector).cpu()
            
        return fake_images