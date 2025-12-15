import comet_ml
import torch.nn as nn
import torchvision.utils as vutils
import torch
import os

from models.diffusion.utils.probability import marginal_prob_std
from tqdm import tqdm
from utils.printing import printing_test
from loss import Loss

from models.gan.visualizer import generate_image

class Tester():
    def __init__(self, test_loader, device, track_flag, experiment):
        self.test_loader = test_loader
        self.device = device
        self.track_flag = track_flag
        self.experiment = experiment

    def test_diffusion(self, model, path):
        self.model = model.to(self.device)
        model_name = 'Diffusion_model'
        self.loss = Loss(model= self.model)
        
        self.model.load_state_dict(torch.load(path, weights_only=True, map_location= self.device))
        self.model.eval()

        printing_test(model_name)

        batch_bar = tqdm(self.test_loader, desc="Test Loss: 0.000")

        for step, (img_batch, label_batch) in enumerate(batch_bar):

            self.img_batch = img_batch.to(self.device)
            self.label_batch = label_batch.to(self.device)

            loss = self.loss.loss_diffusion(img_batch= self.img_batch, label_batch= self.label_batch, marginal_prob_std= marginal_prob_std)

            batch_bar.set_description(f"Test Loss: {loss.item():.4f}")

            if self.track_flag:
                self.experiment.log_metric("test_batch_loss", loss.item(), step=step)

        avg_loss = sum(loss.item() * img_batch.size(0) for img_batch, label_batch in self.test_loader) / len(self.test_loader.dataset)
        print(f"Test loss: {avg_loss}")
        
        if self.track_flag:
            self.experiment.log_metric("test_loss", avg_loss)

        return self.model
    
    def test_gan(self, generator_arch, path_to_weights, num_classes, latent_size, img_size, channels, dataset_name, images_per_class=10):
        """
        Loads the best Generator, calculates the quantitative Reconstruction Loss on the 
        test set, and generates visual comparisons.
        """
        model_name = 'Conditional_GAN_Generator'
        # Assuming 'printing_test' is an auxiliary function
        # printing_test(model_name) 

        # 1. Initialize Generator and Load Weights
        # generator_arch is an untrained instance of ConditionalGenerator
        self.generator = generator_arch.to(self.device)
        
        if not os.path.exists(path_to_weights):
            print(f"Error: Generator checkpoint not found at {path_to_weights}. Aborting test.")
            return

        print(f"Loading best generator state from: {path_to_weights}")
        self.generator.load_state_dict(torch.load(path_to_weights, map_location=self.device))
        self.generator.eval() # Set the generator to evaluation mode
        
        # Define the loss function for reconstruction (L1 Loss is common for pixel-wise comparison)
        # Instantiate the loss function here for the quantitative evaluation
        l1_loss_fn = nn.L1Loss()
        total_recon_loss = 0.0
        
        # Prepare one-hot class labels for easy conditioning
        all_labels = torch.eye(num_classes, dtype=torch.float32, device=self.device)


        # 2. QUANTITATIVE EVALUATION: Calculate Reconstruction Loss
        if hasattr(self, 'test_loader') and self.test_loader is not None:
            print("Starting Reconstruction Loss calculation on test set...")
            
            with torch.no_grad(): # Disable gradient calculations
                
                for real_images, real_class_indices in tqdm(self.test_loader, desc="Calculating Recon Loss"):
                    
                    batch_size = real_images.size(0)
                    real_images = real_images.to(self.device)
                    
                    # Convert class indices to one-hot vectors for conditioning
                    real_class_labels = all_labels[real_class_indices].to(self.device)
                    
                    # Generate random noise (z)
                    noise = torch.randn((batch_size, latent_size), device=self.device)
                    
                    # Generate fake images conditioned on the real class labels
                    fake_images = self.generator(noise, real_class_labels)
                    
                    # Calculate the L1 loss (pixel-wise difference) between Fake and Real
                    loss = l1_loss_fn(fake_images, real_images)
                    
                    # Accumulate the loss, weighted by the batch size
                    total_recon_loss += loss.item() * batch_size 
            
            # Calculate the average reconstruction loss over the entire test set
            avg_recon_loss = total_recon_loss / len(self.test_loader.dataset)
            
            print(f"\n==============================================")
            print(f"✅ TEST RECONSTRUCTION LOSS (L1): {avg_recon_loss:.6f}")
            print(f"==============================================")
            
            if self.track_flag:
                self.experiment.log_metric("test_reconstruction_L1_loss", avg_recon_loss)
                
        else:
            print("Warning: self.test_loader is not defined. Skipping Reconstruction Loss calculation.")


        # 3. QUALITATIVE EVALUATION: Generate Visual Comparison Grid (Unchanged)
        
        all_comparison_images = []
        
        # Search for enough real images for all classes
        real_images_dict = {}
        for img_batch, label_batch in self.test_loader:
            for img, label in zip(img_batch, label_batch):
                label_idx = label.item()
                if label_idx not in real_images_dict:
                    real_images_dict[label_idx] = []
                if len(real_images_dict[label_idx]) < images_per_class:
                    real_images_dict[label_idx].append(img)
            
            # Exit when we have enough images for all classes of interest
            if all(len(real_images_dict.get(i, [])) >= images_per_class for i in range(num_classes)):
                break
        
        # Generation and Comparison Class by Class
        
        for class_idx in range(num_classes):
            print(f"Processing Class {class_idx}...")
            
            # A. Generate Fake Images (10 per class)
            # Assuming 'generate_image' is an auxiliary function
            fake_images = generate_image(
                generator=self.generator, 
                class_index=class_idx, 
                num_images=images_per_class, 
                latent_size=latent_size,
                num_classes=num_classes,
                device=self.device
            )
            
            # B. Retrieve Real Images (10 per class)
            real_images = real_images_dict.get(class_idx, [])
            if len(real_images) < images_per_class:
                # Handle missing real images
                padding = torch.zeros(channels, img_size, img_size) 
                real_images.extend([padding] * (images_per_class - len(real_images)))
                
            real_images = torch.stack(real_images)
            
            # C. Combine Real and Fake batches for visualization grid
            # Appends Real images batch, then Fake images batch (resulting in 2 rows per class)
            all_comparison_images.append(real_images) 
            all_comparison_images.append(fake_images) 

        # 4. Create the Final Grid
        final_comparison_tensor = torch.cat(all_comparison_images, dim=0)
        
        # Create grid (10 columns, with alternating Real/Fake rows for each class)
        grid = vutils.make_grid(final_comparison_tensor, 
                                padding=2, 
                                normalize=True, 
                                nrow=images_per_class)
        
        # 5. Save the Result
        save_path = f"./figures/gan/{dataset_name}/test_comparison_cgan.png"
        vutils.save_image(grid, save_path)
        print(f"Comparison grid saved to: {save_path}")
        
        if self.track_flag:
            self.experiment.log_asset(save_path)

        return self.generator