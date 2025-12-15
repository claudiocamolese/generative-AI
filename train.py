import comet_ml
import torch

from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from tqdm import tqdm
from loss import Loss
import torch.nn.functional as F

# --------- Diffusion -------------
from models.diffusion.utils.probability import diffusion_coeff, marginal_prob_std
from utils.printing import printing_model
from utils.printing import printing_train
# ---------------------------------

# ------------ GAN ----------------
from models.gan.gan import weights_init_gan 
# ---------------------------------


class Trainer():
    def __init__(self, train_loader, val_loader, device, track_flag, experiment, dataset_name) -> None:
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.track_flag = track_flag
        self.experiment = experiment 
        self.dataset_name = dataset_name
        
    
    def train_diffusion_model(self, model, epochs, lr):
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.model_name = "Diffusion_model"
        last_loss= float('inf')

        self.loss = Loss(model= self.model)        

        printing_train(self.model_name)

        if self.track_flag:
            self.experiment.log_parameter("learning_rate", self.lr)
            self.experiment.log_parameter("epochs", self.epochs)
            self.experiment.log_parameter("model_name", self.model_name)


        self.model.to(self.device)
        self.model.train()
        printing_model(model= self.model, model_name= self.model_name)

        optimizer = Adam(self.model.parameters(), lr= self.lr)
        #scheduler = StepLR(optimizer= optimizer, step_size=25)
        scheduler = CosineAnnealingLR(optimizer= optimizer, T_max= self.epochs, eta_min=1e-5)

        for epoch in range(self.epochs):
            avg_loss = 0.
            num_items = 0

            batch_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} Loss: 0.000")

            for step, (img_batch, label_batch) in enumerate(batch_bar):
                img_batch = img_batch.to(self.device)
                label_batch = label_batch.to(self.device)
                # if "ldm" in model_name:
                #     loss = loss_fn_cond_ldm(self.model, img_batch, label_batch, marginal_prob_std_fn)
                # else:
                marginal_fn = lambda t_: marginal_prob_std(t_, device=self.device)  
                loss = self.loss.loss_diffusion(img_batch= img_batch, label_batch= label_batch, marginal_prob_std= marginal_fn, device= self.device)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                avg_loss += loss.item() * img_batch.shape[0]
                num_items += img_batch.shape[0]
                
                batch_bar.set_description(f"Epoch {epoch + 1} Loss: {loss.item():.4f}")
            
                if self.track_flag and (step % 40 == 0):
                    self.experiment.log_metric("batch_loss", loss.item(), step=epoch * len(self.train_loader) + step)
                        
            scheduler.step()
            lr_current = scheduler.get_last_lr()[0]

            epoch_loss = avg_loss / num_items
            print('{} Average Loss: {:5f} lr {:.1e}'.format(epoch + 1, epoch_loss, lr_current))
            
            if self.track_flag:
                self.experiment.log_metric("epoch_loss", epoch_loss, step=epoch)
                self.experiment.log_metric("learning_rate", lr_current, step=epoch)

            if epoch_loss < last_loss:
                torch.save(self.model.state_dict(), f"./models/diffusion/checkpoints/{self.dataset_name}/final_model.pth")
                last_loss = epoch_loss
                print(f"New model saved in models/diffusion/checkpoints/{self.dataset_name}/ !")

            if self.track_flag:    
                self.experiment.log_asset(f"./models/diffusion/checkpoints/{self.dataset_name}/final_model.pth")

    def train_gan_model(self, generator, discriminator, epochs, g_lr, d_lr, n_critic, latent_size, num_classes, gp_lambda=10.0):
        """
        Training for Conditional Generator and Discriminator.
        Uses WGAN-GP upgrades for stability
        """
        self.generator = generator
        self.discriminator = discriminator
        self.loss = Loss(model=self.discriminator)
        self.epochs = epochs
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.n_critic = n_critic  # Number of upgrades of the discriminator between two upgrades of the generator
        self.latent_size = latent_size
        self.num_classes = num_classes
        self.gp_lambda = gp_lambda
        self.model_name = "Conditional_GAN"
        
        best_d_loss = float('inf')
        g_losses, d_losses = [], []
        iters = 0
        
        # Class preparation for fixed noise
        # all_labels is the identity matrix (one-hot embedding)
        all_labels = torch.eye(self.num_classes, dtype=torch.float32, device=self.device)
        
        printing_train(self.model_name)

        if self.track_flag:
            self.experiment.log_parameter("g_learning_rate", self.g_lr)
            self.experiment.log_parameter("d_learning_rate", self.d_lr)
            self.experiment.log_parameter("epochs", self.epochs)
            self.experiment.log_parameter("n_critic", self.n_critic)
            self.experiment.log_parameter("gp_lambda", self.gp_lambda)
            self.experiment.log_parameter("model_name", self.model_name)


        self.generator.apply(weights_init_gan)
        self.discriminator.apply(weights_init_gan)
        
        self.generator.to(self.device).train()
        self.discriminator.to(self.device).train()
        
        printing_model(model=self.generator, model_name="Generator")
        printing_model(model=self.discriminator, model_name="Discriminator")

        # optimizers
        adamw_lambda = 0.0
        g_optimizer = AdamW(self.generator.parameters(), lr=self.g_lr, betas=(0., 0.9))
        d_optimizer = AdamW(self.discriminator.parameters(), lr=self.d_lr, betas=(0., 0.9))
                
        for epoch in range(self.epochs):
            
            batch_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")

            for step, (real_images, real_class_indices) in enumerate(batch_bar):
                
                # real_class_indices assumed as a tensor of indices
                batch_size = real_images.size(0)
                
                real_images = real_images.to(self.device)
                # Convertion of indices in one-hot vectors
                real_class_labels = all_labels[real_class_indices].to(self.device) 
                
                # --- Discriminator upgrade ---
                d_optimizer.zero_grad()
                
                # 1. Output on real images (V(D, G))
                d_output_real = self.discriminator(real_images, real_class_labels)
                
                # 2. generation of fake images for discriminator
                noise = torch.randn((batch_size, self.latent_size), device=self.device)
                
                with torch.no_grad(): 
                    fake_image = self.generator(noise, real_class_labels) 
                
                d_output_fake = self.discriminator(fake_image, real_class_labels)
                
                d_loss = self.loss.discriminator_loss(real_images, fake_image, d_output_real, d_output_fake, real_class_labels, self.gp_lambda, batch_size, self.device)

                # Backpropagation and upgrade
                d_loss.backward()
                d_optimizer.step()
                
                
                # --- Generator Upgrade ---
                if step % self.n_critic == 0:
                    g_optimizer.zero_grad()
                    
                    # casual class labels for a robust training
                    fake_class_indices = torch.randint(self.num_classes, size=[batch_size], device=self.device)
                    fake_class_labels = all_labels[fake_class_indices]
                    
                    # New fake image
                    noise = torch.randn((batch_size, self.latent_size), device=self.device)
                    fake_image = self.generator(noise, fake_class_labels)
                    
                    # discriminator output on fake image
                    d_output_fake_for_g = self.discriminator(fake_image, fake_class_labels)
                    
                    g_loss = self.loss.generator_loss(d_output_fake_for_g)
                    
                    # Backpropagation and upgrade
                    g_loss.backward()
                    g_optimizer.step()
                    
                    g_losses.append(g_loss.item())
                
                d_losses.append(d_loss.item())
                
                batch_bar.set_description(f"Epoch {epoch + 1} | D_Loss: {d_loss.item():.4f} | G_Loss: {g_loss.item():.4f}")

                if self.track_flag and (step % 40 == 0):
                    self.experiment.log_metric("D_batch_loss", d_loss.item(), step=epoch * len(self.train_loader) + step)
                    if step % self.n_critic == 0:
                        self.experiment.log_metric("G_batch_loss", g_loss.item(), step=epoch * len(self.train_loader) + step)
                        
                iters += 1
            
            avg_g_loss = sum(g_losses[-len(self.train_loader)//self.n_critic:]) / (len(self.train_loader)//self.n_critic) if len(g_losses) > 0 else 0
            avg_d_loss = sum(d_losses[-len(self.train_loader):]) / len(self.train_loader) if len(d_losses) > 0 else 0
            
            print('{} Average D_Loss: {:5f} | Average G_Loss: {:5f}'.format(
                epoch + 1, avg_d_loss, avg_g_loss
            ))
            
            if self.track_flag:
                self.experiment.log_metric("epoch_D_loss", avg_d_loss, step=epoch)
                self.experiment.log_metric("epoch_G_loss", avg_g_loss, step=epoch)

            if abs(avg_d_loss) < abs(best_d_loss):
                pass
            
            torch.save(self.generator.state_dict(), f"./models/gan/checkpoints/{self.dataset_name}/generator_final_model.pth")
            torch.save(self.discriminator.state_dict(), f"./models/gan/checkpoints/{self.dataset_name}/discriminator_final_model.pth")
            best_d_loss = avg_d_loss
            print(f"New models saved in models/gan/checkpoints/{self.dataset_name}/ !")

            if self.track_flag:    
                self.experiment.log_asset(f"./models/gan/checkpoints/{self.dataset_name}/generator_final_model.pth")
                self.experiment.log_asset(f"./models/gan/checkpoints/{self.dataset_name}/discriminator_final_model.pth")

    def train_vae_model(self):
        pass
    
    def train_flowbased_model(self):
        pass