import comet_ml
import torch

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from tqdm import tqdm

# --------- Diffusion -------------
from models.diffusion.utils.probability import diffusion_coeff, marginal_prob_std
from printing import printing_model
from models.diffusion.utils.loss import loss_diffusion
from printing import printing_train
# ---------------------------------


class Trainer():
    def __init__(self, train_loader, val_loader, device, track_flag, experiment, config) -> None:
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.track_flag = track_flag
        self.experiment = experiment 
        self.config = config
    
    def train_diffusion_model(self, model, epochs, lr):
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.model_name = "Diffusion_model"
        last_loss= float('inf')

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
                loss = loss_diffusion(model= self.model, img_batch= img_batch, label_batch= label_batch, marginal_prob_std= marginal_fn, device= self.device)
                
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
            print('{} Average Loss: {:5f} lr {:.1e}'.format(epoch, epoch_loss, lr_current))
            
            if self.track_flag:
                self.experiment.log_metric("epoch_loss", epoch_loss, step=epoch)
                self.experiment.log_metric("learning_rate", lr_current, step=epoch)

            if epoch_loss < last_loss:
                torch.save(self.model.state_dict(), f"./models/diffusion/checkpoints/{self.config['dataset']['name']}/final_model.pth")
                last_loss = epoch_loss
                print(f"New model saved in models/diffusion/checkpoints/{self.config['dataset']['name']}/ !")

            if self.track_flag:    
                self.experiment.log_asset(f"./models/diffusion/checkpoints/{self.config['dataset']['name']}/final_model.pth")

    def train_gan_model(self):
        pass
    
    def train_vae_model(self):
        pass
    
    def train_flowbased_model(self):
        pass