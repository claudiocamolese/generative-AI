import torch
from torchviz import make_dot

from models.diffusion.diffusion import DiffusionModel

class PlotModel():
    def __init__(self, model):
        self.model = model
        self.model.eval()
        
    def input_diffusion(self):
        batch_size = 2
        img = torch.randn(batch_size, 1, 28, 28) 
        t   = torch.rand(batch_size)               
        labels = torch.randint(0, 10, (batch_size,))   
        return img, t, labels


    def plot_model(self, input, path):
        y = self.model(*input)
        dot = make_dot(y, params=dict(self.model.named_parameters()))
        dot.format = 'png'
        dot.render(f'{path}/model_graph') 
