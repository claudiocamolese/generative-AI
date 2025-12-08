import torch
from torchviz import make_dot

class PlotModel():
    """Generate and plot the computational graph of a PyTorch model."""
    
    def __init__(self, model, device):
        """Initialize the PlotModel helper.

        Args:
            model (_type_): The PyTorch model to visualize.
            device (_type_): Device on which the model will run.
        """
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        
    def input_diffusion(self):
        """Create random diffusion inputs for testing the model.

        Returns:
            tuple: Random tensors (img, t, labels) for a test forward pass.
        """
        batch_size = 2
        img = torch.randn(batch_size, 1, 28, 28, device= self.device) 
        t   = torch.rand(batch_size, device= self.device)               
        labels = torch.randint(0, 10, (batch_size,), device= self.device)   
        return img, t, labels


    def plot_model(self, input, path):
        """Plot and save the model's computational graph.

        Args:
            input (_type_): Tuple containing model inputs.
            path (_type_): Directory where the graph will be saved.
        """
        y = self.model(*input)
        dot = make_dot(y, params=dict(self.model.named_parameters()))
        dot.format = 'png'
        dot.render(f'{path}/model_graph') 
