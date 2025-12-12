import torch
from torchviz import make_dot

class PlotModel():
    """Generate and plot the computational graph of a PyTorch model."""
    
    def __init__(self, model, device, in_channel=1, img_size=28):
        """Initialize the PlotModel helper with dynamic dimensions.

        Args:
            model (nn.Module): The PyTorch model to visualize.
            device (str): Device on which the model will run.
            in_channel (int): Number of input channels (e.g., 1 for MNIST, 3 for CIFAR).
            img_size (int): Height/Width of the input image (e.g., 28 or 32).
        """
        self.model = model
        self.device = device
        self.in_channel = in_channel
        self.img_size = img_size
        
        self.model.to(self.device)
        self.model.eval()
        
    def input_diffusion(self):
        """Create random diffusion inputs for testing the model using configured dims.

        Returns:
            tuple: Random tensors (img, t, labels) for a test forward pass.
        """
        batch_size = 2
        # Usa le variabili in_channel e img_size definite nell'init
        img = torch.randn(batch_size, self.in_channel, self.img_size, self.img_size, device=self.device) 
        t   = torch.rand(batch_size, device=self.device)               
        labels = torch.randint(0, 10, (batch_size,), device=self.device)   
        return img, t, labels
    
    def input_flow_match(self):
        """Create random inputs for FlowMatchingClassCond models."""
        batch_size = 2

        img = torch.randn(batch_size, self.in_channel, self.img_size, self.img_size, device=self.device)
        labels = torch.randint(0, 10, (batch_size,), device=self.device)

        return (img, labels)


    def plot_model(self, input, path):
        """Plot and save the model's computational graph.

        Args:
            input (tuple): Tuple containing model inputs.
            path (str): Directory where the graph will be saved.
        """

        y = self.model(*input)
        
        # Genera il grafico
        dot = make_dot(y, params=dict(self.model.named_parameters()))
        dot.format = 'png'
        dot.render(f'{path}/model_graph')