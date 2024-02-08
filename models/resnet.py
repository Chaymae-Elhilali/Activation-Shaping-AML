import torch
import torch.nn as nn
from torch.types import Device
from torchvision.models import resnet18, ResNet18_Weights
from globals import CONFIG

class BaseResNet18(nn.Module):
    def __init__(self):
        super(BaseResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)

    def forward(self, x):
        return self.resnet(x)
    
    def disable_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
    
    def enable_hooks(self):
        for index, layer in enumerate(self.layers_with_hooks):
            self.hook_handles[index] = layer.register_forward_hook(self.hook_fun)


    def hook_activation_shaping(self, alpha=0.5, every_n_convolution=1, skip_first_n_layers=0):
        
        def M_random_generator(shape, alpha):
            #Generate a random matrix M with values 0 or 1
            M = torch.ones(shape, device='cuda:0')
            M = torch.where(torch.rand(shape, device='cuda:0') <= alpha, M, torch.zeros(shape, device='cuda:0')).to(CONFIG.device)
            return M

        #Get activation shaping hook returns a function that can be used as a hook while containing the M matrix
        def get_activation_shaping_hook(generate_M, alpha):
            def activation_shaping_hook(module, input, output):
                #Apply a transformation to the output
                #Binarize output: if < 0 -> 0, else -> 1
                activation = torch.where(output <= 0, torch.zeros_like(output), torch.ones_like(output))
                #Multiply output with random M
                output = torch.mul(activation, generate_M(activation.shape, alpha))
                #Return output
                return output
            return activation_shaping_hook

        #Make a list of all network convolutional layers to easily iterate over them
        all_layers = []
        layers_with_hooks = []
        hook_handles = []
        hook_fun = get_activation_shaping_hook(M_random_generator, alpha)

        for layer_group in [self.resnet.layer1, self.resnet.layer2, self.resnet.layer3, self.resnet.layer4]:
            for layer in layer_group:
                all_layers.append(layer.conv1)
                all_layers.append(layer.conv2)
        #Hook into the convolutional layers
        n_applied = 0
        for i, layer in enumerate(all_layers):
            if (i >= skip_first_n_layers) and (every_n_convolution==0 or ((i-skip_first_n_layers) % every_n_convolution == 0)):
                layers_with_hooks.append(layer)
                hook_handles.append(layer.register_forward_hook(hook_fun))
                n_applied += 1
        
        self.layers_with_hooks = layers_with_hooks
        self.hook_handles = hook_handles
        self.hook_fun = hook_fun
        print(f"Applied to {n_applied} layers")

