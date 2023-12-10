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


def hook_activation_shaping(model: BaseResNet18, generate_M, every_n_convolution=1, skip_first_n_layers=0):
    #TODO: check if we'll need different M for each layer we apply the hook to
    #Make a list of all network convolutional layers to easily iterate over them
    all_layers = []
    output_size = 56

    #all_layers.append(model.resnet.conv1)
    for layer_group in [model.resnet.layer1, model.resnet.layer2, model.resnet.layer3, model.resnet.layer4]:
        for layer in layer_group:
            all_layers.append(layer.conv1)
            all_layers.append(layer.conv2)
    #Hook into the convolutional layers
    for i, layer in enumerate(all_layers):
        output_size = int((output_size + 2*layer.padding[0] - layer.kernel_size[0]) / layer.stride[0] + 1)
        if i % every_n_convolution == 0 and i >= skip_first_n_layers:
            #print(layer.out_channels);
            M = generate_M([output_size,output_size])
            layer.register_forward_hook(get_activation_shaping_hook(M))


def get_activation_shaping_hook(M):
    #Get M and binarize it
    
    def activation_shaping_hook(module, input, output):
        #Apply a transformation to the output
        #Output shape: [ batch_size, n_filters, n, n ]

        #Binarize output: if < 0 -> 0, else -> 1
        activation = torch.where(output < 0, torch.zeros_like(output), torch.ones_like(output))
        #Multiply output with M
        expanded_M = M.unsqueeze(0).unsqueeze(0).expand(activation.shape[0], activation.shape[1], -1, -1)
        output = torch.mul(activation, expanded_M)
        #Return output
        return output
    return activation_shaping_hook



######################################################
# TODO: either define the Activation Shaping Module as a nn.Module
#class ActivationShapingModule(nn.Module):
#...
#
# OR as a function that shall be hooked via 'register_forward_hook'
#def activation_shaping_hook(module, input, output):
#...
#
######################################################
# TODO: modify 'BaseResNet18' including the Activation Shaping Module
#class ASHResNet18(nn.Module):
#    def __init__(self):
#        super(ASHResNet18, self).__init__()
#        ...
#    
#    def forward(self, x):
#        ...
#
######################################################
