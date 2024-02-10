import torch
import torch.nn as nn
from torch.types import Device
from torchvision.models import resnet18, ResNet18_Weights
from globals import CONFIG

def get_domain_adaptation_hook(model, i):
        def activation_shaping_hook(module, input, output):

            #RECORD mode: record the activations using the chosen record_mode function
            if (model.state == DomainAdaptationMode.RECORD):
                model.M[i] = model.record_mode(model, output)

            #Apply mode: apply M as in the previous versions   
            elif (model.state == DomainAdaptationMode.APPLY):
                if CONFIG.experiment in ['domain_adaptation']:
                    activation = model.record_mode(model, output)
                else:
                    activation = output
                output = torch.mul(activation, model.M[i])
            
            #TEST_BINARIZE mode: binarize the output using the chosen record_mode function
            elif (model.state == DomainAdaptationMode.TEST_BINARIZE):
                if CONFIG.experiment in ['domain_adaptation']:
                    output = model.record_mode(model, output)
                elif CONFIG.experiment in ['domain_adaptation_alt']:
                    output = torch.mul(output, model.record_mode(model, output))
            
            #In TEST mode, do nothing
                
            return output
        return activation_shaping_hook

class ResNet18ForDomainAdaptation(nn.Module):
    def __init__(self, record_mode, K, layers_to_apply):
        super(ResNet18ForDomainAdaptation, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)
        self.record_mode = record_mode
        self.K = K
        #This net has a state: RECORD if it is recording the activations, APPLY if it is applying the activations
        self.state = DomainAdaptationMode.RECORD

        #Register hooks
        #Make a list of all network convolutional layers to easily iterate over them
        all_layers = []
        for layer_group in [self.resnet.layer1, self.resnet.layer2, self.resnet.layer3, self.resnet.layer4]:
            for layer in layer_group:
                all_layers.append(layer.conv1)
                all_layers.append(layer.conv2)
        #Hook into the convolutional layers
        for l in layers_to_apply:
            all_layers[l].register_forward_hook(get_domain_adaptation_hook(self, l))

        
        #Use an array to keep last generated M
        self.M = [None for _ in range(len(layers_to_apply))]

    def forward(self, x):
        return self.resnet(x)
