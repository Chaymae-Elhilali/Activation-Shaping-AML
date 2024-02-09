import torch
import torch.nn as nn
from torch.types import Device
from torchvision.models import resnet18, ResNet18_Weights
from models.resnet_domain_adaptation import DomainAdaptationMode
from globals import CONFIG

def get_domain_adaptation_hook(model, i):
        def activation_shaping_hook(module, input, output):

            #RECORD mode: record the activations using the chosen record_mode function
            if (model.state == DomainAdaptationMode.RECORD):
                M = model.record_mode(model, output)
                #To handle the different Mi, we can simply multiply the new M with the product of all the previous
                #or just set model.M[i] = M if it's the first
                if model.M[i] is None:
                    model.M[i] = M
                else:
                    model.M[i] = torch.mul(model.M[i], M)

            #Apply mode: apply M as in the previous versions   
            elif (model.state == DomainAdaptationMode.APPLY):
                activation = model.record_mode(model, output)
                #model.M[i] is already the product of all the Mi
                activation = torch.mul(activation, model.M[i])
                output = activation
            
            #TEST_BINARIZE mode: binarize the output using the chosen record_mode function
            elif (model.state == DomainAdaptationMode.TEST_BINARIZE):
                output = model.record_mode(model, output)
            
            #In TEST mode, do nothing
                
            return output
        return activation_shaping_hook

class ResNet18ForDomainAdaptationExtension(nn.Module):
    def __init__(self, record_mode, K, layers_to_apply):
        super(ResNet18ForDomainAdaptationExtension, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)
        self.record_mode = record_mode
        self.K = K
        #This net has a state: RECORD if it is recording the activations, APPLY if it is applying the activations
        self.state = DomainAdaptationMode.RECORD

        #Register hooks
        #Make a list of all network convolutional layers to easily iterate over them
        all_layers = []
        output_size = 56
        for layer_group in [self.resnet.layer1, self.resnet.layer2, self.resnet.layer3, self.resnet.layer4]:
            for layer in layer_group:
                all_layers.append(layer.conv1)
                all_layers.append(layer.conv2)
        #Hook into the convolutional layers
        n_applied = len(layers_to_apply)
        for index,l in enumerate(layers_to_apply):
            all_layers[l].register_forward_hook(get_domain_adaptation_hook(self, index))
        
        #Use an array to keep last generated M
        self.M = [None for _ in range(n_applied)]

    def forward(self, x):
        return self.resnet(x)

    def reset_M(self):
        for i in range(len(self.M)):
            self.M[i] = None
    
    def extend_M_for_bigger_batch(self, multiply_factor):
        #For each m in M, and for each matrix in m, extend it by multiply_factor over dimension 0 (batch size)
        for i in range(len(self.M)):
            self.M[i] = torch.cat([self.M[i]]*multiply_factor, dim=0)
        
