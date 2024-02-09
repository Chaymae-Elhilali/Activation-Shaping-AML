import torch
import torch.nn as nn
from torch.types import Device
from torchvision.models import resnet18, ResNet18_Weights
from globals import CONFIG

#Define an enum for network state: RECORD and APPLY, + TEST (in which the hook does nothing)
class DomainAdaptationMode:
    RECORD = 0
    APPLY = 1
    TEST = 2
    TEST_BINARIZE = 3

#Define a class for the RECORD mode's rule for generating M: by threshold or by top-k
class RecordModeRule:
    def THRESHOLD(model, output):
        return torch.where(output <= model.K, torch.zeros_like(output), torch.ones_like(output))
    
    def TOP_K(model, output):
        #If the record mode is by top-k, we record the top-k
        #Matrix Output has 4 dimensions: batch_size, filters, height, width
        #Can apply best-k independently for each filter or for whole output
        #For now, we apply best-k for each filter
        #Reshape the output to 2D: batch_size*filters, height*width
        K = int(model.K * (output.shape[2] * output.shape[3]))
        reshaped_output = output.view(output.shape[0]*output.shape[1], -1)
        #Get the top-k indices
        _, top_k_indices = torch.topk(reshaped_output, K, dim=1)
        #Create a 2D tensor with zeros
        binarized_output = torch.zeros_like(reshaped_output)
        #Set the top-k indices to 1
        binarized_output.scatter_(1, top_k_indices, 1)
        #Reshape the output back to 4D
        return binarized_output.view(output.shape)

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
        output_size = 56
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
