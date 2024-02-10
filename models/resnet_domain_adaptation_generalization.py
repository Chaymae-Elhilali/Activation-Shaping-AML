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
                M = model.record_mode(model, output)
                #To handle the different Mi, we can simply multiply the new M with the product of all the previous
                #or just set model.M[i] = M if it's the first
                if model.M[i] is None:
                    model.M[i] = M
                else:
                    model.M[i] = torch.mul(model.M[i], M)

            #Apply mode: apply M as in the previous versions   
            elif (model.state == DomainAdaptationMode.APPLY):
                #In extension 0, output is binarized-filtered as matrix M. In extension 1, output left as is
                if (CONFIG.EXTENSION == 0):
                    activation = model.record_mode(model, output)
                elif (CONFIG.EXTENSION == 1):
                    activation = output
                #model.M[i] is already the product of all the Mi
                activation = torch.mul(activation, model.M[i])
                output = activation
            
            #TEST_BINARIZE mode: binarize the output using the chosen record_mode function
            elif (model.state == DomainAdaptationMode.TEST_BINARIZE):
                #In extension 0, output is binarized-filtered as matrix M. In extension 1, output is multiplied by binarized version of itself
                if (CONFIG.EXTENSION == 0):
                    output = model.record_mode(model, output)
                elif (CONFIG.EXTENSION == 1):
                    output = torch.mul(output, model.record_mode(model, output))
            
            #In TEST mode, do nothing
                
            return output
        return activation_shaping_hook

class ResNet18Extended(nn.Module):
    def __init__(self, record_mode, K, layers_to_apply):
        super(ResNet18Extended, self).__init__()
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
        
