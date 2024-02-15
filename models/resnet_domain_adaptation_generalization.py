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

"""EXTENSION = 0: Like extension point 1.
   EXTENSION = 1: Binarize stored Ms, do not binarize activations when applying M (M works as a filter)
   EXTENSION = 2: Do not binarize either Ms nor activations when applying M"""
def get_domain_adaptation_hook(model, i):
        def activation_shaping_hook(module, input, output):
            #RECORD mode: record the activations using the chosen record_mode function
            if (model.state == DomainAdaptationMode.RECORD):
                if (CONFIG.EXTENSION < 2):
                    M = model.record_mode(model, output).detach()
                else:
                    M = output.clone().detach()

                #To handle the different Mi, we can simply multiply the new M with the product of all the previous
                #or just set model.M[i] = M if it's the first
                if model.M[i] is None:
                    model.M[i] = M
                else:
                    model.M[i] = torch.mul(model.M[i], M)
                
                #Statistics
                if (CONFIG.print_stats == 1):
                    percentage_m = torch.sum(M) / M.numel()
                    percentage_tot_m = torch.sum(model.M[i]) / model.M[i].numel()
                    if (not f"{i}" in model.statistics):
                        model.statistics[f"{i}"] = {"single_l": []}
                    model.statistics[f"{i}"]["single_l"].append(percentage_m)
                    model.statistics[f"{i}"]["total_m"] = percentage_tot_m

            #Apply mode: apply M as in the previous versions   
            elif (model.state == DomainAdaptationMode.APPLY):
                #In extension 0, output is binarized. In extension 1 and 2, output left as is
                if (CONFIG.EXTENSION == 0):
                    activation = torch.where(output <= 0, torch.zeros_like(output), torch.ones_like(output))
                elif (CONFIG.EXTENSION == 1 or CONFIG.EXTENSION == 2):
                    activation = output
                #model.M[i] is already the product of all the Mi
                
                activation = torch.mul(activation, model.M[i])
                #IF index is in layers_only_for_stats, do not apply the transformation. It is used only to produce statistics
                if (i not in CONFIG.layers_only_for_stats):
                    output = activation

                if (CONFIG.print_stats == 1):
                    #Statistics
                    model.statistics[f"{i}"]["total_out"] = torch.sum(activation) / activation.numel()

            
            #TEST_BINARIZE mode: binarize the output using the chosen record_mode function
            elif (model.state == DomainAdaptationMode.TEST_BINARIZE):
                #In extension 0, output is binarized-filtered as matrix M. In extension 1, output is multiplied by binarized version of itself
                if (CONFIG.EXTENSION == 0):
                    output = model.record_mode(model, output)# torch.where(output <= 0, torch.zeros_like(output), torch.ones_like(output))
                #With EXTENSION=2, TEST_BINARIZE does not actually binarize activations but still keeps 
                #only the values >= K; kind of like computing M over itself and then applying it
                elif (CONFIG.EXTENSION == 1):
                    output = torch.mul(output, model.record_mode(model, output))
                #With EXTENSION=2, TEST_BINARIZE is equal to TEST
            
            #In TEST mode, do nothing
                
            return output
        return activation_shaping_hook

def M_random_generator(shape, alpha):
    M = torch.ones(shape, device='cuda:0')
    M = torch.where(torch.rand(shape, device='cuda:0') <= alpha, M, torch.zeros(shape, device='cuda:0')).to(CONFIG.device)
    return M

def simple_activation_shaping_hook(module, input, output):
    activation = torch.where(output <= 0, torch.zeros_like(output), torch.ones_like(output))
    output = torch.mul(activation, M_random_generator(activation.shape, 0.9))
    return output

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
        all_layers = [] #[self.resnet.layer1, self.resnet.layer2, self.resnet.layer3, self.resnet.layer4] 
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

        #Use a dictionary to store arbitrary statistics
        self.statistics = {}
        self.layers_to_apply = layers_to_apply

        #Extra experiment, using random activation map on second (third) layer
        if (CONFIG.random_M_on_second == 1):
            all_layers[1].register_forward_hook(simple_activation_shaping_hook)

    def format_statistics(self):
        s = ""
        for key in self.statistics:
            stats = self.statistics[key]
            single_activations = stats["single_l"]
            average_single_activation = sum(single_activations) / len(single_activations)
            s = s + f"Layer {key}: average M {average_single_activation} - product of M {stats['total_m']} - output activations {stats['total_out']}\n"
        self.statistics = {}
        return s
    
    def forward(self, x):
        return self.resnet(x)

    def reset_M(self):
        for i in range(len(self.M)):
            self.M[i] = None
    
    def extend_M_for_bigger_batch(self, multiply_factor):
        #For each m in M, and for each matrix in m, extend it by multiply_factor over dimension 0 (batch size)
        for i in range(len(self.M)):
            self.M[i] = torch.cat([self.M[i]]*multiply_factor, dim=0)

    #DEBUG ONLY method - clone entire model
    def clone(self):
        new_model = ResNet18Extended(self.record_mode, self.K, self.layers_to_apply)
        new_model.load_state_dict(self.state_dict())
        new_model.resnet.cuda()
        return new_model
        
