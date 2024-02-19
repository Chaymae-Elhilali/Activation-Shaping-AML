import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy
from tqdm import tqdm

import os
import logging
import warnings
import random
import numpy as np
from parse_args import parse_arguments

from dataset import PACS
from models.resnet import BaseResNet18
from models.resnet_domain_adaptation_generalization import *

from globals import CONFIG

@torch.no_grad()
def evaluate(model, data, extra_str=""):
    model.eval()
    for target_domain in data.keys():
        acc_meter = Accuracy(task='multiclass', num_classes=CONFIG.num_classes)
        acc_meter = acc_meter.to(CONFIG.device)
        loss = [0.0, 0]
        for x, y in tqdm(data[target_domain]):
            with torch.autocast(device_type=CONFIG.device, dtype=torch.float16, enabled=True):
                x, y = x.to(CONFIG.device), y.to(CONFIG.device)
                logits = model(x)
                acc_meter.update(logits, y)
                loss[0] += F.cross_entropy(logits, y).item()
                loss[1] += x.size(0)
        
        accuracy = acc_meter.compute()
        loss = loss[0] / loss[1]
        logging.info(f'{extra_str}Evaluate: {target_domain}: Accuracy: {100 * accuracy:.2f} - Loss: {loss}')


def train(model, data):

    # Create optimizers & schedulers
    optimizer = torch.optim.SGD(model.parameters(), weight_decay=CONFIG.weight_decay, momentum=0.9, nesterov=True, lr=CONFIG.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(CONFIG.epochs * 0.8), gamma=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    cur_epoch = 0

    
    # Optimization loop
    for epoch in range(cur_epoch, CONFIG.epochs):
        if ((CONFIG.apply_progressively == 1 or CONFIG.apply_progressively_perm==1) and CONFIG.experiment in ['domain_generalization','domain_adaptation']):
            l = min(int(epoch / int(CONFIG.epochs / len(CONFIG.LAYERS_LIST))), len(CONFIG.LAYERS_LIST) - 1)
            model.set_current_layer_to_apply(l)

        total_loss = [0.0, 0]
        model.train()
        for batch_idx, batch in enumerate(tqdm(data['train'])):
            
            #Reset statistics
            if (CONFIG.print_stats == 1):
                model.statistics = {}

            if CONFIG.experiment in ['baseline', 'activation_shaping_experiments'] :
                with torch.autocast(device_type=CONFIG.device, dtype=torch.float16, enabled=True):
                    x, y = batch
                    x, y = x.to(CONFIG.device), y.to(CONFIG.device)
                    loss = F.cross_entropy(model(x), y)
                    total_loss[0] += loss.item()
                    total_loss[1] += x.size(0)


            elif CONFIG.experiment in ['domain_adaptation']:
                x, y, target_x = batch
                x, y, target_x = x.to(CONFIG.device), y.to(CONFIG.device), target_x.to(CONFIG.device) 
                
                #Record activation maps
                model.state = DomainAdaptationMode.RECORD
                model.reset_M()
                model.eval()
                with torch.autocast(device_type=CONFIG.device, dtype=torch.float16, enabled=True):
                    with torch.no_grad():
                        _ = model(target_x)
                model.train()

                #Run the model applying the activation maps
                model.state = DomainAdaptationMode.APPLY
                with torch.autocast(device_type=CONFIG.device, dtype=torch.float16, enabled=True):
                    loss = F.cross_entropy(model(x), y)
                total_loss[0] += loss.item()
                total_loss[1] += x.size(0)

                if (CONFIG.print_stats == 1 and (epoch % 5 == 0) and batch_idx <= 4):
                    logging.info(f"Statistics at epoch {epoch}:\n" + model.format_statistics())
            
            elif CONFIG.experiment in ['domain_generalization']:
                x1, x2, x3, y = batch
                x1, x2, x3, y = x1.to(CONFIG.device), x2.to(CONFIG.device), x3.to(CONFIG.device), y.to(CONFIG.device) #move to gpu
                x = torch.cat((x1, x2, x3), 0)
                y = torch.cat((y, y, y), 0)
    
                #Record the activation matrices for x1, x2, x3
                model.state = DomainAdaptationMode.RECORD
                model.reset_M()
                with torch.autocast(device_type=CONFIG.device, dtype=torch.float16, enabled=True):
                    with torch.no_grad():
                        model.eval()
                        _ = model(x1)
                        _ = model(x2)
                        _ = model(x3)
                
                #Run the model applying the activation maps
                model.train()
                with torch.autocast(device_type=CONFIG.device, dtype=torch.float16, enabled=True):
                    #The matrices M must be extended to match the new batch size
                    #On dim 0, M had activations for multiple elements of the activations on the original batch
                    model.extend_M_for_bigger_batch(3)
                    #parameter "generalization_test_only" is used for the ablation test
                    if (CONFIG.generalization_test_only != 1):
                        model.state = DomainAdaptationMode.APPLY
                    else:
                        model.state = DomainAdaptationMode.TEST
                        
                    loss = F.cross_entropy(model(x), y)
                    total_loss[0] += loss.item()
                    total_loss[1] += x.size(0)
            
                    if (CONFIG.print_stats == 1 and (epoch % 5 == 0) and batch_idx <= 4):
                        logging.info("Statistics at epoch {epoch}:\n" + model.format_statistics())

            
            # Optimization step
            scaler.scale(loss / CONFIG.grad_accum_steps).backward()
            if ((batch_idx + 1) % CONFIG.grad_accum_steps == 0) or (batch_idx + 1 == len(data['train'])):
                scaler.step(optimizer)
                optimizer.zero_grad(set_to_none=True)
                scaler.update()


        scheduler.step()
        # Test current epoch
        logging.info(f'[TEST @ Epoch={epoch}]')
        logging.info(f'Train: Loss: {total_loss[0] / total_loss[1]}')
        
        if (CONFIG.experiment in ['domain_adaptation']):
            model.state = DomainAdaptationMode.TEST
            evaluate(model, data['test'], extra_str="TEST SIMPLE ")
            model.state = DomainAdaptationMode.TEST_BINARIZE
            evaluate(model, data['test'], extra_str="TEST SIMPLE BINARIZED ")

        elif (CONFIG.experiment in ['domain_generalization']):
            model.state = DomainAdaptationMode.TEST
            evaluate(model, data['test'], extra_str="TEST SIMPLE ")
            model.state = DomainAdaptationMode.TEST_BINARIZE
            evaluate(model, data['test'], extra_str="TEST SIMPLE BINARIZED ")

        elif (CONFIG.experiment in ['activation_shaping_experiments']):
            evaluate(model, data['test'], extra_str="TEST WITH BINARIZATION")
            model.disable_hooks()
            evaluate(model, data['test'], extra_str="TEST WITHOUT BINARIZATION ")
            model.enable_hooks()

        else:
            evaluate(model, data['test'])



def main():
    # Load dataset
    data = PACS.load_data()

    # Load model
    if CONFIG.experiment in ['baseline']:
        model = BaseResNet18()
    
    elif CONFIG.experiment in ['activation_shaping_experiments']:
        model = BaseResNet18()
        #Apply hooks
        model.hook_activation_shaping(CONFIG.ALPHA, CONFIG.LAYERS_LIST)

    elif CONFIG.experiment in ['domain_adaptation']:
        #In previous experiments we could have multiple target domains and test on all of them, now only one at a time
        assert len(CONFIG.dataset_args["target_domain"])==1
        assert not CONFIG.test_only

        #Specify the function to use for recording the activations
        if CONFIG.RECORD_MODE == "threshold":
            record_mode = RecordModeRule.THRESHOLD
        elif CONFIG.RECORD_MODE == "topk":
            record_mode = RecordModeRule.TOP_K
        else:
            exit("RECORD_MODE must be either topk or threshold")

        model = ResNet18Extended(record_mode, CONFIG.K, CONFIG.LAYERS_LIST)
    
    elif CONFIG.experiment in ['domain_generalization']:
        #In previous experiments we could have multiple target domains and test on all of them, now only one at a time
        assert len(CONFIG.dataset_args["target_domain"])==1
        assert len(CONFIG.dataset_args["source_domain"])==3
        assert not CONFIG.test_only

        #Specify the function to use for recording the activations
        if CONFIG.RECORD_MODE == "threshold":
            record_mode = RecordModeRule.THRESHOLD
        elif CONFIG.RECORD_MODE == "topk":
            record_mode = RecordModeRule.TOP_K
        else:
            exit("RECORD_MODE must be either topk or threshold")
        model = ResNet18Extended(record_mode, CONFIG.K, CONFIG.LAYERS_LIST)
    if (CONFIG.print_stats == 1):
        CONFIG.epochs = 16

    model.to(CONFIG.device)

    if not CONFIG.test_only:
        train(model, data)
    else:
        evaluate(model, data['test'])

if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=UserWarning)
    # Parse arguments
    args = parse_arguments()
    CONFIG.update(vars(args))

    # Set experiment's device & deterministic behavior
    if CONFIG.cpu:
        CONFIG.device = torch.device('cpu')

    torch.manual_seed(CONFIG.seed)
    random.seed(CONFIG.seed)
    np.random.seed(CONFIG.seed)
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(mode=True, warn_only=True)

    #Parse experiment-specific arguments
    #-------------------------------------------------------------------
    if ("ALPHA" in CONFIG.dataset_args):
      CONFIG.ALPHA = CONFIG.dataset_args["ALPHA"]
    if ("RECORD_MODE" in CONFIG.dataset_args):
        CONFIG.RECORD_MODE = CONFIG.dataset_args["RECORD_MODE"]
    if ("K" in CONFIG.dataset_args):
      CONFIG.K = CONFIG.dataset_args["K"]
    if CONFIG.experiment in ['activation_shaping_experiments', 'domain_adaptation', 'domain_generalization']:
        LAYERS_LIST = CONFIG.dataset_args["LAYERS_LIST"]
        CONFIG.LAYERS_LIST = [int(x) for x in CONFIG.dataset_args["LAYERS_LIST"].split(',')]
    
    #Target domain: if set, transform it in a list separated by comma
    if ("target_domain" in CONFIG.dataset_args and CONFIG.dataset_args["target_domain"] != ""):
      CONFIG.dataset_args["target_domain"] = CONFIG.dataset_args["target_domain"].replace(" ", "").split(',')
    #Specific for domain generalization, set source_domain as a list with all the domains except the target
    if (CONFIG.experiment in ["domain_generalization"]):
        CONFIG.dataset_args["source_domain"] = ["art_painting","cartoon","photo","sketch"]
        CONFIG.dataset_args["source_domain"].remove(CONFIG.dataset_args["target_domain"][0])
        CONFIG.batch_size = int(CONFIG.batch_size/3)
    #Extensions for domain generalization (not included in the paper)
    if 'EXTENSION' in CONFIG.dataset_args:
        CONFIG.EXTENSION = CONFIG.dataset_args['EXTENSION']
    else:
        CONFIG.EXTENSION = 0  

    #Allow to attach the module to some layers only to produce statistics without applying the M
    if (len(CONFIG.layers_only_for_stats) > 0):
        CONFIG.layers_only_for_stats = [int(x) for x in CONFIG.layers_only_for_stats.split(',')]
    else:
        CONFIG.layers_only_for_stats = []

    assert CONFIG.layer_type in ['conv', 'bn']
        
    # Setup output directory
    CONFIG.save_dir = os.path.join('record', CONFIG.experiment_name, CONFIG.extra_str)
    os.makedirs(CONFIG.save_dir, exist_ok=True)
    # Setup logging
    LOG_FILENAME = "log.txt"
    if (CONFIG.experiment == "activation_shaping_experiments"):
      LOG_FILENAME = f"L_{LAYERS_LIST}__ALPHA__{CONFIG.ALPHA}-log.txt"
    elif (CONFIG.experiment in ["domain_adaptation"]):
      LOG_FILENAME = f"L_{LAYERS_LIST}__K__{CONFIG.K}__RECORD_MODE__{CONFIG.RECORD_MODE}-log.txt"
    elif (CONFIG.experiment in ["domain_generalization"]):
      LOG_FILENAME = f"L_{LAYERS_LIST}__K__{CONFIG.K}__RECORD_MODE{CONFIG.RECORD_MODE}-log.txt"
    if (CONFIG.random_M_on_second):
        LOG_FILENAME = "random_M_" + LOG_FILENAME
    if (CONFIG.apply_progressively == 1):
        LOG_FILENAME = "progressive_" + LOG_FILENAME
    if (CONFIG.apply_progressively_perm == 1):
        LOG_FILENAME = "progressive_perm_" + LOG_FILENAME
    logging.basicConfig(
        filename=os.path.join(CONFIG.save_dir, LOG_FILENAME), 
        format='%(message)s', 
        level=logging.INFO, 
        filemode='a'
    )


    main()
