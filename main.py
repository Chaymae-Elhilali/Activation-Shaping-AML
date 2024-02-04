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
from models.resnet import BaseResNet18, hook_activation_shaping
from models.resnet_domain_adaptation import *

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


DEBUG_TRAIN_EACH_TIME=True

def train(model, data):

    # Create optimizers & schedulers
    optimizer = torch.optim.SGD(model.parameters(), weight_decay=0.0005, momentum=0.9, nesterov=True, lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(CONFIG.epochs * 0.8), gamma=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    # Load checkpoint (if it exists)
    cur_epoch = 0
    if not DEBUG_TRAIN_EACH_TIME and os.path.exists(os.path.join('record', CONFIG.experiment_name, 'last.pth')):
        print("Reloading from saved state")
        checkpoint = torch.load(os.path.join('record', CONFIG.experiment_name, 'last.pth'))
        cur_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        model.load_state_dict(checkpoint['model'])
    
    # Optimization loop
    for epoch in range(cur_epoch, CONFIG.epochs):
        model.train()
        
        for batch_idx, batch in enumerate(tqdm(data['train'])):
            
            # Compute loss
            with torch.autocast(device_type=CONFIG.device, dtype=torch.float16, enabled=True):

                if CONFIG.experiment in ['baseline', 'activation_shaping_experiments'] :
                    #x: feature; y: label
                    x, y = batch
                    x, y = x.to(CONFIG.device), y.to(CONFIG.device) #move to gpu
                    loss = F.cross_entropy(model(x), y) #cross entropy
                elif CONFIG.experiment in ['domain_adaptation']:
                    x, y, target_x = batch
                    x, y, target_x = x.to(CONFIG.device), y.to(CONFIG.device), target_x.to(CONFIG.device) #move to gpu
                    #If domain adaptation, must run model twice
                    model.state = DomainAdaptationMode.RECORD
                    _ = model(target_x)
                    model.state = DomainAdaptationMode.APPLY
                    loss = F.cross_entropy(model(x), y)
                    
            # Optimization step
            scaler.scale(loss / CONFIG.grad_accum_steps).backward()
            if ((batch_idx + 1) % CONFIG.grad_accum_steps == 0) or (batch_idx + 1 == len(data['train'])):
                scaler.step(optimizer)
                optimizer.zero_grad(set_to_none=True)
                scaler.update()

        scheduler.step()
        # Test current epoch
        logging.info(f'[TEST @ Epoch={epoch}]')
        #If experiment is domain adaptation must set model in test mode:
        if (CONFIG.experiment in ['domain_adaptation']):
            model.state = DomainAdaptationMode.TEST
            evaluate(model, data['test'], extra_str="TEST SIMPLE ")
            model.state = DomainAdaptationMode.TEST_BINARIZE
            evaluate(model, data['test'], extra_str="TEST SIMPLE BINARIZED ")
        else:
            evaluate(model, data['test'])

        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'model': model.state_dict()
        }
        torch.save(checkpoint, os.path.join('record', CONFIG.experiment_name, 'last.pth'))

def get_M_random_generator_function(alpha):
  #Input: tensor size; output: random 
  def M_random_generator(size):
    M = torch.ones(size)
    M = torch.where(torch.rand(size) <= alpha, M, torch.zeros(size))
    M = M.to(CONFIG.device, non_blocking=False)
    return M
  return M_random_generator


def main():
    # Load dataset
    data = PACS.load_data()

    # Load model
    if CONFIG.experiment in ['baseline']:
        model = BaseResNet18()
    elif CONFIG.experiment in ['activation_shaping_experiments']:
        model = BaseResNet18()
        #Apply hooks
        hook_activation_shaping(model, get_M_random_generator_function(CONFIG.ALPHA), CONFIG.APPLY_EVERY_N, CONFIG.SKIP_FIRST_N)
    elif CONFIG.experiment in ['domain_adaptation']:
        #In previous experiments we could have multiple target domains and test on all of them, now only one at a time
        assert len(CONFIG.dataset_args["target_domain"])==1
        assert not CONFIG.test_only
        #IF CONFIG.RECORD_MODE = "threshold", use threshold, if = "topk" use topK 
        if CONFIG.RECORD_MODE == "threshold":
            record_mode = RecordModeRule.THRESHOLD
        elif CONFIG.RECORD_MODE == "topk":
            record_mode = RecordModeRule.TOP_K
        else:
            exit("RECORD_MODE must be either topk or threshold")

        model = ResNet18ForDomainAdaptation(record_mode, CONFIG.K, CONFIG.SKIP_FIRST_N, CONFIG.APPLY_EVERY_N)


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
    if ("ALPHA" in CONFIG.dataset_args):
      CONFIG.ALPHA = CONFIG.dataset_args["ALPHA"]
    if ("SKIP_FIRST_N" in CONFIG.dataset_args):
      CONFIG.SKIP_FIRST_N = CONFIG.dataset_args["SKIP_FIRST_N"]
    if ("APPLY_EVERY_N" in CONFIG.dataset_args):
      CONFIG.APPLY_EVERY_N = CONFIG.dataset_args["APPLY_EVERY_N"]
    if ("RECORD_MODE" in CONFIG.dataset_args):
      CONFIG.RECORD_MODE = CONFIG.dataset_args["RECORD_MODE"]
    if ("K" in CONFIG.dataset_args):
      CONFIG.K = CONFIG.dataset_args["K"]
    #target domain: if set, transform it in a list separated by comma
    if ("target_domain" in CONFIG.dataset_args and CONFIG.dataset_args["target_domain"] != ""):
      CONFIG.dataset_args["target_domain"] = CONFIG.dataset_args["target_domain"].replace(" ", "").split(',')
      
    # Setup output directory
    CONFIG.save_dir = os.path.join('record', CONFIG.experiment_name)
    os.makedirs(CONFIG.save_dir, exist_ok=True)
    # Setup logging
    LOG_FILENAME = "log.txt"
    if (CONFIG.experiment == "activation_shaping_experiments"):
      LOG_FILENAME = f"SKIP_{CONFIG.SKIP_FIRST_N}_ALPHA_{CONFIG.ALPHA}-APPLY_EVERY_N_{CONFIG.APPLY_EVERY_N}-log.txt"
    elif (CONFIG.experiment == "domain_adaptation"):
      LOG_FILENAME = f"SKIP_{CONFIG.SKIP_FIRST_N}_K_{CONFIG.K}-APPLY_EVERY_N_{CONFIG.APPLY_EVERY_N}-RECORD_MODE{CONFIG.RECORD_MODE}-log.txt"
    logging.basicConfig(
        filename=os.path.join(CONFIG.save_dir, LOG_FILENAME), 
        format='%(message)s', 
        level=logging.INFO, 
        filemode='a'
    )

    # Set experiment's device & deterministic behavior
    if CONFIG.cpu:
        CONFIG.device = torch.device('cpu')

    torch.manual_seed(CONFIG.seed)
    random.seed(CONFIG.seed)
    np.random.seed(CONFIG.seed)
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(mode=True, warn_only=True)

    main()
