import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.datasets import CIFAR10

import numpy as np
import random
from PIL import Image

from globals import CONFIG

class BaseDataset(Dataset):
    def __init__(self, examples, transform):
        self.examples = examples
        self.T = transform
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        x, y = self.examples[index]
        x = Image.open(x).convert('RGB')
        x = self.T(x).to(CONFIG.dtype)
        y = torch.tensor(y).long()
        return x, y

class DomainAdaptationDataset(Dataset):
    def __init__(self, source_examples, target_examples, transform):
        self.source_examples = source_examples
        self.target_examples = target_examples
        self.T = transform
    
    def __len__(self):
        return len(self.source_examples)
    
    def __getitem__(self, index):
        x, y = self.source_examples[index]
        x = Image.open(x).convert('RGB')
        x = self.T(x).to(CONFIG.dtype)
        y = torch.tensor(y).long()

        targ_x, _ = self.target_examples[random.randint(0, len(self.target_examples) - 1)]
        #targ_x, _ = self.target_examples[index % len(self.target_examples)]
        targ_x = Image.open(targ_x).convert('RGB')
        targ_x = self.T(targ_x).to(CONFIG.dtype)
        return x, y, targ_x


class DomainGeneralizationDataset(Dataset):
    def __init__(self, examples, transform):
        self.examples = DomainGeneralizationDataset.build_examples(examples)
        self.T = transform

    def build_examples(examples):
        labels = list(examples[0].keys())
        new_examples = []
        for label in labels:
            #List of examples for each domain
            examples_by_domain = [x[label] for x in examples]
            for e in examples_by_domain:
                random.shuffle(e)
            max_len = max([len(x) for x in examples_by_domain])
            for i in range(max_len):
                l = []
                for j in range(len(examples_by_domain)):
                    l.append(examples_by_domain[j][i % len(examples_by_domain[j])][0])
                new_examples.append(l + [label])
        random.shuffle(new_examples)
        #Each element in new_examples is a list, with 3 examples w.t. same label and one label
        #(ex_1, ex_2, ex_3, label)
        return new_examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        examples = self.examples[index]
        x = [self.T(Image.open(x).convert('RGB')) for x in examples[:-1]]
        y = examples[-1]
        y = torch.tensor(y).long()
        return x + [y]


class SeededDataLoader(DataLoader):
    def __init__(self, dataset: Dataset, batch_size=1, shuffle=None, 
                 sampler=None, 
                 batch_sampler=None, 
                 num_workers=0, collate_fn=None, 
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None, multiprocessing_context=None, 
                 generator=None, *, prefetch_factor=None, persistent_workers=False, 
                 pin_memory_device=""):
        
        if not CONFIG.use_nondeterministic:
            def seed_worker(worker_id):
                worker_seed = torch.initial_seed() % 2**32
                np.random.seed(worker_seed)
                random.seed(worker_seed)

            generator = torch.Generator()
            generator.manual_seed(CONFIG.seed)

            worker_init_fn = seed_worker
        
        super().__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn, 
                         pin_memory, drop_last, timeout, worker_init_fn, multiprocessing_context, generator, 
                         prefetch_factor=prefetch_factor, persistent_workers=persistent_workers, 
                         pin_memory_device=pin_memory_device)

