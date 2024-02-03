import torch
import os
import torchvision.transforms as T
from dataset.utils import BaseDataset #, DomainAdaptationDataset, DomainGeneralizationDataset
from dataset.utils import SeededDataLoader

from globals import CONFIG

def get_transform(size, mean, std, preprocess):
    transform = []
    if preprocess:
        transform.append(T.Resize(256))
        transform.append(T.RandomResizedCrop(size=size, scale=(0.7, 1.0)))
        transform.append(T.RandomHorizontalFlip())
    else:
        transform.append(T.Resize(size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean, std))
    return T.Compose(transform)


def load_data():
    CONFIG.num_classes = 7
    CONFIG.data_input_size = (3, 224, 224)

    # Create transforms
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225) # ImageNet Pretrain statistics
    train_transform = get_transform(size=224, mean=mean, std=std, preprocess=True)
    test_transform = get_transform(size=224, mean=mean, std=std, preprocess=False)

    # Load examples & create Dataset
    if CONFIG.experiment in ['baseline', 'activation_shaping_experiments', 'domain_adaptation']:
        source_examples = []
        test_dataset = {}

        # Load source
        with open(os.path.join(CONFIG.dataset_args['root'], f"{CONFIG.dataset_args['source_domain']}.txt"), 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            path, label = line[0].split('/')[1:], int(line[1])
            source_examples.append((os.path.join(CONFIG.dataset_args['root'], *path), label))

        # Load target
        for target_domain in CONFIG.dataset_args['target_domain']:
            target_examples = []
            with open(os.path.join(CONFIG.dataset_args['root'], f"{target_domain}.txt"), 'r') as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip().split()
                path, label = line[0].split('/')[1:], int(line[1])
                target_examples.append((os.path.join(CONFIG.dataset_args['root'], *path), label))
            test_dataset[target_domain] = BaseDataset(target_examples, transform=test_transform)

        train_dataset = BaseDataset(source_examples, transform=train_transform)

    # Dataloaders
    train_loader = SeededDataLoader(
        train_dataset,
        batch_size=CONFIG.batch_size,
        shuffle=True,
        num_workers=CONFIG.num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    test_loaders = {}
    
    for target_domain in CONFIG.dataset_args['target_domain']:
        test_loaders[target_domain] = SeededDataLoader(
            test_dataset[target_domain],
            batch_size=CONFIG.batch_size,
            shuffle=False,
            num_workers=CONFIG.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    return {'train': train_loader, 'test': test_loaders}