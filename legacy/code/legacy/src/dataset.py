import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, TensorDataset, DataLoader


def prepare_dataset(configure):
    dataset_name = configure['data']
    dataset_name = dataset_name.upper()
    
    if hasattr(torchvision.datasets, dataset_name):
        if dataset_name == 'CIFAR10':
            transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        transforms.ToTensor(),
                    ])
        
        elif dataset_name == 'CIFAR100':
            transform = transforms.Compose([
                        transforms.ToTensor(),
                    ])
        
        trainset = torchvision.datasets.__dict__[dataset_name](root='/workspace/shared/data', train=True,
                                            download=True, transform=transform)
        testset = torchvision.datasets.__dict__[dataset_name](root='/workspace/shared/data', train=False,
                                            download=True, transform=transform)

    return trainset, testset