import torch
import torchvision
import torchvision.transforms as transforms

def prepare_dataset(config)->tuple:
    dataset = config.dataset
    if dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    else:
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
    trainset = torchvision.datasets.__dict__[dataset.upper()](
        root='./dataset',
        train=True,
        download=True,
        transform=transform
    )
    testset = torchvision.datasets.__dict__[dataset.upper()](
        root='./dataset',
        train=False,
        download=True,
        transform=transform
    )
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size, shuffle=False)
    
    return train_loader, test_loader