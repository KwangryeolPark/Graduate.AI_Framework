import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import logging
import time
import resource

from tqdm.auto import tqdm
from src.dataset import prepare_dataset

class Trainer(object):
    def __init__(self, config, accelerator):
        self.config = config
        self.accelerator = accelerator
        num_classes = 10 if config.dataset == 'cifar10' else 100
        
        if config.model_name == 'resnet':
            from torchvision.models import resnet50
            self.model = resnet50(num_classes=num_classes)
        else:
            from torchvision.models import efficientnet_b0
            self.model = efficientnet_b0(num_classes=num_classes)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=config.lr)
        self.scheduler = scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
        
        self.train_loader, self.test_loader = prepare_dataset(config)
        
        if config.compile:
            self.model = torch.compile(self.model)
            
        self.model, self.optimizer, self.train_loader, self.test_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.test_loader, self.scheduler
        )
        self.result = {
            'train': {
                'wall_time': list(),
                'kernel_time': list(),
            },
            'test': {
                'loss': list(),
                'accuracy': list(),
                'wall_time': list(),
                'kernel_time': list(),            
            },
        }
    
    def train(self):
        self.model.train()
        progress_bar = tqdm(self.train_loader, disable=not self.accelerator.is_local_main_process, desc='Train')
        
        start_time = time.process_time()
        start_kernel_time = resource.getrusage(resource.RUSAGE_SELF)

        for data, labels in progress_bar:
            with self.accelerator.accumulate(self.model):
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.scheduler.step()
                
        end_time = time.process_time()
        end_kernel_time = resource.getrusage(resource.RUSAGE_SELF)

        wall_time = end_time - start_time
        kernel_time = end_kernel_time.ru_stime - start_kernel_time.ru_stime

        return wall_time, kernel_time
    
    def test(self):
        self.model.eval()
        loss = 0
        accuracy = 0
        with torch.no_grad():
            start_time = time.process_time()
            start_kernel_time = resource.getrusage(resource.RUSAGE_SELF)
            
            for data, labels in self.test_loader:
                outputs = self.model(data)
                
                loss += self.criterion(outputs, labels).item()
                prediction = outputs.argmax(dim=1, keepdim=True)
                accuracy += prediction.eq(labels.view_as(prediction)).sum().item()
                
            end_time = time.process_time()
            end_kernel_time = resource.getrusage(resource.RUSAGE_SELF)
        loss /= len(self.test_loader)
        accuracy /= len(self.test_loader)
        wall_time = end_time - start_time
        kernel_time = end_kernel_time.ru_stime - start_kernel_time.ru_stime
        
        return loss, accuracy, wall_time, kernel_time
    
    def fit(self):
        for epoch in tqdm(range(self.config.epochs), disable=not self.accelerator.is_local_main_process, desc='Epoch'):
            wall_time, kernel_time = self.train()
            self.result['train']['wall_time'].append(wall_time)
            self.result['train']['kernel_time'].append(kernel_time)
            
            loss, accuracy, wall_time, kernel_time = self.test()
            self.result['test']['loss'].append(loss)
            self.result['test']['accuracy'].append(accuracy)
            self.result['test']['wall_time'].append(wall_time)
            self.result['test']['kernel_time'].append(kernel_time)
            
            self.accelerator.log({
                'train_wall_time':self.result['train']['wall_time'][-1],
                'train_kernel_time':self.result['train']['kernel_time'][-1],
                
                'test_wall_time':self.result['test']['wall_time'][-1],
                'test_kernel_time':self.result['test']['kernel_time'][-1],
                'test_loss':self.result['test']['loss'][-1],
                'test_accuracy':self.result['test']['accuracy'][-1],
            })