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
        if self.config.optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=config.lr)
        elif self.config.optimizer == 'sgdm':
            self.optimizer = optim.SGD(self.model.parameters(), lr=config.lr, momentum=0.9)
        elif self.config.optimizer == 'adam':
            self.optimizer = optim.Adam(self.parameters(), lr=config.lr)
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
                'memory': list()
            },
            'test': {
                'loss': list(),
                'accuracy': list(),
                'wall_time': list(),
                'kernel_time': list(),            
                'memory': list()
            },
        }
        self.starter = torch.cuda.Event(enable_timing=True)
        self.ender = torch.cuda.Event(enable_timing=True)
    
    def train(self):
        self.model.train()
        progress_bar = tqdm(self.train_loader, disable=not self.accelerator.is_local_main_process, desc='Train')
        
        start_time = time.process_time()
        self.starter.record()
        # start_kernel_time = resource.getrusage(resource.RUSAGE_SELF)

        for data, labels in progress_bar:
            with self.accelerator.accumulate(self.model):
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.scheduler.step()
                
        self.ender.record()
        end_time = time.process_time()
        # end_kernel_time = resource.getrusage(resource.RUSAGE_SELF)

        torch.cuda.synchronize()
        wall_time = end_time - start_time
        kernel_time = self.starter.elapsed_time(self.ender)

        return wall_time, kernel_time, torch.cuda.memory_allocated()
    
    def test(self):
        self.model.eval()
        loss = 0
        accuracy = 0
        with torch.no_grad():
            start_time = time.process_time()
            self.starter.record()
            # start_kernel_time = resource.getrusage(resource.RUSAGE_SELF)
            
            for data, labels in self.test_loader:
                outputs = self.model(data)
                
                loss += self.criterion(outputs, labels).item()
                prediction = outputs.argmax(dim=1, keepdim=True)
                accuracy += prediction.eq(labels.view_as(prediction)).sum().item()
                
            self.ender.record()
            end_time = time.process_time()
            # end_kernel_time = resource.getrusage(resource.RUSAGE_SELF)

            torch.cuda.synchronize()
        loss /= len(self.test_loader)
        accuracy /= len(self.test_loader)
        wall_time = end_time - start_time
        kernel_time = self.starter.elapsed_time(self.ender)
        
        return loss, accuracy, wall_time, kernel_time, torch.cuda.memory_allocated()
    
    def fit(self):
        for epoch in tqdm(range(self.config.epochs), disable=not self.accelerator.is_local_main_process, desc='Epoch'):
            wall_time, kernel_time, memory = self.train()
            self.result['train']['wall_time'].append(wall_time)
            self.result['train']['kernel_time'].append(kernel_time)
            self.result['train']['memory'].append(memory)
            
            loss, accuracy, wall_time, kernel_time, memory = self.test()
            self.result['test']['loss'].append(loss)
            self.result['test']['accuracy'].append(accuracy)
            self.result['test']['wall_time'].append(wall_time)
            self.result['test']['kernel_time'].append(kernel_time)
            self.result['test']['memory'].append(memory)
            
            self.accelerator.log({
                'train_wall_time':self.result['train']['wall_time'][-1],
                'train_kernel_time':self.result['train']['kernel_time'][-1],
                'train_memory':self.result['train']['memory'][-1],
                
                'test_wall_time':self.result['test']['wall_time'][-1],
                'test_kernel_time':self.result['test']['kernel_time'][-1],
                'test_loss':self.result['test']['loss'][-1],
                'test_accuracy':self.result['test']['accuracy'][-1],
                'test_memory':self.result['test']['memory'][-1],
            })