import torch
import torch.nn as nn
import torchvision
import numpy as np
import wandb
from tqdm import tqdm 

from src.dataset import prepare_dataset
from src.models import resnet
from src.models import efficientnet

from accelerate import Accelerator

class Trainer(object):
    
    def __init__(self, configure):
        self.config = configure
        
        self.model_name = configure['model']
        self.num_classes = configure['num_classes']
        self.epochs = configure['epochs']
        self.bs = configure['bs']
        self.lr = configure['lr']
        self.optim = configure['optim']
        self.multi_gpu = configure['multi_gpu']
        self.dataset_name = configure['data']
        self.device = configure['device']
        self.wandb = configure['wandb']
        
        if self.model_name.upper() == 'RESNET':
            self.model = resnet.ResNet50(num_classes=self.num_classes)
        elif self.model_name.upper() == 'EFFICIENTNET':
            self.model = efficientnet.EfficientNetB0(num_classes=self.num_classes)
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
        
        # prepare dataset
        self.trainset, self.testset = prepare_dataset(configure)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.bs, shuffle=True)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.bs, shuffle=False)
        
        # accelerator options
        self.grad_accum_step = configure['grad_accum']
        self.accelerator = Accelerator(gradient_accumulation_steps=self.grad_accum_step)
        
        self.model, self.optimizer, self.trainloader, self.scheduler =\
            self.accelerator.prepare(self.model, self.optimizer, self.trainloader, self.scheduler)
    
    def train(self):
        self.model.train()
        for data, labels in self.trainloader:
            data, labels = data.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
    def train_with_accelerator(self):
        self.model.train()
        for idx, (data, labels) in enumerate(self.trainloader):
            with self.accelerator.accumulate(self.model):
                self.optimizer.zero_grad()
                
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                self.accelerator.backward(loss)
            
                self.optimizer.step()
                self.scheduler.step()
                
    
    def test(self):
        self.model.eval()
        loss = 0; correct = 0
        with torch.no_grad():
            for data, labels in self.testloader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                
                loss += self.criterion(outputs, labels).item()
                predicted = outputs.argmax(dim=1, keepdim=True)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()
        loss = loss / len(self.testloader)
        accuracy = correct / len(self.testloader.dataset)
        return loss, accuracy
    
    def fit(self):
        self.result = {'loss': [], 'accuracy': []}
        for epoch in tqdm(range(self.epochs)):
            # self.train()
            self.train_with_accelerator()
            loss, acc = self.test()
            self.result['loss'].append(loss)
            self.result['accuracy'].append(acc)
            
            if self.wandb:
                wandb.log({f'Acc': acc, 'epoch': epoch})
                wandb.log({f'Loss': loss, 'epoch': epoch})
    