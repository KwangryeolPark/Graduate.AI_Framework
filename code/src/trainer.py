import torch
import torch.nn as nn
import numpy as np
import torchvision
import wandb

from src.dataset import prepare_dataset

from src.models.resnet import ResNet
from src.models.efficientnet import EfficientNet

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
            self.model = ResNet50(num_classes=num_classes)
        elif self.model_name.upper() == 'EFFICIENTNET':
            self.model = EfficientNetB0(num_classes=num_classes)
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        
        # prepare dataset
        self.trainset, self.testset = prepare_dataset(configure)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.bs, shuffle=True)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.bs, shuffle=False)
    
    def train(self):
        self.model.train()
        for data, labels in self.trainloader:
            data, labels = data.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
    
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
        for epoch in range(self.epochs):
            self.train()
            loss, acc = self.test()
            self.result['loss'].append(loss)
            self.result['accuracy'].append(acc)
            
            if self.wandb:
                wandb.log({f'Acc': acc, 'epoch': epoch})
                wandb.log({f'Loss': loss, 'epoch': epoch})
    