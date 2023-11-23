import pickle
import wandb
import os

from src.utils import set_seed
from src.options import args_parser
from src.trainer import Trainer


if __name__ == "__main__":
    args = args_parser()
    set_seed(args.seed)
        
    if args.data.upper() == 'CIFAR10':
        num_classes = 10
    elif args.data.upper() == 'CIFAR100':
        num_classes = 100
   
    if args.wandb == True:
        wandb.init(project = args.wandb_project_name, 
                   config = args, 
                   name = 'run')
        
    trainer = Trainer(config)
    trainer.fit()
    
    with open('./result,pkl', 'wb') as f:
        pickle.dump(trainer.result, f)