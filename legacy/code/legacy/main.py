import pickle
import wandb
import os

from src.options import args_parser
from src.trainer import Trainer

if __name__ == "__main__":
    args = args_parser()
    set_seed(args.seed)
    
    args['num_classes'] = 10 if args.data.upper() == 'CIFAR10' else 100
    
    trainer = Trainer(args)
    trainer.fit()
    
    with open(f'./{args.save_dir}.pkl', 'wb') as f:
        pickle.dump(trainer.result, f)