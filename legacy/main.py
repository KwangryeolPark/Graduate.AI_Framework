import os
import pickle
import json
import random
from src.accelerator import get_accelerator
from src.trainer import Trainer
from src.options import args_parser

if __name__ == "__main__":
    args = args_parser()
    accelerator = get_accelerator(args)
    
    trainer = Trainer(args, accelerator)
    trainer.fit()
    
    rand = random.randint(0, 10000)
    
    os.makedirs('./results/log', exist_ok=True)
    with open(f'./results/log/{args.timestamp}_{rand}.pkl', 'wb') as f:
        pickle.dump(trainer.result, f)
        
    os.makedirs('./results/configs', exist_ok=True)
    with open(f'./results/configs/{args.timestamp}_{rand}.json', 'w') as f:
        json.dump(vars(args), f, indent=4)