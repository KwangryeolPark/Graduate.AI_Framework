import os
import pickle
from src.accelerator import get_accelerator
from src.trainer import Trainer
from src.options import args_parser

if __name__ == "__main__":
    args = args_parser()
    accelerator = get_accelerator(args)
    
    trainer = Trainer(args, accelerator)
    trainer.fit()
    
    os.makedir('./results', exist_ok=True)
    with open(f'./results/{args.timestamp}.pkl', 'wb') as f:
        pickle.dump(trainer.result, f)