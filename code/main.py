import pickle
import wandb
import os

from src.utils import set_seed
from src.options import args_parser
from src.trainer import Trainer


if __name__ == "__main__":
    args = args_parser()

    set_seed(args.seed)
    if args.device != 'cpu':
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        device = 'cuda'
    else:
        device = 'cpu'     
        
    if args.data.upper() == 'CIFAR10':
        num_classes = 10
    elif args.data.upper() == 'CIFAR100':
        num_classes = 100
   
    config = {
       'data': args.data,
       'num_classes': num_classes, 
       'model': args.model,
       'epochs': args.epochs,
       'bs': args.bs,
       'lr': args.lr,
       'optim': args.optim,
       'multi_gpu': args.multi_gpu,
       'device': device,
       'wandb': args.wandb,
       
       # accelerator options
       'grad_chk_pointing': args.grad_chk_pointing,
       'grad_accum_step': args.grad_accum_step,
       'mix_prec_fp16': args.mix_prec_fp16,
       'torch_compile': args.torch_compile,
    }
    
    if args.wandb == True:
        wandb.init(project = args.wandb_project_name, 
                   config = config, 
                   name = 'run')
        
    trainer = Trainer(config)
    trainer.fit()
    
    with open('./result,pkl', 'wb') as f:
        pickle.dump(trainer.result, f)