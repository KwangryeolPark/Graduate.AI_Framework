import pickle
import wandb

from src.utils import set_seed
from src.options import args_parser
from src.trainer import Trainer


if __name == "__main__":
    set_seed(args.seed)
    
    args = args_parser()
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
   }
    if args.wandb == True:
        wandb.init(project = args.wandb_project_name, 
                   config = config, 
                   name = 'run')
        
    trainer = Trainer(config)
    trainer.fit()
    
    with open('./result,pkl', 'wb') as f:
        pickle.dump(trainer.result, f)