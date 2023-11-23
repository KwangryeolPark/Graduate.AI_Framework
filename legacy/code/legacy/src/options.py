import argparse

# def str_to_bool(param):
#     if isinstance(param, bool):
#         return param
#     if param.lower() in ('true', '1'): 
#         return True
#     elif param.lower() in ('false', '0'):
#         return False
#     else:
#         raise argparse.argparse.ArgumentTypeError('boolean value expected')


def args_parser():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--data', help="name of dataset", required=True, choices=['cifar10', 'cifar100'])
    
    parser.add_argument('--model', help="name of model", required=True, choices=['resnet', 'efficientnet']) # Resnet50 or EfficientNet
    parser.add_argument('--epochs', type=int, default=10, help="epochs")
    parser.add_argument('--bs', type=int, default=64, help="batch size") # per device batch size
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    # parser.add_argument('--optim', help="optimizer", required=True)   #   trainer.py에 구현 안 돼 있음.
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    
    # acclerator options
    # parser.add_argument('--per_dev_bs', type=int, help='per device batch size') # -> bs
    parser.add_argument('--grad_accum_step', type=int, default=None, help='Gradient accumulation') 
    # parser.add_argument('--grad_chk_pointing', type=bool, default=False, help='whether to use gradient checkpointing') 
    # parser.add_argument('--mix_prec_fp16', type=bool, default=False, help='whether to use mixed precision training') It can be configured in `accelerate config` commend.
    parser.add_argument('--torch_compile', type=bool, default=False, help='whether to use torch compile') 


    #   wandb
    parser.add_argument('--wandb', type=bool, default=True, help='wandb logger')
    parser.add_argument('--wandb_project', type=str, default="AI-Framework", help='wandb project name')
    parser.add_argument('--wandb_name', type=str, default='run', help='wandb name')

    args = parser.parse_args()
    return args
