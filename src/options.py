import argparse
from datetime import datetime

def args_parser():
    parser = argparse.ArgumentParser()
    now = datetime.now()
    timestamp = now.strftime('%Y_%m_%d__%H_%M_%S')

    #   Others
    parser.add_argument('--dataset', type=str, required=True, help='dataset', choices=['cifar10', 'cifar100'])
    parser.add_argument('--model_name', type=str, required=True, help='model name', choices=['resnet', 'efficientnet'])
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size, N x 16')
    parser.add_argument('--torch_compile', type=bool, default=False, help='torch compile')
    parser.add_argument('--epochs', type=int, default=20, help='epochs')
    parser.add_argument('--timestamp', type=str, default=str(timestamp), help='current time')    

    #   Accelerate    
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient accumulation steps')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    #   Wandb
    parser.add_argument('--wandb_project_name', type=str, default='AI Framework Project', help='wandb project name')
    parser.add_argument('--wandb_name', type=str, default=str(timestamp), help='wandb name')
    return parser.parse_args()