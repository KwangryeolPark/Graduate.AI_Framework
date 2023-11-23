import argparse

def str_to_bool(param):
    if isinstance(param, bool):
        return param
    if param.lower() in ('true', '1'): 
        return True
    elif param.lower() in ('false', '0'):
        return False
    else:
        raise argparse.argparse.ArgumentTypeError('boolean value expected')


def args_parser():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--data', help="name of dataset")
    
    parser.add_argument('--model', help="name of model") # Resnet50 or EfficientNet
    parser.add_argument('--epochs', type=int, default=100, help="epochs")
    parser.add_argument('--bs', type=int, default=100, help="batch size")
    parser.add_argument('--lr', type=float, default=100, help="learning rate")
    parser.add_argument('--optim', help="optimizer")
    
    # system setting
    parser.add_argument('--multi_gpu', type=str_to_bool, default=False, help='use multi gpu or not') 
    parser.add_argument('--device', default='cuda', help='set specific GPU number of CPU')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--wandb', type=str_to_bool, default=False, help='wandb')
    parser.add_argument('--wandb_project_name', default="AI-Framework", help='wandb project name')


