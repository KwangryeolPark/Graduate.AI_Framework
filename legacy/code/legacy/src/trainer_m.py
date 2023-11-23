import torch
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import torch.nn as nn
import torchvision
import numpy as np
import wandb

from torchvision.models import resnet50, efficientnet_b0
from tqdm import tqdm
from accelerate import Accelerator

from src.dataset import prepare_dataset
from src.utils import set_seed

class Trainer(object):
    def __init__(self, config):
        self.config = config
        accelerate = Accelerator(
            gradient_accumulation_steps=config.grad_accum_step,
            log_with='wandb',
            project_dir='./wandb'
        )
        
        if accelerator.is_local_main_process:
            accelerator.init_trackers(
                project_name=config.wandb_project,
                init_kwargs={
                    "wandb": {
                        "name": config.wandb_name
                    }
                }
            )
        if accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()
        set_seed(config.seed)
        accelerator.wait_for_everyone()