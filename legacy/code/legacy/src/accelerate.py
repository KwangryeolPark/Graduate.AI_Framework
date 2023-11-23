from accelerate import Accelerator
from accelerate.utils import set_seed

def get_accelerator(args):
    args = args
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with='wandb',
        project_dir='./wandb'
    )
    if accelerator.is_local_main_process:
        accelerator.init_trackers(
            project_name=args.wandb_project_name,
            init_kwargs={
                'wandb': {
                    'name': args.wandb_name,
                }
            },
            config=args
        )
    
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    
    set_seed(args.seed)
    
    accelerator.wait_for_everyone()
    
    return accelerate