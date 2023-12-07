from utils.trainer import Trainer
from utils.args import get_args

def main():
    args = get_args()
    trainer = Trainer(args)
    
    trainer.fit()
    
if __name__ == '__main__':
    main()