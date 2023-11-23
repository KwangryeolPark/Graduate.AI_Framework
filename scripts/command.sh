# python main.py --data cifar100 --model resnet --bs 128 --lr 0.1 --epochs 5 --device 0 --grad_accum_step 2 
# # --grad_chk_pointing True --mix_prec_fp16 True 

# # command for multi gpu 
# # ref: https://huggingface.co/docs/accelerate/usage_guides/explore
# accelerate launch --multi_gpu --num_gpus 2 --gpu_ids 0,1 main.py --data cifar100 --model resnet --bs 128 --lr 0.1 --epochs 5 

#/bin/bash

GPU_OPTIONS="
    --multi_gpu \
    --num_gpus  2 \
    --gpu_ids   0,1
"

MAIN_OPTIONS="
    --dataset   cifar10 \
    --model_name    resnet  \
    --lr    0.01    \
    --batch_size    64  \
    --torch_compile False    \
    --epochs    5   \

    --gradient_accumulation_steps 1 \
"

accelerate launch \
    $GPU_OPTIONS \
    main.py \
    $MAIN_OPTIONS