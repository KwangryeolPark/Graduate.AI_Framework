# python main.py --data cifar100 --model resnet --bs 128 --lr 0.1 --epochs 5 --device 0 --grad_accum_step 2 
# # --grad_chk_pointing True --mix_prec_fp16 True 

# # command for multi gpu 
# # ref: https://huggingface.co/docs/accelerate/usage_guides/explore
# accelerate launch --multi_gpu --num_gpus 2 --gpu_ids 0,1 main.py --data cifar100 --model resnet --bs 128 --lr 0.1 --epochs 5 

#/bin/bash

RANDOM_PORT=$(shuf -i 1024-49151 -n 1)


gradient_accumulation_steps=1
dataset=cifar10
deep_speed=True
multi_gpu=True
mix_prec=True
compile=True
pin_memory=True
num_workers=4

GPU_OPTIONS="
    --multi_gpu \
    --num_processes  2 \
    --gpu_ids   0,1
"

MIX_PREC_OPTION="
    --mixed_precision fp16\
"

DEEP_SPEED_OPTIONS="
    --use_deepspeed \
    --num_processes 2 \
    --zero_stage 3\
    --gradient_clipping 1\
    --zero3_init_flag True\
    --zero3_save_16bit_model True\
    --offload_optimizer_device cpu \
    --offload_param_device cpu \
"

MAIN_OPTIONS="
    --dataset $cifar10 \
    --model_name resnet \
    --lr 0.01 \
    --batch_size 64 \
    --epochs 20 \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --pin_memory $pin_memory\
    --num_workers $num_workers
"

if
accelerate launch \
    $DEEP_SPEED_OPTIONS\
    main.py \
    $MAIN_OPTIONS \
    $@

# $MIX_PREC_OPTION\