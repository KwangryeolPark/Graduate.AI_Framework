#!/bin/bash

LAUNCH_ARGS=""
PYTHON_ARGS=""

#   Single GPU launch
for mixed_precision in fp16 bf16 fp8
do
    LAUNCH_ARGS="$LAUNCH_ARGS --mixed_precision $mixed_precision"
    for use_deepspeed in false true #   use_deepspeed is store_ture.?
    do
        if [[ $use_deepspeed == "true" ]]; then
            LAUNCH_ARGS="$LAUNCH_ARGS --use_deepspeed"
        fi
        #   Single GPU Python args
        for dataset in cifar10 cifar100
        do
            PYTHON_ARGS="$PYTHON_ARGS --dataset $dataset"
            for model_name in resnet efficientnet
            do
                PYTHON_ARGS="$PYTHON_ARGS --model_name $model_name"
                for pin_memory in false 
                do
                    PYTHON_ARGS="$PYTHON_ARGS --pin_memory $pin_memory"
                    for num_workers in 1 2 4 8
                    do
                        PYTHON_ARGS="$PYTHON_ARGS --num_workers $num_workers"
                        for batch_size in 64 128 256 512
                        do
                            PYTHON_ARGS="$PYTHON_ARGS --batch_size $batch_size"
                            for optimizer in sgd adam sgdm
                            do
                                PYTHON_ARGS="$PYTHON_ARGS --optimizer $optimizer"
                                for gradient_accumulation_steps in 1 2 4
                                do
                                    PYTHON_ARGS="$PYTHON_ARGS --gradient_accumulation_steps $gradient_accumulation_steps"
                                    (
                                        LAUNCH_ARGS="$LAUNCH_ARGS --gpu_ids 0"
                                        PYTHON_ARGS="$PYTHON_ARGS --seed 0"
                                        accelerate launch \
                                            $LAUNCH_ARGS \
                                            main.py \
                                            $PYTHON_ARGS \
                                            $LAUNCH_ARGS
                                    )&
                                    (
                                        LAUNCH_ARGS="$LAUNCH_ARGS --gpu_ids 1"
                                        PYTHON_ARGS="$PYTHON_ARGS --seed 1"
                                        accelerate launch \
                                            $LAUNCH_ARGS \
                                            main.py \
                                            $PYTHON_ARGS \
                                            $LAUNCH_ARGS
                                    )&
                                    (
                                        LAUNCH_ARGS="$LAUNCH_ARGS --gpu_ids 2"
                                        PYTHON_ARGS="$PYTHON_ARGS --seed 2"
                                        accelerate launch \
                                            $LAUNCH_ARGS \
                                            main.py \
                                            $PYTHON_ARGS \
                                            $LAUNCH_ARGS
                                    )&
                                    (
                                        LAUNCH_ARGS="$LAUNCH_ARGS --gpu_ids 3"
                                        PYTHON_ARGS="$PYTHON_ARGS --seed 3"
                                        accelerate launch \
                                            $LAUNCH_ARGS \
                                            main.py \
                                            $PYTHON_ARGS \
                                            $LAUNCH_ARGS
                                    )&
                                    wait;
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
