export CUDA_VISIBLE_DEVICES=0,1

MODEL_NAME=meta-llama/Llama-3.1-8B
#meta-llama/Llama-3.2-1B
#meta-llama/Llama-3.1-8B
MODEL_SIZE=8B
NUM_GPUS=2
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=32
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
MAIN_PROCESS_PORT=0 # default 29400, automated selection using 0

echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

# Lora training
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --main_process_port $MAIN_PROCESS_PORT \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    --deepspeed_multinode_launcher standard open_instruct/finetune_peft.py  \
    --model_name_or_path $MODEL_NAME \
    --use_flash_attn \
    --use_prefix_tuning \
    --num_virtual_tokens 30 \
    --tokenizer_name $MODEL_NAME \
    --use_slow_tokenizer \
    --train_file data/tulu-3-sft-mixture-json/train.json \
    --max_seq_length 2048 \
    --preprocessing_num_workers 16 \
    --checkpointing_steps epoch \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 1e-4 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 1 \
    --output_dir output/llama_3_${MODEL_SIZE}_prefix_tuning_seqlen_2048/ \
    --with_tracking \
    --report_to wandb \
    --logging_steps 1 

