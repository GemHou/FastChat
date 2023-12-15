# Install
```bash
conda install python<3.11
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install wavedrom --use-pep517
pip3 install -e ".[model_worker,webui]"
pip install packaging
pip3 install -e ".[train]"
strings /usr/lib64/libstdc++.so.6 | grep GLIBCXX
strings /usr/lib64/libstdc++.so.6 | grep CXXABI
pip install bitsandbytes-cuda116
pip install scipy
python -m bitsandbytes
```

remove (comment) the bitsandbytes code in /mnt/nfs/envs_hj/envs/FASTCHAT/lib/python3.9/site-packages/peft/tuners/lora/model.py manually!!!

# Eval
```bash
conda activate XXX

CUDA_VISIBLE_DEVICES=1 python3 -m fastchat.serve.cli --model-path /mnt/nfs/zhangqi/zhangqi_nfs/DLM-project/public_models/modelWeights/vicuna-7b-v1.5

CUDA_VISIBLE_DEVICES=1 python3 -m fastchat.serve.cli --model-path /mnt/nfs/zhangqi/zhangqi_nfs/DLM-project/public_models/modelWeights/vicuna-13b-v1.5
```

# Train
```bash
torchrun --nproc_per_node=4 --master_port=20001 fastchat/train/train_mem.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --data_path data/dummy_conversation.json \
    --bf16 True \
    --output_dir output_vicuna \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1200 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True
```

```bash
deepspeed --master_port=29668 --include localhost:3 fastchat/train/train_lora.py \
    --model_name_or_path /mnt/nfs/zhangqi/zhangqi_nfs/DLM-project/public_models/modelWeights/vicuna-7b-v1.5 \
    --lora_r 32 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --data_path /mnt/nfs/zhangqi/zhangqi_nfs/DLM-project/slm/data/raw/CQ-project-data/train_vicuna.json \
    --eval_data_path /mnt/nfs/zhangqi/zhangqi_nfs/DLM-project/slm/data/raw/CQ-project-data/train_vicuna.json \
    --output_dir /mnt/nfs/zhangqi/zhangqi_nfs/Evol_LLaMA-project/model_garden/vicuna-7b-lora-CQ-v0-1212 \
    --num_train_epochs 5 \
    --fp16 True \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "steps" \
    --eval_steps 100  \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --q_lora True \
    --deepspeed /mnt/nfs/zhangqi/zhangqi_nfs/Evol_LLaMA-project/FastChat-D/scripts/zq/zero2.json \
    --gradient_checkpointing True \
    --flash_attn True \
    --run_name vicuna-7b-lora-CQ-v0-1212
```

```bash
python fastchat/train/train_lora.py \
    --model_name_or_path ~/nfs/zhangqi/zhangqi_nfs/DLM-project/public_models/modelWeights/vicuna-7b-v1.5 \
    --lora_r 32 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --data_path ~/nfs/zhangqi/zhangqi_nfs/DLM-project/slm/data/raw/CQ-project-data/train_vicuna.json \
    --eval_data_path ~/nfs/zhangqi/zhangqi_nfs/DLM-project/slm/data/raw/CQ-project-data/train_vicuna.json \
    --output_dir ~/nfs/zhangqi/zhangqi_nfs/Evol_LLaMA-project/model_garden/vicuna-7b-lora-CQ-v0-1212 \
    --num_train_epochs 5 \
    --fp16 True \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "steps" \
    --eval_steps 100  \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --q_lora True \
    --gradient_checkpointing True \
    --flash_attn True \
    --run_name vicuna-7b-lora-CQ-v0-1212
```

```bash
python fastchat/train/train_lora.py \
    --model_name_or_path /mnt/nfs/zhangqi/zhangqi_nfs/DLM-project/public_models/modelWeights/vicuna-7b-v1.5 \
    --data_path /mnt/nfs/zhangqi/zhangqi_nfs/DLM-project/slm/data/raw/CQ-project-data/train_vicuna.json \
    --eval_data_path /mnt/nfs/zhangqi/zhangqi_nfs/DLM-project/slm/data/raw/CQ-project-data/train_vicuna.json \
    --output_dir ./data/interim/vicuna-7b-lora-CQ-v0-1215 \
    --run_name vicuna-7b-lora-CQ-v0-1215 \
    --fp16 True \
    --tf32 True \
    --q_lora True \
    --gradient_checkpointing True \
    --flash_attn True \
    --lr_scheduler_type "cosine" \
    --logging_strategy "steps" \
    --evaluation_strategy "steps" \
    --eval_steps 100  \
    --save_strategy "steps" \
    --save_steps 100 \
    --num_train_epochs 5 \
    --lora_r 32 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --model_max_length 2048
```