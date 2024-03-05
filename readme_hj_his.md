
official readme: 
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

zhangqi: 
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

zhangqi python
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
zhangqi python hjPara:
```bash
CUDA_VISIBLE_DEVICES=6 python fastchat/train/train_lora.py \
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

zhangqi python hjPara splitData:
```bash
CUDA_VISIBLE_DEVICES=6 python fastchat/train/train_lora.py \
    --model_name_or_path /mnt/nfs/zhangqi/zhangqi_nfs/DLM-project/public_models/modelWeights/vicuna-7b-v1.5 \
    --data_path /mnt/nfs/zhangqi/zhangqi_nfs/DLM-project/slm/data/raw/CQ-project-data/train_vicuna.json \
    --dev_ratio 0.1 \
    --output_dir ./data/interim/vicuna-7b-lora-CQ-v0-121518 \
    --run_name vicuna-7b-lora-CQ-v0-121518 \
    --fp16 True \
    --tf32 True \
    --q_lora True \
    --gradient_checkpointing True \
    --flash_attn True \
    --lr_scheduler_type "cosine" \
    --logging_strategy "steps" \
    --evaluation_strategy "steps" \
    --eval_steps 20  \
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


zhangqi python hjPara splitData backgroundData:
```bash
CUDA_VISIBLE_DEVICES=6 python fastchat/train/train_lora.py \
    --model_name_or_path /mnt/nfs/zhangqi/zhangqi_nfs/DLM-project/public_models/modelWeights/vicuna-7b-v1.5 \
    --data_path ./data/raw/data_date121314_dataNum911.json \
    --dev_ratio 0.1 \
    --output_dir ./data/interim/vicuna-7b-lora-CQ-v0-1217-epoch100 \
    --run_name vicuna-7b-lora-CQ-v0-1217-epoch100 \
    --fp16 True \
    --tf32 True \
    --q_lora True \
    --gradient_checkpointing True \
    --flash_attn True \
    --lr_scheduler_type "cosine" \
    --logging_strategy "steps" \
    --evaluation_strategy "steps" \
    --eval_steps 20  \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 10 \
    --num_train_epochs 100 \
    --lora_r 32 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --model_max_length 2048
```

zhangqi python hjPara splitData backgroundData largeLearningRate:
```bash
CUDA_VISIBLE_DEVICES=6 python fastchat/train/train_lora.py \
    --model_name_or_path /mnt/nfs/zhangqi/zhangqi_nfs/DLM-project/public_models/modelWeights/vicuna-7b-v1.5 \
    --data_path ./data/raw/data_date121314_dataNum911.json \
    --dev_ratio 0.1 \
    --output_dir ./data/interim/vicuna-7b-lora-CQ-v0-1219-epoch10-lr2em4 \
    --run_name vicuna-7b-lora-CQ-v0-1219-epoch10-lr2em4 \
    --fp16 True \
    --tf32 True \
    --q_lora True \
    --gradient_checkpointing True \
    --flash_attn True \
    --lr_scheduler_type "cosine" \
    --logging_strategy "steps" \
    --evaluation_strategy "steps" \
    --eval_steps 20  \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 10 \
    --num_train_epochs 10 \
    --lora_r 32 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --model_max_length 2048
```

zhangqi python hjPara splitData backgroundData largeLearningRate newData:
```bash
CUDA_VISIBLE_DEVICES=6 python fastchat/train/train_lora.py \
    --model_name_or_path /mnt/nfs/zhangqi/zhangqi_nfs/DLM-project/public_models/modelWeights/vicuna-7b-v1.5 \
    --data_path ./data/interim/data_vicuna_date122116_dataNum3969.json \
    --dev_ratio 0.1 \
    --output_dir ./data/interim/vicuna-7b-lora-CQ-v0-1219-epoch10-lr2em4-vdata3969 \
    --run_name vicuna-7b-lora-CQ-v0-1219-epoch10-lr2em4-vdata3969 \
    --fp16 True \
    --tf32 True \
    --q_lora True \
    --gradient_checkpointing True \
    --flash_attn True \
    --lr_scheduler_type "cosine" \
    --logging_strategy "steps" \
    --evaluation_strategy "steps" \
    --eval_steps 20  \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 10 \
    --num_train_epochs 10 \
    --lora_r 32 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --model_max_length 2048
```

zhangqi python hjPara splitData backgroundData largeLearningRate newData 13B:
```bash
CUDA_VISIBLE_DEVICES=0 nohup python fastchat/train/train_lora.py \
    --model_name_or_path /mnt/nfs/zhangqi/zhangqi_nfs/DLM-project/public_models/modelWeights/vicuna-13b-v1.5 \
    --data_path ./data/interim/data_vicuna/data_vicuna_date122816_dataNum1164.json \
    --dev_ratio 0.1 \
    --output_dir ./data/interim/vicuna-13b-lora-CQ-v0-1219-epoch10-lr2em4-vdata1164 \
    --run_name vicuna-13b-lora-CQ-v0-1219-epoch10-lr2em4-vdata1164 \
    --fp16 True \
    --tf32 True \
    --q_lora True \
    --gradient_checkpointing True \
    --flash_attn True \
    --lr_scheduler_type "cosine" \
    --logging_strategy "steps" \
    --evaluation_strategy "steps" \
    --eval_steps 20  \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 10 \
    --num_train_epochs 10 \
    --lora_r 32 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --model_max_length 2048 \
    > ./data/interim/nohup_train_lora_epoch10_vdata1164.log 2>&1 &
```

zhangqi python hjPara splitData backgroundData largeLearningRate newData 13B evalGPT:
```bash
CUDA_VISIBLE_DEVICES=7 nohup python fastchat/train/train_lora.py \
    --model_name_or_path /mnt/nfs/zhangqi/zhangqi_nfs/DLM-project/public_models/modelWeights/vicuna-13b-v1.5 \
    --data_path ./data/interim/data_vicuna/data_vicuna_date122909_dataNum19298.json \
    --eval_data_path ./data/raw/data_date121314_dataNum911.json \
    --output_dir ./data/interim/vicuna-13b-lora-CQ-v0-0102-epoch5-lr2em4-vdata19298-evalGPT \
    --run_name vicuna-13b-lora-CQ-v0-0102-epoch5-lr2em4-vdata19298-evalGPT \
    --fp16 True \
    --tf32 True \
    --q_lora True \
    --gradient_checkpointing True \
    --flash_attn True \
    --lr_scheduler_type "cosine" \
    --logging_strategy "steps" \
    --evaluation_strategy "steps" \
    --eval_steps 200  \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 10 \
    --num_train_epochs 5 \
    --lora_r 32 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --model_max_length 2048 \
    > ./data/interim/output-epoch5-lr2em4-vdata19298-evalGPT.log 2>&1 &
```

zhangqi python hjPara largeLearningRate mixData 13B evalGPT:
```bash
CUDA_VISIBLE_DEVICES=3 nohup python fastchat/train/train_lora.py \
    --model_name_or_path /mnt/nfs/zhangqi/zhangqi_nfs/DLM-project/public_models/modelWeights/vicuna-13b-v1.5 \
    --data_path ./data/interim/data_vicuna_keyword/data_vicuna_keyword__date011521_dataNum20438.json \
    --eval_data_path ./data/raw/data_date121314_dataNum911.json \
    --output_dir ./data/interim/vicuna-13b-lora-CQ-v0-0102-epoch5-lr2em4-vDataKeyword20438-evalGPT \
    --run_name vicuna-13b-lora-CQ-v0-0102-epoch5-lr2em4-vDataKeyword20438-evalGPT \
    --fp16 True \
    --tf32 True \
    --q_lora True \
    --gradient_checkpointing True \
    --flash_attn True \
    --lr_scheduler_type "cosine" \
    --logging_strategy "steps" \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --num_train_epochs 5 \
    --lora_r 32 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --model_max_length 2048 \
    > ./data/interim/output-epoch5-lr2em4-vDataKeyword20438-evalGPT.log 2>&1 &
```

zhangqi python hjPara largeLearningRate mixData 13B evalGPT:
```bash
CUDA_VISIBLE_DEVICES=6 nohup python fastchat/train/train_lora.py \
    --model_name_or_path /mnt/nfs/zhangqi/zhangqi_nfs/DLM-project/public_models/modelWeights/vicuna-13b-v1.5 \
    --data_path ./data/interim/data_vicuna_keyword/data_vicuna_keyword__date011521_dataNum20438_4622.json \
    --eval_data_path ./data/raw/data_date121314_dataNum911.json \
    --output_dir ./data/interim/vicuna-13b-lora-CQ-v0-0102-epoch5-lr2em4-vDataKeyword20438_4622-evalGPT \
    --output_dir ./data/interim/vicuna-13b-lora-CQ-v0-0102-epoch5-lr2em4-vDataKeyword20438_4622-evalGPT \
    --run_name vicuna-13b-lora-CQ-v0-0102-epoch5-lr2em4-vDataKeyword20438_4622-evalGPT \
    --fp16 True \
    --tf32 True \
    --q_lora True \
    --gradient_checkpointing True \
    --flash_attn True \
    --lr_scheduler_type "cosine" \
    --logging_strategy "steps" \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --num_train_epochs 5 \
    --lora_r 32 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --model_max_length 2048 \
    > ./data/interim/output-epoch5-lr2em4-vDataKeyword20438_4622-evalGPT.log 2>&1 &
```

zhangqi python hjPara largeLearningRate mixData 13B evalGPT:
```bash
CUDA_VISIBLE_DEVICES=3 nohup python fastchat/train/train_lora.py \
    --model_name_or_path /mnt/nfs/zhangqi/zhangqi_nfs/DLM-project/public_models/modelWeights/vicuna-13b-v1.5 \
    --data_path ./data/interim/data_vicuna_keyword/data_vicuna_keyword__date011521_dataNum4550_4733.json \
    --eval_data_path ./data/raw/data_date121314_dataNum911.json \
    --output_dir ./data/interim/vicuna-13b-lora-CQ-v0-0102-epoch20-lr2em4-vDataKeyword4550_4733-evalGPT \
    --output_dir ./data/interim/vicuna-13b-lora-CQ-v0-0102-epoch20-lr2em4-vDataKeyword4550_4733-evalGPT \
    --run_name vicuna-13b-lora-CQ-v0-0102-epoch20-lr2em4-vDataKeyword4550_4733-evalGPT \
    --fp16 True \
    --tf32 True \
    --q_lora True \
    --gradient_checkpointing True \
    --flash_attn True \
    --lr_scheduler_type "cosine" \
    --logging_strategy "steps" \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --num_train_epochs 20 \
    --lora_r 32 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --model_max_length 2048 \
    > ./data/interim/output-epoch20-lr2em4-vDataKeyword4550_4733-evalGPT.log 2>&1 &
```

```bash
CUDA_VISIBLE_DEVICES=2 nohup python fastchat/train/train_lora.py \
    --model_name_or_path /mnt/nfs/zhangqi/zhangqi_nfs/DLM-project/public_models/modelWeights/vicuna-13b-v1.5 \
    --data_path ./data/interim/data_vicuna_keyword/data_vicuna_keyword__date011521_dataNum37232.json \
    --dev_ratio 0.1 \
    --output_dir ./data/interim/vicuna-13b-lora-CQ-v0-1219-epoch10-lr2em4-vdata37232 \
    --run_name vicuna-13b-lora-CQ-v0-1219-epoch10-lr2em4-vdata37232 \
    --fp16 True \
    --tf32 True \
    --q_lora True \
    --gradient_checkpointing True \
    --flash_attn True \
    --lr_scheduler_type "cosine" \
    --logging_strategy "steps" \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --num_train_epochs 10 \
    --lora_r 32 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --model_max_length 2048 \
    > ./data/interim/nohup_train_lora_epoch10_vdata37232.log 2>&1 &
```

```bash
CUDA_VISIBLE_DEVICES=6 python fastchat/train/train_lora.py \
    --model_name_or_path /mnt/nfs/zhangqi/zhangqi_nfs/DLM-project/public_models/modelWeights/vicuna-13b-v1.5 \
    --data_path ./data/interim/data_vicuna_keyword/data_vicuna_keyword__date011521_dataNum37232.json \
    --dev_ratio 0.1 \
    --output_dir ./data/interim/vicuna-13b-lora-CQ-v0-1219-epoch10-lr2em4-vdata37232-2 \
    --run_name vicuna-13b-lora-CQ-v0-1219-epoch10-lr2em4-vdata37232-2 \
    --fp16 True \
    --tf32 True \
    --q_lora True \
    --gradient_checkpointing True \
    --flash_attn True \
    --lr_scheduler_type "cosine" \
    --logging_strategy "steps" \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --num_train_epochs 10 \
    --lora_r 32 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --model_max_length 2048
```

```bash
CUDA_VISIBLE_DEVICES=6 python fastchat/train/train_lora.py \
    --model_name_or_path /mnt/nfs/zhangqi/zhangqi_nfs/DLM-project/public_models/modelWeights/vicuna-7b-v1.5 \
    --data_path /mnt/nfs/houjing/repo/FastChat/data/raw/train_vicuna6-s-feedback.json \
    --dev_ratio 0.1 \
    --output_dir ./data/interim/vicuna-7b-lora-0229-epoch10-lr2em4-zqData \
    --run_name vicuna-7b-lora-0229-epoch10-lr2em4-zqData \
    --fp16 True \
    --tf32 True \
    --q_lora True \
    --gradient_checkpointing True \
    --flash_attn True \
    --lr_scheduler_type "cosine" \
    --logging_strategy "steps" \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --num_train_epochs 10 \
    --lora_r 32 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --model_max_length 2048
```

```bash
CUDA_VISIBLE_DEVICES=4 python fastchat/train/train_lora.py \
    --model_name_or_path /mnt/nfs/houjing/repo/FastChat/data/interim/vicuna-13b-lora-CQ-v0-1219-epoch10-lr2em4-vdata37232/checkpoint-10470 \
    --data_path /mnt/nfs/houjing/repo/FastChat/data/raw/train_vicuna6-s-feedback.json \
    --dev_ratio 0.1 \
    --output_dir ./data/interim/vicuna-13b-lora-0229-epoch10-lr2em4-37232+zqData-03042038 \
    --run_name vicuna-13b-lora-0229-epoch10-lr2em4-37232+zqData-03042038 \
    --fp16 False \
    --tf32 True \
    --q_lora True \
    --gradient_checkpointing True \
    --flash_attn False \
    --lr_scheduler_type "cosine" \
    --logging_strategy "steps" \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --num_train_epochs 1 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --learning_rate 3e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --model_max_length 256
```

```bash
CUDA_VISIBLE_DEVICES=5 python fastchat/train/train_lora.py \
    --model_name_or_path /mnt/nfs/zhangqi/zhangqi_nfs/DLM-project/public_models/modelWeights/vicuna-13b-v1.5 \
    --data_path /mnt/nfs/houjing/repo/FastChat/data/raw/train_vicuna6-s-feedback.json \
    --dev_ratio 0.1 \
    --output_dir ./data/interim/vicuna-13b-lora-0229-epoch10-lr2em4-37232+zqData-03042050 \
    --run_name vicuna-13b-lora-0229-epoch10-lr2em4-37232+zqData-03042050 \
    --fp16 True \
    --tf32 True \
    --q_lora True \
    --gradient_checkpointing True \
    --flash_attn False \
    --lr_scheduler_type "cosine" \
    --logging_strategy "steps" \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --num_train_epochs 1 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --learning_rate 3e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --model_max_length 256
```

```bash
CUDA_VISIBLE_DEVICES=6 python fastchat/train/train_lora.py \
    --model_name_or_path /mnt/nfs/zhangqi/zhangqi_nfs/DLM-project/public_models/modelWeights/vicuna-7b-v1.5 \
    --data_path ./data/interim/data_vicuna_keyword/data_vicuna_keyword__date011521_dataNum37232.json \
    --dev_ratio 0.1 \
    --output_dir ./data/interim/vicuna-13b-lora-0229-epoch10-lr2em4-data37232 \
    --run_name vicuna-13b-lora-0229-epoch10-lr2em4-data37232 \
    --fp16 True \
    --tf32 True \
    --q_lora True \
    --gradient_checkpointing True \
    --flash_attn True \
    --lr_scheduler_type "cosine" \
    --logging_strategy "steps" \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --num_train_epochs 10 \
    --lora_r 32 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --model_max_length 2048
```

