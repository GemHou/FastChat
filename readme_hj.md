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

CUDA_VISIBLE_DEVICES=1 python3 -m fastchat.serve.hj_clear_cli --model-path /mnt/nfs/zhangqi/zhangqi_nfs/DLM-project/public_models/modelWeights/vicuna-7b-v1.5

CUDA_VISIBLE_DEVICES=6 python3 -m fastchat.serve.hj_clear_cli --model-path /mnt/nfs/zhangqi/zhangqi_nfs/DLM-project/public_models/modelWeights/vicuna-13b-v1.5

CUDA_VISIBLE_DEVICES=7 python3 -m fastchat.serve.hj_clear_cli --model-path ./data/interim/vicuna-7b-lora-CQ-v0-1215/checkpoint-100

CUDA_VISIBLE_DEVICES=7 python3 -m fastchat.serve.hj_clear_cli --model-path ./data/interim/vicuna-7b-lora-CQ-v0-1217-epoch50/checkpoint-1200

CUDA_VISIBLE_DEVICES=6 python3 -m fastchat.serve.hj_clear_cli --model-path ./data/interim/vicuna-7b-lora-CQ-v0-1219-epoch10-lr2em4-vdata1543/checkpoint-400

CUDA_VISIBLE_DEVICES=3 python3 -m fastchat.serve.hj_clear_cli --model-path ./data/interim/vicuna-13b-lora-CQ-v0-1219-epoch20-lr2em4-vdata8196-evalGPT/checkpoint-1700

CUDA_VISIBLE_DEVICES=6 python3 -m fastchat.serve.hj_clear_cli --model-path ./data/interim/vicuna-13b-lora-CQ-v0-0102-epoch5-lr2em4-vdata19298-evalGPT/checkpoint-1400

CUDA_VISIBLE_DEVICES=6 python3 -m fastchat.serve.hj_clear_cli --model-path ./data/interim/vicuna-13b-lora-CQ-v0-0102-epoch5-lr2em4-vDataKeyword6797-evalGPT/checkpoint-637

CUDA_VISIBLE_DEVICES=5 python3 -m fastchat.serve.hj_clear_cli --model-path ./data/interim/vicuna-7b-lora-CQ-v0-1217-epoch100/checkpoint-2500

CUDA_VISIBLE_DEVICES=1 python3 -m fastchat.serve.hj_clear_cli --model-path ./data/interim/vicuna-13b-lora-CQ-v0-0102-epoch5-lr2em4-vDataKeyword20438_4622-evalGPT/checkpoint-3133

CUDA_VISIBLE_DEVICES=2 python3 -m fastchat.serve.hj_clear_cli --model-path ./data/interim/vicuna-13b-lora-CQ-v0-0102-epoch20-lr2em4-vDataKeyword4550_4733-evalGPT/checkpoint-2031
```

# Collect data
```bash
CUDA_VISIBLE_DEVICES=2 nohup python fastchat/serve/hj_infer.py \
    > ./data/interim/nohup_hj_infer.log 2>&1 &
```

```bash
CUDA_VISIBLE_DEVICES=2 nohup python fastchat/serve/hj_infer_keyword.py \
    > ./data/interim/nohup_hj_infer_keyword.log 2>&1 &
```

# Train

zhangqi python hjPara splitData backgroundData:
```bash
CUDA_VISIBLE_DEVICES=2 python fastchat/train/train_lora.py \
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
CUDA_VISIBLE_DEVICES=3 python fastchat/train/train_lora.py \
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