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

## pip install transformers==4.31 or 4.32!!!
```bash
pip install huggingface-hub==0.19.4
pip install transformers==4.31
```

remove (comment) the bitsandbytes code in /mnt/nfs/envs_hj/envs/FASTCHAT/lib/python3.9/site-packages/peft/tuners/lora/model.py manually!!!

# Eval
```bash
conda activate XXX

CUDA_VISIBLE_DEVICES=7 python3 -m fastchat.serve.hj_clear_cli --model-path /mnt/nfs/zhangqi/zhangqi_nfs/DLM-project/public_models/modelWeights/vicuna-7b-v1.5

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

CUDA_VISIBLE_DEVICES=7 python3 -m fastchat.serve.hj_clear_cli --model-path /mnt/nfs/houjing/repo/FastChat/data/interim/baseHj13B-seed1-bs8-date0305/checkpoint-2130

CUDA_VISIBLE_DEVICES=6 python3 -m fastchat.serve.hj_clear_cli --model-path /mnt/nfs/houjing/repo/FastChat/data/interim/vicuna-13b-lora-0229-epoch10-lr2em4-37232+zqData-03042038/checkpoint-652
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

# Judege truth
```bash
CUDA_VISIBLE_DEVICES=6 python fastchat/serve/hj_infer_truth.py
```

# PPO
```bash
CUDA_VISIBLE_DEVICES=3 python fastchat/train/hj_ppo.py
```

```bash
CUDA_VISIBLE_DEVICES=3 nohup python fastchat/train/hj_ppo.py \
    > ./data/interim/nohup_hj_ppo_20240221.log 2>&1 &
```

```bash
torchrun --nproc_per_node=2 --master_port=20001 fastchat/train/hj_ppo.py
```

# DPO
```bash
CUDA_VISIBLE_DEVICES=7 python fastchat/train/hj_dpo.py
```

# Train
See mindmaster

# Path Without FT
```bash
CUDA_VISIBLE_DEVICES=6 python /mnt/nfs/houjing/repo/FastChat/fastchat/train/hj_eval_checkpoint_json.py \
    --model_name_or_path /mnt/nfs/zhangqi/zhangqi_nfs/DLM-project/public_models/modelWeights/vicuna-13b-v1.5 \
    --eval_data_path /mnt/nfs/houjing/repo/FastChat/data/raw/data_date121314_dataNum911.json \
    --output_dir /mnt/nfs/houjing/repo/FastChat/data/interim/vicuna-13b-lora-eval \
    --per_device_eval_batch_size 2
```

# Path With FT
```bash
CUDA_VISIBLE_DEVICES=6 python /mnt/nfs/houjing/repo/FastChat/fastchat/train/hj_eval_checkpoint_json.py \
    --model_path /mnt/nfs/houjing/repo/FastChat/data/interim/vicuna-7b-lora-CQ-v0-1217-epoch100/checkpoint-2500 \
    --eval_data_path /mnt/nfs/houjing/repo/FastChat/data/raw/data_date121314_dataNum911.json \
    --output_dir /mnt/nfs/houjing/repo/FastChat/data/interim/vicuna-13b-lora-eval \
    --per_device_eval_batch_size 2
```
