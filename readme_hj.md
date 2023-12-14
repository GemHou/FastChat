# Install
```bash
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip3 install -e ".[model_worker,webui]"
```

# Eval
```bash
conda activate XXX

CUDA_VISIBLE_DEVICES=1 python3 -m fastchat.serve.cli --model-path /mnt/nfs/zhangqi/zhangqi_nfs/DLM-project/public_models/modelWeights/vicuna-7b-v1.5

CUDA_VISIBLE_DEVICES=1 python3 -m fastchat.serve.cli --model-path /mnt/nfs/zhangqi/zhangqi_nfs/DLM-project/public_models/modelWeights/vicuna-13b-v1.5
```

# Train
```bash

```