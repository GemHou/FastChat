# Install
```bash
pip3 install -e ".[model_worker,webui]"
pip install prompt_toolkit
pip install rich
```

# Eval
```bash
python3 -m fastchat.serve.cli --model-path /mnt/nfs/zhangqi/zhangqi_nfs/DLM-project/public_models/modelWeights/vicuna-7b-v1.5

python3 -m fastchat.serve.cli --model-path /mnt/nfs/zhangqi/zhangqi_nfs/DLM-project/public_models/modelWeights/vicuna-13b-v1.5
```

# Train
```bash

```