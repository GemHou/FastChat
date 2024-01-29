import torch
from trl import PPOTrainer, PPOConfig
from trl import AutoModelForCausalLMWithValueHead
from peft import LoraConfig, get_peft_model
from dataclasses import dataclass, field
import typing

from fastchat.serve.hj_utils_llm import load_llm_model


@dataclass
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: typing.List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


def main():
    model_name_or_path = "/mnt/nfs/zhangqi/zhangqi_nfs/DLM-project/public_models/modelWeights/vicuna-7b-v1.5"  # 13b 7b
    device = "cuda"  # cuda cpu
    model, tokenizer = load_llm_model(model_path=model_name_or_path, device=device)  # -> transformers
    # tokenizer.model_max_length = 10
    model.to(torch.bfloat16)

    lora_args = LoraArguments(lora_r=2)
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.lora_target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        task_type="CAUSAL_LM",
        # target_modules=[name.strip() for name in lora_target.split(",")],
        # init_lora_weights="loftq",
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)  # transformers -> peft

    model: "AutoModelForCausalLMWithValueHead" = AutoModelForCausalLMWithValueHead.from_pretrained(model)  # peft/transformers -> trl

    ppo_config = PPOConfig(mini_batch_size=1,
                           batch_size=1,
                           )
    ppo_trainer = PPOTrainer(config=ppo_config,
                             model=model,  # trl ->
                             tokenizer=tokenizer,
                             device=device
                             )

    str_queries = "who are you"
    batchEncoding_queries = tokenizer(str_queries, return_tensors="pt",
        # padding="max_length",
        # max_length=tokenizer.model_max_length,
        truncation=True,)
    tensor_token_queries = batchEncoding_queries["input_ids"]  # [1, 4096]
    tensor_token_queries = tensor_token_queries[0]

    str_responses = "JingHou"
    batchEncoding_responses = tokenizer(str_responses, return_tensors="pt",
        # padding="max_length",
        # max_length=tokenizer.model_max_length,
        truncation=True,)
    tensor_token_responses = batchEncoding_responses["input_ids"]  # [1, 4096]
    tensor_token_responses = tensor_token_responses[0]

    int_reward = 10.01
    tensor_value = torch.tensor(int_reward, dtype=torch.float32)
    
    stats = ppo_trainer.step([tensor_token_queries], 
                             [tensor_token_responses], 
                             [tensor_value])

    print("Finished...")


if __name__ == "__main__":
    main()
