import torch
from trl import PPOTrainer, PPOConfig
from trl import AutoModelForCausalLMWithValueHead

from fastchat.serve.hj_utils_llm import load_llm_model

def main():
    model_name_or_path = "/mnt/nfs/zhangqi/zhangqi_nfs/DLM-project/public_models/modelWeights/vicuna-13b-v1.5"
    device = "cuda"
    model, tokenizer = load_llm_model(model_path=model_name_or_path, device=device)

    model: "AutoModelForCausalLMWithValueHead" = AutoModelForCausalLMWithValueHead.from_pretrained(model)

    ppo_config = PPOConfig(mini_batch_size=2,
                           batch_size=2,
                           )
    ppo_trainer = PPOTrainer(config=ppo_config,
                             model=model,
                             tokenizer=tokenizer,                             
                             )

    str_queries = "who are you"
    batchEncoding_queries = tokenizer(str_queries, return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,)
    tensor_token_queries = batchEncoding_queries["input_ids"]  # [1, 4096]
    tensor_token_queries = tensor_token_queries[0]

    str_responses = "JingHou"
    batchEncoding_responses = tokenizer(str_responses, return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,)
    tensor_token_responses = batchEncoding_responses["input_ids"]  # [1, 4096]
    tensor_token_responses = tensor_token_responses[0]

    int_reward = 10
    tensor_value = torch.tensor(int_reward, dtype=torch.int32)
    
    stats = ppo_trainer.step([tensor_token_queries, tensor_token_queries], 
                             [tensor_token_responses, tensor_token_responses], 
                             [tensor_value, tensor_value])

    print("Finished...")


if __name__ == "__main__":
    main()
