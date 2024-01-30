import torch
from trl import PPOTrainer, PPOConfig
from trl import AutoModelForCausalLMWithValueHead
from peft import LoraConfig, get_peft_model
from dataclasses import dataclass, field
import typing
import tqdm

from fastchat.serve.hj_utils_llm import load_llm_model, infer_llm, load_llm_setting


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

def load_trainer():
    model_name_or_path = "/mnt/nfs/zhangqi/zhangqi_nfs/DLM-project/public_models/modelWeights/vicuna-7b-v1.5"  # 13b 7b
    device = "cuda"  # cuda cpu
    model_transformers, tokenizer = load_llm_model(model_path=model_name_or_path, device=device)  # -> transformers
    model_transformers.to(torch.bfloat16)

    lora_args = LoraArguments(lora_r=2)
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.lora_target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        task_type="CAUSAL_LM",
        inference_mode=False,
    )
    model = get_peft_model(model_transformers, lora_config)  # transformers -> peft

    model: "AutoModelForCausalLMWithValueHead" = AutoModelForCausalLMWithValueHead.from_pretrained(model)  # peft/transformers -> trl

    ppo_config = PPOConfig(mini_batch_size=1,
                           batch_size=1,
                           log_with="wandb",
                           )
    ppo_trainer = PPOTrainer(config=ppo_config,
                             model=model,  # trl ->
                             tokenizer=tokenizer,
                             device=device
                             )
                             
    return tokenizer,ppo_trainer, model_transformers, model_name_or_path

def change_data_format(tokenizer, str_queries, str_responses, float_reward):
    batchEncoding_queries = tokenizer(str_queries, return_tensors="pt", truncation=True,)
    tensor_token_queries = batchEncoding_queries["input_ids"]
    tensor_token_queries = tensor_token_queries[0]
    batchEncoding_responses = tokenizer(str_responses, return_tensors="pt", truncation=True,)
    tensor_token_responses = batchEncoding_responses["input_ids"]
    tensor_token_responses = tensor_token_responses[0]
    tensor_value = torch.tensor(float_reward, dtype=torch.float32)
    return tensor_token_queries,tensor_token_responses,tensor_value


def main():
    tokenizer, ppo_trainer, model, model_path = load_trainer()

    generate_stream_func, repetition_penalty, max_new_tokens, context_len, judge_sent_end = load_llm_setting(model_path, model)

    device = "cuda"
    str_prompt = "hello"
    print("str_prompt: ", str_prompt)
    print("str_llm_answer: ")
    str_llm_answer = infer_llm(model_path, device, model, tokenizer, generate_stream_func, repetition_penalty, max_new_tokens, context_len, judge_sent_end, str_prompt, temperature=0)

    str_queries = "who are you"
    str_responses = "JingHou"
    float_reward = 10.01

    tensor_token_queries, tensor_token_responses, tensor_value = change_data_format(tokenizer, str_queries, str_responses, float_reward)
    
    for i in tqdm.tqdm(range(1000)):
        queries = [tensor_token_queries]
        responses = [tensor_token_responses]
        rewards = [tensor_value]
        stats = ppo_trainer.step(queries, 
                                responses, 
                                rewards)
        batch = {}
        batch["query"] = tokenizer.batch_decode(queries, skip_special_tokens=True)
        batch["response"] = tokenizer.batch_decode(responses, skip_special_tokens=True)
        ppo_trainer.log_stats(stats, batch, [tensor_value])
        print("stats: ", stats)

    print("Finished...")


if __name__ == "__main__":
    main()
