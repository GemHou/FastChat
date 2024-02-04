import torch
from trl import PPOTrainer, PPOConfig
from trl import AutoModelForCausalLMWithValueHead
from peft import LoraConfig, get_peft_model
from dataclasses import dataclass, field
import typing
import tqdm
import numpy as np
import copy
import random

from fastchat.serve.hj_utils_llm import load_llm_model, infer_llm, load_llm_setting, eval_llm_truth, judge_truth
from fastchat.serve.hj_utils_language import load_qa_pairs_from_json

BATCH_SIZE = 8
REWARD_MODE = "Truth"  # Long Short HRatio Truth


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
    device = "cuda"  # cuda cpu

    if False:
        model_name_or_path = "/mnt/nfs/zhangqi/zhangqi_nfs/DLM-project/public_models/modelWeights/vicuna-7b-v1.5"  # 13b 7b
        model_transformers, tokenizer = load_llm_model(model_path=model_name_or_path, device=device)  # -> transformers
        model_path = model_name_or_path

        if True:
            lora_args = LoraArguments(lora_r=8)  # lora_r=2, lora_dropout=0.0
            lora_config = LoraConfig(
                r=lora_args.lora_r,
                lora_alpha=lora_args.lora_alpha,
                target_modules=lora_args.lora_target_modules,
                lora_dropout=lora_args.lora_dropout,
                bias=lora_args.lora_bias,
                task_type="CAUSAL_LM",
                inference_mode=False,
            )
            model_peft = get_peft_model(model_transformers, lora_config)  # transformers -> peft
        else:
            model_peft = model_transformers
    else:
        model_path = "/mnt/nfs/houjing/repo/FastChat/data/interim/vicuna-7b-lora-CQ-v0-1217-epoch100/checkpoint-2500"
        from fastchat.model.model_adapter import get_model_adapter
        kwargs = {"torch_dtype": torch.float16, "revision": 'main'}
        adapter = get_model_adapter(model_path)
        model_peft, tokenizer = adapter.load_model(model_path, kwargs)
        model_peft.to(device)

    model_trl: "AutoModelForCausalLMWithValueHead" = AutoModelForCausalLMWithValueHead.from_pretrained(model_peft)  # peft/transformers -> trl
    # return_val = model_trl.stream_chat(tokenizer, "goodbye", history=[], max_length=2048, top_k=1, temperature=0.3, do_sample=False)
    # print("return_val: ", return_val)

    ppo_config = PPOConfig(mini_batch_size=BATCH_SIZE,
                           batch_size=BATCH_SIZE,
                           log_with="wandb",
                           learning_rate=5e-5,
                           init_kl_coef=0.05,
                           )
    ppo_trainer = PPOTrainer(config=ppo_config,
                             model=model_trl,  # trl ->
                             tokenizer=tokenizer,
                             device=device
                             )
                             
    return tokenizer, ppo_trainer, model_trl, model_path


def change_data_format(tokenizer, str_queries, str_responses, float_reward):
    batchEncoding_queries = tokenizer(str_queries, return_tensors="pt", truncation=True,)
    tensor_token_queries = batchEncoding_queries["input_ids"]
    tensor_token_queries = tensor_token_queries[0]
    batchEncoding_responses = tokenizer(str_responses, return_tensors="pt", truncation=True,)
    tensor_token_responses = batchEncoding_responses["input_ids"]
    tensor_token_responses = tensor_token_responses[0]
    tensor_value = torch.tensor(float_reward, dtype=torch.float32)
    return tensor_token_queries,tensor_token_responses,tensor_value


def log_wandb(tokenizer, ppo_trainer:PPOTrainer, tensor_value, queries, responses, stats):
    batch = {}
    batch["query"] = tokenizer.batch_decode(queries, skip_special_tokens=True)
    batch["response"] = tokenizer.batch_decode(responses, skip_special_tokens=True)
    ppo_trainer.log_stats(stats, batch, [tensor_value])


def eval_llm_once(tokenizer, model, model_path, generate_stream_func, repetition_penalty, max_new_tokens, context_len, judge_sent_end, device):
    str_prompt = "who are you?"
    print("str_prompt: ", str_prompt)
    print("str_llm_answer: ")
    str_llm_answer = infer_llm(model_path, device, model, tokenizer, generate_stream_func, repetition_penalty, max_new_tokens, context_len, judge_sent_end, str_prompt, temperature=0)


def reformat_once_dataset(tokenizer, list_str_dataset):
    str_queries = list_str_dataset[0]
    str_responses = list_str_dataset[1]
    float_reward = list_str_dataset[2]

    tensor_token_queries, tensor_token_responses, tensor_value = change_data_format(tokenizer, str_queries, str_responses, float_reward)
    # print("tensor_token_queries.shape: ", tensor_token_queries.shape)
    # print("tensor_token_responses.shape: ", tensor_token_responses.shape)
    if tensor_token_responses.shape[0] > 512:
        # 切除512以后的部分
        tensor_token_responses = tensor_token_responses[:512]
        # print("已经切除512以后的部分")
        # print("tensor_token_responses.shape: ", tensor_token_responses.shape)
    else:
        # print("没有超过512")
        pass
    
    list_tensor_dataset = [tensor_token_queries, tensor_token_responses, tensor_value]
    return list_tensor_dataset


def reformat_list_dataset(tokenizer, list_list_str_dataset):
    list_list_tensor_dataset = []
    for list_str_dataset in list_list_str_dataset:
        list_tensor_dataset = reformat_once_dataset(tokenizer, list_str_dataset)
        list_list_tensor_dataset.append(list_tensor_dataset)
    return list_list_tensor_dataset


def train_once(tokenizer, ppo_trainer:PPOTrainer, list_list_tensor_dataset):
    queries = []
    responses = []
    rewards = []
    for list_tensor_dataset in list_list_tensor_dataset:
        # queries = [list_list_tensor_dataset[0][0]]
        # responses = [list_list_tensor_dataset[0][1]]
        # rewards = [list_list_tensor_dataset[0][2]]
        queries.append(list_tensor_dataset[0])
        responses.append(list_tensor_dataset[1])
        rewards.append(list_tensor_dataset[2])
    stats = ppo_trainer.step(queries, 
                                    responses, 
                                    rewards)
    log_wandb(tokenizer, ppo_trainer, rewards, queries, responses, stats)


def calc_reward(str_llm_answer, model_path, device, model, tokenizer, generate_stream_func, repetition_penalty, max_new_tokens, context_len, judge_sent_end, corpus=None, question=None):
    if REWARD_MODE == "Long":
        float_reward = len(str_llm_answer) / 100
    elif REWARD_MODE == "Short":
        float_reward = -len(str_llm_answer) / 100
    elif REWARD_MODE == "HRatio":
        float_reward = 0
        for i in str_llm_answer:
            if i == "h":
                float_reward += 1
        float_reward /= len(str_llm_answer)
    elif REWARD_MODE == "Truth":
        str_prompt = "请根据以下语料，判断对问题的回答是否符合事实:\n语料:" + corpus + "。\n问题:" + question + "\n回答：" + str_llm_answer + "\n"
        print("str_prompt: ", str_prompt)
        print("str_llm_answer: ")
        str_llm_answer = infer_llm(model_path, device, model, tokenizer, generate_stream_func, repetition_penalty, max_new_tokens, context_len, judge_sent_end, str_prompt, temperature=0)
        truth_ratio = judge_truth(str_llm_answer)
        float_reward = truth_ratio
    else:
        raise
    print("float_reward: ", float_reward)
    return float_reward


def main():
    tokenizer, ppo_trainer, model_trl, model_path = load_trainer()

    generate_stream_func, repetition_penalty, max_new_tokens, context_len, judge_sent_end = load_llm_setting(model_path, model_trl)

    device = "cuda"

    json_file_path = '/mnt/nfs/houjing/repo/FastChat/data/interim/data_vicuna_keyword/data_vicuna_keyword_date012318_dataNum679.json'
    loaded_qa_pairs = load_qa_pairs_from_json(json_file_path)
    list_truth_ratio_llm = eval_llm_truth(loaded_qa_pairs[:10], device, model_path, model_trl, tokenizer, generate_stream_func, repetition_penalty, max_new_tokens, context_len, judge_sent_end)
    print("np.mean(list_truth_ratio_llm): ", np.mean(list_truth_ratio_llm))

    eval_llm_once(tokenizer, model_trl, model_path, generate_stream_func, repetition_penalty, max_new_tokens, context_len, judge_sent_end, device)

    update_step = 0
    while True:
        update_step += 1
        print("update_step: ", update_step)

        # collect data
        list_list_tensor_dataset = []
        for _ in range(BATCH_SIZE):
            qa_pair = random.choice(loaded_qa_pairs)
            if False:
                str_prompt = "who are you?"
            else:
                str_prompt = qa_pair["question"]
            print("str_prompt: ", str_prompt)
            print("str_llm_answer: ")
            str_llm_answer = infer_llm(model_path, device, model_trl, tokenizer, generate_stream_func, repetition_penalty, max_new_tokens, context_len, judge_sent_end, str_prompt, temperature=0.3)
            # str_llm_answer = "I str_llm_answer = infer_llm(model_path, device, model_trl, tokenizer, generate_stream_func, repetition_penalty, max_new_tokens, context_len, judge_sent_end, str_prompt" * 20
            if len(str_llm_answer) == 0:
                str_llm_answer = " "

            float_reward = calc_reward(str_llm_answer, model_path, device, model_trl, tokenizer, generate_stream_func, repetition_penalty, max_new_tokens, context_len, judge_sent_end, qa_pair["corpus"], qa_pair["question"])

            list_str_dataset = [str_prompt, str_llm_answer, float_reward]
            list_tensor_dataset = reformat_once_dataset(tokenizer, list_str_dataset)
            list_list_tensor_dataset.append(list_tensor_dataset)

        # train data
        train_once(tokenizer, ppo_trainer, list_list_tensor_dataset)

    print("Finished...")


if __name__ == "__main__":
    main()
