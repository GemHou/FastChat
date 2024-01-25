import json
import tqdm
import numpy as np
import torch

from hj_utils_llm import load_llm_model, infer_llm, load_llm_setting

def load_qa_pairs_from_json(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    qa_pairs = []
    for conv_data in data:
        human_msg = next(msg["value"] for msg in conv_data["conversations"] if msg["from"] == "human")
        gpt_msg = next(msg["value"] for msg in conv_data["conversations"] if msg["from"] == "gpt")
        corpus_qa = conv_data.get("corpus", None)

        qa_pairs.append({
            "question": human_msg,
            "answer": gpt_msg,
            "corpus": corpus_qa
        })

    return qa_pairs

def judge_truth(str_llm_answer):
    if "回答的内容是符合事实的" in str_llm_answer:
        truth_ratio = 1
    elif "回答的答案是正确" in str_llm_answer:
        truth_ratio = 1
    elif "回答是符合事实的" in str_llm_answer:
        truth_ratio = 1
    elif "内容与事实不符" in str_llm_answer:
        truth_ratio = 0
    elif "回答不符合事实" in str_llm_answer:
        truth_ratio = 0
    elif "回答与语料相符" in str_llm_answer:
        truth_ratio = 1
    elif "回答符合事实" in str_llm_answer:
        truth_ratio = 1
    elif "回答是正确的" in str_llm_answer:
        truth_ratio = 1
    elif "我无法回答" in str_llm_answer:
        truth_ratio = 0.5
    elif "符合事实" in str_llm_answer:
        truth_ratio = 1
    else:
        print("Unknown truth!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        truth_ratio = 0.5
    return truth_ratio

def eval_dataset_truth(loaded_qa_pairs, model_path, device, model, tokenizer, generate_stream_func, repetition_penalty, max_new_tokens, context_len, judge_sent_end):
    list_truth_ratio = []
    for qa_pair in tqdm.tqdm(loaded_qa_pairs):
        question = qa_pair["question"]
        answer_llm_env = qa_pair["answer"]
        corpus = qa_pair["corpus"]
        str_prompt = "请根据以下语料，判断对问题的回答是否符合事实:\n语料:" + corpus + "。\n问题:" + question + "\n回答：" + answer_llm_env + "\n"
        print("str_prompt: ", str_prompt)
        print("str_llm_answer: ")
        str_llm_answer = infer_llm(model_path, device, model, tokenizer, generate_stream_func, repetition_penalty, max_new_tokens, context_len, judge_sent_end, str_prompt, temperature=0)
        truth_ratio = judge_truth(str_llm_answer)
        print("truth_ratio: ", truth_ratio)
        list_truth_ratio.append(truth_ratio)
    return list_truth_ratio

def eval_llm_truth(loaded_qa_pairs, device, model_path, model, tokenizer, generate_stream_func, repetition_penalty, max_new_tokens, context_len, judge_sent_end):
    list_truth_ratio = []
    for qa_pair in tqdm.tqdm(loaded_qa_pairs):
        question = qa_pair["question"]
        answer_llm_env = qa_pair["answer"]
        corpus = qa_pair["corpus"]
        str_prompt = question
        print("str_prompt: ", str_prompt)
        print("str_llm_answer: ")
        str_llm_answer = infer_llm(model_path, device, model, tokenizer, generate_stream_func, repetition_penalty, max_new_tokens, context_len, judge_sent_end, str_prompt, temperature=0)
        answer_llm_policy = str_llm_answer

        str_prompt = "请根据以下语料，判断对问题的回答是否符合事实:\n语料:" + corpus + "。\n问题:" + question + "\n回答：" + answer_llm_policy + "\n"
        print("str_prompt: ", str_prompt)
        print("str_llm_answer: ")
        str_llm_answer = infer_llm(model_path, device, model, tokenizer, generate_stream_func, repetition_penalty, max_new_tokens, context_len, judge_sent_end, str_prompt, temperature=0)
        truth_ratio = judge_truth(str_llm_answer)
        print("truth_ratio: ", truth_ratio)
        list_truth_ratio.append(truth_ratio)
    return list_truth_ratio


def main():
    print("Loading...")
    json_file_path = '/mnt/nfs/houjing/repo/FastChat/data/interim/data_vicuna_keyword/data_vicuna_keyword_date012318_dataNum679.json'
    loaded_qa_pairs = load_qa_pairs_from_json(json_file_path)
    loaded_qa_pairs = loaded_qa_pairs[:100]

    device = "cuda"
    if False:
        model_name_or_path = "/mnt/nfs/zhangqi/zhangqi_nfs/DLM-project/public_models/modelWeights/vicuna-13b-v1.5"
        model, tokenizer = load_llm_model(model_path=model_name_or_path, device=device)
        generate_stream_func, repetition_penalty, max_new_tokens, context_len, judge_sent_end = load_llm_setting(model_path=model_name_or_path, model=model)
    else:
        model_path = "/mnt/nfs/houjing/repo/FastChat/data/interim/vicuna-7b-lora-CQ-v0-1217-epoch100/checkpoint-2500"
        from fastchat.model.model_adapter import get_model_adapter
        kwargs = {"torch_dtype": torch.float16, "revision": 'main'}
        adapter = get_model_adapter(model_path)
        model, tokenizer = adapter.load_model(model_path, kwargs)
        model.to(device)
        generate_stream_func, repetition_penalty, max_new_tokens, context_len, judge_sent_end = load_llm_setting(model_path, model)

    print("Processing...")
    list_truth_ratio_llm = eval_llm_truth(loaded_qa_pairs, device, model_path, model, tokenizer, generate_stream_func, repetition_penalty, max_new_tokens, context_len, judge_sent_end)
    print("np.mean(list_truth_ratio_llm): ", np.mean(list_truth_ratio_llm))
    list_truth_ratio_dataset = eval_dataset_truth(loaded_qa_pairs, model_path, device, model, tokenizer, generate_stream_func, repetition_penalty, max_new_tokens, context_len, judge_sent_end)
    print("np.mean(list_truth_ratio_dataset): ", np.mean(list_truth_ratio_dataset))

    print("Saving...")
    pass

    print("Finished...")


if __name__ == "__main__":
    main()
