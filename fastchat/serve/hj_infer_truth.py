import json
import tqdm
import numpy as np

from hj_utils_llm import load_llm_model, infer_llm

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

def val_dataset_truth(loaded_qa_pairs, model_path, device, model, tokenizer, generate_stream_func, repetition_penalty, max_new_tokens, context_len, judge_sent_end):
    list_truth_ratio = []
    for qa_pair in tqdm.tqdm(loaded_qa_pairs):
        question = qa_pair["question"]
        answer = qa_pair["answer"]
        corpus = qa_pair["corpus"]
        str_prompt = "请根据以下语料，判断对问题的回答是否符合事实:\n语料:" + corpus + "。\n问题:" + question + "\n回答：" + answer + "\n"
        print("str_prompt: ", str_prompt)
        print("truth answer: ")
        str_llm_answer = infer_llm(model_path, device, model, tokenizer, generate_stream_func, repetition_penalty, max_new_tokens, context_len, judge_sent_end, str_prompt)
        # print("str_llm_answer: ", str_llm_answer)
        truth_ratio = judge_truth(str_llm_answer)
        print("truth_ratio: ", truth_ratio)
        list_truth_ratio.append(truth_ratio)
    return list_truth_ratio


def main():
    print("Loading...")
    json_file_path = '/mnt/nfs/houjing/repo/FastChat/data/interim/data_vicuna_keyword/data_vicuna_keyword_date012318_dataNum679.json'
    loaded_qa_pairs = load_qa_pairs_from_json(json_file_path)
    loaded_qa_pairs = loaded_qa_pairs[:100]
    model_path, device, model, tokenizer, generate_stream_func, repetition_penalty, max_new_tokens, context_len, judge_sent_end = load_llm_model()

    print("Processing...")
    list_truth_ratio = []
    for qa_pair in tqdm.tqdm(loaded_qa_pairs):
        question = qa_pair["question"]
        answer = qa_pair["answer"]
        corpus = qa_pair["corpus"]
        str_prompt = question
        print("str_prompt: ", str_prompt)
        print("truth answer: ")
        str_llm_answer = infer_llm(model_path, device, model, tokenizer, generate_stream_func, repetition_penalty, max_new_tokens, context_len, judge_sent_end, str_prompt)
        print("str_llm_answer: ", str_llm_answer)
    # list_truth_ratio = val_dataset_truth(loaded_qa_pairs, model_path, device, model, tokenizer, generate_stream_func, repetition_penalty, max_new_tokens, context_len, judge_sent_end)

    print("Saving...")
    print("np.mean(list_truth_ratio): ", np.mean(list_truth_ratio))

    print("Finished...")


if __name__ == "__main__":
    main()
