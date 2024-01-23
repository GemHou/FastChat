import json

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


def main():
    print("Loading...")
    json_file_path = '/mnt/nfs/houjing/repo/FastChat/data/interim/data_vicuna_keyword/data_vicuna_keyword_date012316_dataNum3.json'
    loaded_qa_pairs = load_qa_pairs_from_json(json_file_path)
    model_path, device, model, tokenizer, generate_stream_func, repetition_penalty, max_new_tokens, context_len, judge_sent_end = load_llm_model()

    print("Processing...")
    str_prompt = "hello"
    outputs = infer_llm(model_path, device, model, tokenizer, generate_stream_func, repetition_penalty, max_new_tokens, context_len, judge_sent_end, str_prompt)

    str_prompt = "你好"
    outputs = infer_llm(model_path, device, model, tokenizer, generate_stream_func, repetition_penalty, max_new_tokens, context_len, judge_sent_end, str_prompt)

    print("Saving...")
    pass

    print("Finished...")


if __name__ == "__main__":
    main()