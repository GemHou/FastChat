from hj_utils_llm import load_llm_model, infer_llm

def main():
    model_path, device, model, tokenizer, generate_stream_func, repetition_penalty, max_new_tokens, context_len, judge_sent_end = load_llm_model()

    str_prompt = "hello"
    outputs = infer_llm(model_path, device, model, tokenizer, generate_stream_func, repetition_penalty, max_new_tokens, context_len, judge_sent_end, str_prompt)

    str_prompt = "你好"
    outputs = infer_llm(model_path, device, model, tokenizer, generate_stream_func, repetition_penalty, max_new_tokens, context_len, judge_sent_end, str_prompt)

    print("finished...")


if __name__ == "__main__":
    main()