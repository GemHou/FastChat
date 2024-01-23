from hj_utils_llm import load_llm_model

def main():
    model_path, device, model, tokenizer, generate_stream_func, repetition_penalty, max_new_tokens, context_len, judge_sent_end = load_llm_model()

    print("finished...")


if __name__ == "__main__":
    main()