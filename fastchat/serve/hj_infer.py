import torch
from hj_utils_llm import load_llm_setting, infer_llm


def main():
    print("Loading...")
    model_path = "/mnt/nfs/houjing/repo/FastChat/data/interim/vicuna-7b-lora-CQ-v0-1217-epoch100/checkpoint-2500"
    from fastchat.model.model_adapter import get_model_adapter
    kwargs = {"torch_dtype": torch.float16, "revision": 'main'}
    adapter = get_model_adapter(model_path)
    model, tokenizer = adapter.load_model(model_path, kwargs)
    model.to("cuda")
    generate_stream_func, repetition_penalty, max_new_tokens, context_len, judge_sent_end = load_llm_setting(model_path, model)
    str_prompt = "输出倾向角色有哪些技能？"
    print("str_prompt: ", str_prompt)
    print("str_llm_answer: ")
    str_llm_answer, str_prompt_wSystem = infer_llm(model_path, "cuda", model, tokenizer, generate_stream_func, repetition_penalty, max_new_tokens, context_len, judge_sent_end, str_prompt, temperature=0)

    print("Finished...")


if __name__ == "__main__":
    main()
