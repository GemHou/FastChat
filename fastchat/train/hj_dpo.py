import torch
import copy
from trl import AutoModelForCausalLMWithValueHead
# from trl import PPOTrainer, PPOConfig
from trl import DPOTrainer
from transformers import Seq2SeqTrainingArguments

from fastchat.model.model_adapter import get_model_adapter
from fastchat.serve.hj_utils_llm import load_llm_setting, infer_llm

def main():
    device = "cuda"  # cuda cpu
    model_path = "/mnt/nfs/houjing/repo/FastChat/data/interim/vicuna-7b-lora-CQ-v0-1217-epoch100/checkpoint-2500"
    kwargs = {"torch_dtype": torch.float16, "revision": 'main'}
    adapter = get_model_adapter(model_path)
    model_peft, tokenizer = adapter.load_model(model_path, kwargs)
    model_peft.to(device)

    model_peft_copy_disable = copy.deepcopy(model_peft)
    model_peft_copy_disable.disable_adapter_layers()

    model_trl: "AutoModelForCausalLMWithValueHead" = AutoModelForCausalLMWithValueHead.from_pretrained(model_peft)  # peft/transformers -> trl

    # training_args_dict = training_args.to_dict()
    # training_args_dict.update(dict(remove_unused_columns=False))  # important for pairwise dataset
    training_args_dict = dict(remove_unused_columns=False, output_dir="./")
    training_args = Seq2SeqTrainingArguments(**training_args_dict)

    dpo_trainer = DPOTrainer(model=model_trl,  # trl ->
                            tokenizer=tokenizer,
                            args=training_args,
                            # device=device
                            )
    
    generate_stream_func, repetition_penalty, max_new_tokens, context_len, judge_sent_end = load_llm_setting(model_path, model_trl)

    str_prompt = "who are you?"
    print("str_prompt: ", str_prompt)
    print("str_llm_answer: ")
    str_llm_answer = infer_llm(model_path, device, model_trl, tokenizer, generate_stream_func, repetition_penalty, max_new_tokens, context_len, judge_sent_end, str_prompt, temperature=0)

    print("finished...")

if __name__ == "__main__":
    main()
