import torch
import transformers
import typing
from dataclasses import dataclass, field
from peft.utils.other import _set_trainable

from hj_utils_llm import load_llm_setting, infer_llm




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

def main():
    print("Loading...")
    model_path = "/mnt/nfs/houjing/repo/FastChat/data/interim/vicuna-13b-lora-CQ-v0-1219-epoch10-lr2em4-vdata37232/checkpoint-10470"
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

    model.print_trainable_parameters()
    # model.train_adapter("default")
    # for param in model.active_adapters["default"].parameters():
    #     param.requires_grad = True

    # for param in model.parameters():
    #     param.requires_grad = True
    for name, param in model.named_parameters(recurse=True):
        if 'lora' in name:
            print("name: ", name)
            param.requires_grad = True
    model.print_trainable_parameters()

    print("Finished...")

    # from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    # parser = transformers.HfArgumentParser(
    #     (LoraArguments)
    # )
    # print("type(model): ", type(model))
    # (
    #     lora_args,
    # ) = parser.parse_args_into_dataclasses()
    # lora_config = LoraConfig(
    #     r=lora_args.lora_r,
    #     lora_alpha=lora_args.lora_alpha,
    #     target_modules=lora_args.lora_target_modules,
    #     lora_dropout=lora_args.lora_dropout,
    #     bias=lora_args.lora_bias,
    #     task_type="CAUSAL_LM",
    # )

    # model = get_peft_model(model, lora_config)
    # print("type(model): ", type(model))

    # model.print_trainable_parameters()
    # for name, param in model.named_parameters(recurse=True):
    #     if param.requires_grad == True:
    #         # print("name: ", name)
    #         pass
    #     param.requires_grad = True
    #     if 'base_model' not in name:
    #         print("name: ", name)
    # model.print_trainable_parameters()
    # # _set_trainable(model, "default")

    # str_prompt = "输出倾向角色有哪些技能？"
    # print("str_prompt: ", str_prompt)
    # print("str_llm_answer: ")
    # str_llm_answer, str_prompt_wSystem = infer_llm(model_path, "cuda", model, tokenizer, generate_stream_func, repetition_penalty, max_new_tokens, context_len, judge_sent_end, str_prompt, temperature=0)

    # print("Finished 2...")


if __name__ == "__main__":
    main()
