import random
import time

from fastchat.modules.gptq import GptqConfig
from fastchat.modules.awq import AWQConfig
from fastchat.model.model_adapter import get_generate_stream_function, load_model
from fastchat.utils import get_context_length

from fastchat.model.model_adapter import get_conversation_template
from fastchat.serve.cli import SimpleChatIO

MODEL_PATH = "/mnt/nfs/zhangqi/zhangqi_nfs/DLM-project/public_models/modelWeights/vicuna-13b-v1.5"

def load_llm_setting(model_path, model):
    generate_stream_func = get_generate_stream_function(model, model_path)
    repetition_penalty = 1.0
    max_new_tokens = 2048
    context_len = get_context_length(model.config)
    judge_sent_end = False
    return generate_stream_func,repetition_penalty,max_new_tokens,context_len,judge_sent_end

def load_llm_model(model_path = MODEL_PATH, device = "cuda"):
    num_gpus = 1
    max_gpu_memory = None
    dtype = None
    load_8bit = False
    cpu_offloading = False
    gptq_ckpt = None
    gptq_wbits = 16
    gptq_groupsize = -1
    gptq_act_order = False
    gptq_config=GptqConfig(
            ckpt=gptq_ckpt or model_path,
            wbits=gptq_wbits,
            groupsize=gptq_groupsize,
            act_order=gptq_act_order,
        )
    awq_ckpt = None
    awq_wbits = 16
    awq_groupsize = -1
    awq_config=AWQConfig(
            ckpt=awq_ckpt or model_path,
            wbits=awq_wbits,
            groupsize=awq_groupsize,
        )
    exllama_config = None
    xft_config = None
    revision = "main"
    debug = False
    model, tokenizer = load_model(
        model_path,
        device=device,
        num_gpus=num_gpus,
        max_gpu_memory=max_gpu_memory,
        dtype=dtype,
        load_8bit=load_8bit,
        cpu_offloading=cpu_offloading,
        gptq_config=gptq_config,
        awq_config=awq_config,
        exllama_config=exllama_config,
        xft_config=xft_config,
        revision=revision,
        debug=debug,
    )
    # generate_stream_func, repetition_penalty, max_new_tokens, context_len, judge_sent_end = load_llm_setting(model_path, model)
    return model,tokenizer



def new_chat(model_path):
        conv = get_conversation_template(model_path)
        return conv


def infer_llm(model_path, device, model, tokenizer, generate_stream_func, repetition_penalty, max_new_tokens, context_len, judge_sent_end, str_prompt):
    conv = new_chat(model_path)
    conv.append_message(conv.roles[0], str_prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    temperature = 0.7 + random.random() * 0.2
    gen_params = {
                    "model": model_path,
                    "prompt": prompt,
                    "temperature": temperature,
                    "repetition_penalty": repetition_penalty,
                    "max_new_tokens": max_new_tokens,
                    "stop": conv.stop_str,
                    "stop_token_ids": conv.stop_token_ids,
                    "echo": False,
                }
    output_stream = generate_stream_func(
                    model,
                    tokenizer,
                    gen_params,
                    device,
                    context_len=context_len,
                    judge_sent_end=judge_sent_end,
                )
    t = time.time()
    multiline = False
    chatio = SimpleChatIO(multiline)
    outputs = chatio.stream_output(output_stream)
    duration = time.time() - t

    print("duration: ", duration)
    return outputs
