"""
Chat with a model with command line interface.

Usage:
python3 -m fastchat.serve.cli --model lmsys/vicuna-7b-v1.5
python3 -m fastchat.serve.cli --model lmsys/fastchat-t5-3b-v1.0

Other commands:
- Type "!!exit" or an empty line to exit.
- Type "!!reset" to start a new conversation.
- Type "!!remove" to remove the last prompt.
- Type "!!regen" to regenerate the last message.
- Type "!!save <filename>" to save the conversation history to a json file.
- Type "!!load <filename>" to load a conversation history from a json file.
"""
import argparse
import os

import torch
from typing import Optional
import time
import tqdm

from fastchat.model.model_adapter import add_model_args
from fastchat.modules.awq import AWQConfig
from fastchat.modules.exllama import ExllamaConfig
from fastchat.modules.xfastertransformer import XftConfig
from fastchat.modules.gptq import GptqConfig
from fastchat.serve.hj_extract_qa_pairs import extract_qa_pairs
from fastchat.serve.inference import ChatIO
from fastchat.utils import str_to_torch_dtype
from fastchat.serve.cli import SimpleChatIO, RichChatIO, ProgrammaticChatIO
from fastchat.model.model_adapter import (
    load_model,
    get_conversation_template,
    get_generate_stream_function,
)
from fastchat.utils import get_context_length
from fastchat.conversation import get_conv_template
import random
from hj_utils_language import split_text_by_dot_and_semicolon, get_date, save_qa_pairs_to_json

MODEL_PATH = "/mnt/nfs/zhangqi/zhangqi_nfs/DLM-project/public_models/modelWeights/vicuna-13b-v1.5"
# TEMPERATURE = 0.8
INPUT_FILE_NAME = './data/raw/corpus_20231228_human.txt'  # None


def corpus_2_outputs(model_path, device, temperature, repetition_penalty, max_new_tokens, chatio, judge_sent_end, debug,
                     model, tokenizer, generate_stream_func, is_codet5p, context_len, reload_conv, conv, inp_system,
                     corpus):
    inp = inp_system + corpus

    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    if is_codet5p:  # codet5p is a code completion model.
        prompt = inp

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

    # try:
        # print("------------------------------------------------------------------------------------------")
    chatio.prompt_for_output(conv.roles[1])
    output_stream = generate_stream_func(
        model,
        tokenizer,
        gen_params,
        device,
        context_len=context_len,
        judge_sent_end=judge_sent_end,
    )
    t = time.time()
    outputs = chatio.stream_output(output_stream)
    duration = time.time() - t
    conv.update_last_message(outputs.strip())

    if debug:
        num_tokens = len(tokenizer.encode(outputs))
        msg = {
            "conv_template": conv.name,
            "prompt": prompt,
            "outputs": outputs,
            "speed (token/s)": round(num_tokens / duration, 2),
        }
        print(f"\n{msg}\n")
    # print("duration: ", duration)
    num_tokens = len(tokenizer.encode(outputs))
    # print("speed (token/s): ", round(num_tokens / duration, 2))
    # print("------------------------------------------------------------------------------------------")

    # except KeyboardInterrupt:
    #     print("stopped generation.")
    #     # If generation didn't finish
    #     if conv.messages[-1][1] is None:
    #         conv.messages.pop()
    #         # Remove last user message, so there isn't a double up
    #         if conv.messages[-1][0] == conv.roles[0]:
    #             conv.messages.pop()

    #         reload_conv(conv)
    #     outputs = None
    return outputs


def chat_hj(
        list_corpus,
        model_path: str,
        device: str,
        num_gpus: int,
        max_gpu_memory: str,
        dtype: Optional[torch.dtype],
        load_8bit: bool,
        cpu_offloading: bool,
        conv_template: Optional[str],
        conv_system_msg: Optional[str],
        # temperature: float,
        repetition_penalty: float,
        max_new_tokens: int,
        chatio: ChatIO,
        gptq_config: Optional[GptqConfig] = None,
        awq_config: Optional[AWQConfig] = None,
        exllama_config: Optional[ExllamaConfig] = None,
        xft_config: Optional[XftConfig] = None,
        revision: str = "main",
        judge_sent_end: bool = True,
        debug: bool = True,
        history: bool = True,
):
    # Model
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
    generate_stream_func = get_generate_stream_function(model, model_path)

    model_type = str(type(model)).lower()
    is_t5 = "t5" in model_type
    is_codet5p = "codet5p" in model_type
    is_xft = "xft" in model_type

    # Hardcode T5's default repetition penalty to be 1.2
    if is_t5 and repetition_penalty == 1.0:
        repetition_penalty = 1.2

    # Set context length
    context_len = get_context_length(model.config)

    # Chat
    def new_chat():
        if conv_template:
            conv = get_conv_template(conv_template)
        else:
            conv = get_conversation_template(model_path)
        if conv_system_msg is not None:
            conv.set_system_message(conv_system_msg)
        return conv

    def reload_conv(conv):
        """
        Reprints the conversation from the start.
        """
        for message in conv.messages[conv.offset:]:
            chatio.prompt_for_output(message[0])
            chatio.print_output(message[1])

    # print("resetting...")
    conv = new_chat()

    inp_system = "基于以下语料，尝试生成1个简洁精简的问题和回答，整理成问答格式，不要胡编乱造内容。语料："
    list_outputs = []
    for corpus_i in range(100):
        print(" ")
        probability = random.random()
        if probability < 1/4:
            print("corpus 1")
            # str_corpus = list_corpus[corpus_i]
            str_corpus = random.choice(list_corpus)
        elif probability < 2/4:
            print("corpus 2")
            # 生成一个随机的起始索引
            start_index = random.randint(0, len(list_corpus) - 2)

            # 选择起始索引和它后续的元素，共两个元素
            random_elements = list_corpus[start_index:start_index + 2]
            random.shuffle(random_elements)
            str_corpus = random_elements[0] + "。" + random_elements[1]
        elif probability < 3/4:
            print("corpus 3")
            # 生成一个随机的起始索引
            start_index = random.randint(0, len(list_corpus) - 3)

            # 选择起始索引和它后续的元素，共两个元素
            random_elements = list_corpus[start_index:start_index + 3]
            random.shuffle(random_elements)
            str_corpus = random_elements[0] + "。" + random_elements[1] + "。" + random_elements[2]
        else:
            print("corpus 2 R")
            str_corpuses = random.sample(list_corpus, 2)
            str_corpus = str_corpuses[0] + "。" + str_corpuses[1]

        print("str_corpus: ", str_corpus)
        temperature = 0.7 + random.random() * 0.2
        str_outputs = corpus_2_outputs(model_path, device, temperature, repetition_penalty, max_new_tokens, chatio,
                                       judge_sent_end, debug, model, tokenizer, generate_stream_func, is_codet5p,
                                       context_len, reload_conv, conv, inp_system, str_corpus)
        # print("str_outputs: ", str_outputs)
        list_outputs.append(str_outputs)

    return list_outputs


def corpus_2_strQa(args, list_corpus):
    args.model_path = MODEL_PATH
    if args.gpus:
        if len(args.gpus.split(",")) < args.num_gpus:
            raise ValueError(
                f"Larger --num-gpus ({args.num_gpus}) than --gpus {args.gpus}!"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        os.environ["XPU_VISIBLE_DEVICES"] = args.gpus
    if args.enable_exllama:
        exllama_config = ExllamaConfig(
            max_seq_len=args.exllama_max_seq_len,
            gpu_split=args.exllama_gpu_split,
            cache_8bit=args.exllama_cache_8bit,
        )
    else:
        exllama_config = None
    if args.enable_xft:
        xft_config = XftConfig(
            max_seq_len=args.xft_max_seq_len,
            data_type=args.xft_dtype,
        )
        if args.device != "cpu":
            print("xFasterTransformer now is only support CPUs. Reset device to CPU")
            args.device = "cpu"
    else:
        xft_config = None
    if args.style == "simple":
        chatio = SimpleChatIO(args.multiline)
    elif args.style == "rich":
        chatio = RichChatIO(args.multiline, args.mouse)
    elif args.style == "programmatic":
        chatio = ProgrammaticChatIO()
    else:
        raise ValueError(f"Invalid style for console: {args.style}")
    list_str_qa = chat_hj(
        list_corpus,
        args.model_path,
        args.device,
        args.num_gpus,
        args.max_gpu_memory,
        str_to_torch_dtype(args.dtype),
        args.load_8bit,
        args.cpu_offloading,
        args.conv_template,
        args.conv_system_msg,
        # args.temperature,
        args.repetition_penalty,
        args.max_new_tokens,
        chatio,
        gptq_config=GptqConfig(
            ckpt=args.gptq_ckpt or args.model_path,
            wbits=args.gptq_wbits,
            groupsize=args.gptq_groupsize,
            act_order=args.gptq_act_order,
        ),
        awq_config=AWQConfig(
            ckpt=args.awq_ckpt or args.model_path,
            wbits=args.awq_wbits,
            groupsize=args.awq_groupsize,
        ),
        exllama_config=exllama_config,
        xft_config=xft_config,
        revision=args.revision,
        judge_sent_end=args.judge_sent_end,
        debug=args.debug,
        history=not args.no_history,
    )
    return list_str_qa


def main(args):
    print("Loading...")
    if INPUT_FILE_NAME is None:
        input_file_name = './data/raw/corpus.txt'
    else:
        input_file_name = INPUT_FILE_NAME
    with open(input_file_name, 'r', encoding='utf-8') as file:
        str_full_corpus = file.read()  # 读取文件的全部内容
        # print("str_full_corpus: ", str_full_corpus)

    list_corpus = split_text_by_dot_and_semicolon(str_full_corpus)
    # print("list_corpus: ", list_corpus)
    str_date = get_date()
    list_qa = []
    start_time = time.time()
    while time.time() - start_time < 60 * 60 * 24 * 3:
        list_str_qa = corpus_2_strQa(args, list_corpus)

        # print("list_str_qa: ", list_str_qa)
        list_qa_temp = extract_qa_pairs(list_str_qa)
        list_qa = list_qa + list_qa_temp
        str_data_num = "_dataNum" + str(len(list_qa))
        output_file = './data/interim/data_vicuna' + '/data_vicuna' + str_date + str_data_num + '.json'
        save_qa_pairs_to_json(list_qa, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument(
        "--conv-system-msg", type=str, default=None, help="Conversation system message."
    )
    # parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument(
        "--style",
        type=str,
        default="simple",
        choices=["simple", "rich", "programmatic"],
        help="Display style.",
    )
    parser.add_argument(
        "--multiline",
        action="store_true",
        help="Enable multiline input. Use ESC+Enter for newline.",
    )
    parser.add_argument(
        "--mouse",
        action="store_true",
        help="[Rich Style]: Enable mouse support for cursor positioning.",
    )
    parser.add_argument(
        "--judge-sent-end",
        action="store_true",
        help="Whether enable the correction logic that interrupts the output of sentences due to EOS.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print useful debug information (e.g., prompts)",
    )
    args = parser.parse_args()
    main(args)
