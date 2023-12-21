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
import re
import sys

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
import torch
from typing import Iterable, Optional, Dict
import json
import time

from fastchat.model.model_adapter import add_model_args
from fastchat.modules.awq import AWQConfig
from fastchat.modules.exllama import ExllamaConfig
from fastchat.modules.xfastertransformer import XftConfig
from fastchat.modules.gptq import GptqConfig
from fastchat.serve.inference import ChatIO, chat_loop
from fastchat.utils import str_to_torch_dtype
from fastchat.serve.cli import SimpleChatIO, RichChatIO, ProgrammaticChatIO
from fastchat.model.model_adapter import (
    load_model,
    get_conversation_template,
    get_generate_stream_function,
)
from fastchat.utils import is_partial_stop, is_sentence_complete, get_context_length
from fastchat.conversation import get_conv_template, SeparatorStyle

MODEL_PATH = "/mnt/nfs/zhangqi/zhangqi_nfs/DLM-project/public_models/modelWeights/vicuna-13b-v1.5"
TEMPERATURE = 0.9
CORPUS_LIST = ["角色可以移动，用于规避伤害，或者到达指定地点执行战术；例如：当BOSS释放一个具有高威胁的大范围伤害技能时，角色需要走到安全位置，等待伤害技能结束，避免受到大量伤害，然后回到输出位置进行攻击；例如：当BOSS战中，BOSS触发了一些机制，角色需要移动到指定机关旁，与机关交互，才能继续正常攻略BOSS。", 
               "D角色是个辅助倾向的角色，拥有减少受到伤害的技能硬化术，拥有范围内治疗队友的技能回春图腾，拥有降低目标防御力的技能脆弱术，那么在战斗开始后，D会先开始对BOSS进行常规攻击，当多名队友受到攻击，治疗倾向角色技能还在CD的时候，D会释放回春图腾，用来临时补充当作一个治疗倾向的角色，为队伍提供治疗，当坦克倾向的角色生命垂危，治疗角色还在治疗其他人时，D会对坦克角色释放硬化术，为坦克角色提供更多的减伤能力，增加存活几率，当全队开始对BOSS进行输出的时候，D会对BOSS释放脆弱术，使得全团的成员在攻击BOSS时候获得更大的收益，提升团队输出。"]

def chat_hj(
    inp,
    model_path: str,
    device: str,
    num_gpus: int,
    max_gpu_memory: str,
    dtype: Optional[torch.dtype],
    load_8bit: bool,
    cpu_offloading: bool,
    conv_template: Optional[str],
    conv_system_msg: Optional[str],
    temperature: float,
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
        for message in conv.messages[conv.offset :]:
            chatio.prompt_for_output(message[0])
            chatio.print_output(message[1])

    conv = None

    print("resetting...")
    conv = new_chat()

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

    try:
        print("------------------------------------------------------------------------------------------")
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
        print("duration: ", duration)
        num_tokens = len(tokenizer.encode(outputs))
        print("speed (token/s): ", round(num_tokens / duration, 2))
        print("------------------------------------------------------------------------------------------")

    except KeyboardInterrupt:
        print("stopped generation.")
        # If generation didn't finish
        if conv.messages[-1][1] is None:
            conv.messages.pop()
            # Remove last user message, so there isn't a double up
            if conv.messages[-1][0] == conv.roles[0]:
                conv.messages.pop()

            reload_conv(conv)
    return outputs


def main(args):
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
    try:
        inp_system = "基于以下语料，尝试生成1个问题和回答，整理成问答格式。语料："
        inp_corpus = CORPUS_LIST[0]
        inp = inp_system + inp_corpus
        outputs = chat_hj(
            inp,
            args.model_path,
            args.device,
            args.num_gpus,
            args.max_gpu_memory,
            str_to_torch_dtype(args.dtype),
            args.load_8bit,
            args.cpu_offloading,
            args.conv_template,
            args.conv_system_msg,
            args.temperature,
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
        print("outputs: ", outputs)
    except KeyboardInterrupt:
        print("exit...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument(
        "--conv-system-msg", type=str, default=None, help="Conversation system message."
    )
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
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
