import time

from fastchat.model.model_adapter import get_generate_stream_function, load_model, get_conversation_template
from fastchat.modules.gptq import GptqConfig
from fastchat.modules.awq import AWQConfig
from fastchat.utils import get_context_length
from fastchat.serve.cli import SimpleChatIO

from hj_utils_language import split_text_by_dot_and_semicolon

INPUT_FILE_NAME = './data/raw/corpus_20231228_human.txt'  # None
MODEL_PATH = "/mnt/nfs/zhangqi/zhangqi_nfs/DLM-project/public_models/modelWeights/vicuna-13b-v1.5"

def merge_and_deduplicate_keywords(list_list_keywords):
    list_keywords = []

    for sublist in list_list_keywords:
        for keyword in sublist:
            if keyword not in list_keywords:
                list_keywords.append(keyword)

    return list_keywords

def get_list_keywords():
    list_list_keywords = [['团战世界', '游戏规则', '团队协同', '角色能力', '敌人击败', '治疗', '输出角色', '大量伤害', '防御', '承伤职业', '仇恨', 'BOSS战', '基础属性', '伤害计算', '技能释放', 'CD', '冷却时间', '单体伤害', '范围伤害', '单体恢复', '范围恢复', '敌对单位', '攻击优先级', '仇恨列表', 'BOSS攻击对象', '增加仇恨', '减少仇恨', '仇恨控制', '伤害预测', 'BOSS伤害对象', '仇恨累积速度', '普通伤害技能', '治疗技能', '高仇恨技能', 'BOSS攻击保护', '无法承受BOSS攻击'],
        ['团战世界', '游戏规则', '游戏胜负判定', '杂兵战', 'BOSS战', '我方人员全部被消灭', '战斗时间', '敌人强化', 'BOSS狂暴化', '生存', '高效输出', '职业倾向', '坦克倾向', '输出倾向', '治疗倾向', '辅助倾向', '角色特性', '防御类属性', '仇恨技能', '自保技能', '破甲', '嘲讽', '盾墙', '火球', '炎爆', '火焰冲击', '小恢复术', '大恢复术', '生命回流', '圣盾术', '硬化术', '回春图腾', '脆弱术', '战斗流程', '仇恨控制', 'CD管理', '输出优化', '治疗策略', '辅助技能'],
        ['团战世界', '游戏规则', '减伤', 'BUFF', '伤害减少', '治疗职业', '阵亡概率', '指挥', '大招', '坦克开减伤', 'AOE', '群体减伤', 'Rush', '固定流程', '切换目标', '敌人掉血', 'AOE技能', 'T', '主T', '副T', '吸引仇恨', '控制仇恨', '吃药', '生命值提升', 'OT', '仇恨失控', '嘲讽技能', '团灭', 'T拉回来', '奶妈抬血', '点名', '压一波血', '灌伤害']]
    # print("keywords_list: ", list_list_keywords)

    list_keywords = merge_and_deduplicate_keywords(list_list_keywords)
    return list_keywords

def get_list_corpus():
    if INPUT_FILE_NAME is None:
        input_file_name = './data/raw/corpus.txt'
    else:
        input_file_name = INPUT_FILE_NAME
    with open(input_file_name, 'r', encoding='utf-8') as file:
        str_full_corpus = file.read()  # 读取文件的全部内容
        # print("str_full_corpus: ", str_full_corpus)
    # print("str_full_corpus: ", str_full_corpus)
    list_corpus = split_text_by_dot_and_semicolon(str_full_corpus)
    return list_corpus

def get_list_valid_keywords_sentences(list_keywords, list_corpus):
    list_valid_keywords_sentences = []
    for keyword in list_keywords:
        # print("keyword: ", keyword)
        mentioned_sentence_num = 0
        list_mentioned_sentences = []
        for sentence in list_corpus:
            if keyword in sentence:
                # print("sentence: ", sentence)
                mentioned_sentence_num += 1
                list_mentioned_sentences.append(sentence)
        if mentioned_sentence_num > 0 and mentioned_sentence_num / len(list_keywords) < 0.4:
            list_valid_keywords_sentences.append([keyword, list_mentioned_sentences])
    return list_valid_keywords_sentences


def new_chat(model_path):
        conv = get_conversation_template(model_path)
        return conv


def main():
    list_keywords = get_list_keywords()
    # print("list_keywords: ", list_keywords)

    list_corpus = get_list_corpus()
    # print("list_corpus: ", list_corpus)

    list_valid_keywords_sentences = get_list_valid_keywords_sentences(list_keywords, list_corpus)
    # print("list_valid_keywords_sentences: ", list_valid_keywords_sentences)

    model_path = MODEL_PATH
    device = "cuda"
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
    generate_stream_func = get_generate_stream_function(model, model_path)
    inp = "hello"
    conv = new_chat(model_path)
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    temperature = 0.7
    repetition_penalty = 1.0
    max_new_tokens = 2048
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
    context_len = get_context_length(model.config)
    judge_sent_end = False
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
    print("outputs: ", outputs)


if __name__ == "__main__":
    main()
