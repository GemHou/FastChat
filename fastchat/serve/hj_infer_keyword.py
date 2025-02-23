import time
import tqdm

from fastchat.serve.hj_extract_qa_pairs import extract_one_qa_pair

from hj_utils_language import split_text_by_dot_and_semicolon, get_date, save_qa_pairs_to_json
from hj_utils_llm import load_llm_model, load_llm_setting, infer_llm

INPUT_FILE_NAME = './data/raw/corpus_20231228_human.txt'  # None

def merge_and_deduplicate_keywords(list_list_keywords):
    list_keywords = []

    for sublist in list_list_keywords:
        for keyword in sublist:
            if keyword not in list_keywords:
                list_keywords.append(keyword)

    return list_keywords

def get_list_keywords():
    list_list_keywords = [['火球', '大治疗术', '硬化', '圣盾', '破甲', '嘲讽', '炎爆'], 
        ['团战世界', '游戏规则', '团队协同', '角色能力', '敌人击败', '治疗', '输出角色', '大量伤害', '防御', '承伤职业', '仇恨', 'BOSS战', '基础属性', '伤害计算', '技能释放', 'CD', '冷却时间', '单体伤害', '范围伤害', '单体恢复', '范围恢复', '敌对单位', '攻击优先级', '仇恨列表', 'BOSS攻击对象', '增加仇恨', '减少仇恨', '仇恨控制', '伤害预测', 'BOSS伤害对象', '仇恨累积速度', '普通伤害技能', '治疗技能', '高仇恨技能', 'BOSS攻击保护', '无法承受BOSS攻击'],
        ['团战世界', '游戏规则', '游戏胜负判定', '杂兵战', 'BOSS战', '我方人员全部被消灭', '战斗时间', '敌人强化', 'BOSS狂暴化', '生存', '高效输出', '职业倾向', '坦克倾向', '输出倾向', '治疗倾向', '辅助倾向', '角色特性', '防御类属性', '仇恨技能', '自保技能', '破甲', '嘲讽', '盾墙', '火球', '炎爆', '火焰冲击', '小恢复术', '大恢复术', '生命回流', '圣盾术', '硬化术', '回春图腾', '脆弱术', '战斗流程', '仇恨控制', 'CD管理', '输出优化', '治疗策略', '辅助技能'],
        ['团战世界', '游戏规则', '减伤', 'BUFF', '伤害减少', '治疗职业', '阵亡概率', '指挥', '大招', '坦克开减伤', 'AOE', '群体减伤', 'Rush', '固定流程', '切换目标', '敌人掉血', 'AOE技能', 'T', '主T', '副T', '吸引仇恨', '控制仇恨', '吃药', '生命值提升', 'OT', '仇恨失控', '嘲讽技能', '团灭', 'T拉回来', '奶妈抬血', '点名', '压一波血', '灌伤害'],
        [
    "团战世界", "游戏规则", "团队协同", "角色能力", "配合", "治疗角色", "输出角色", "承伤职业", "防御", 
    "仇恨", "BOSS战", "基础属性", "力量", "体力", "攻击力", "防御力", "BUFF", "技能释放", "技能类型", 
    "CD冷却", "单体伤害", "范围伤害", "单体恢复", "范围恢复", "集中火力输出", "攻击属性", "暴击属性", 
    "主动技能", "被动技能", "判定范围", "技能效果", "技能CD", "BOSS仇恨列表", "仇恨控制", "伤害预测", 
    "BOSS攻击对象", "伤害规避", "仇恨累积速度", "高仇恨技能", "保护同伴", "避免过快仇恨上升"
],
        [
    "团战世界", "游戏规则", "游戏胜负判定", "杂兵战", "BOSS战", "我方人员全部被消灭", "敌人强化", "BOSS狂暴化",
    "生存", "高效输出", "游戏职业倾向", "坦克倾向", "输出倾向", "治疗倾向", "辅助倾向", "角色特性", "防御类属性",
    "仇恨技能", "自保技能", "破甲技能", "嘲讽技能", "盾墙技能", "伤害技能", "火球技能", "炎爆技能", "火焰冲击技能",
    "小恢复术", "大恢复术", "生命回流技能", "圣盾术", "硬化术", "回春图腾", "脆弱术", "战斗流程", "BOSS攻击对象",
    "仇恨控制", "CD冷却", "伤害规避", "治疗策略", "辅助技能", "团队协作", "团队输出", "游戏机制"
],
        [
    "团战世界", "游戏规则", "减伤", "BUFF", "伤害减少", "治疗职业", "阵亡概率", "指挥", "大招", "AOE", "Rush",
    "固定流程", "目标切换", "敌人掉血", "攻击", "仇恨", "吸引仇恨", "控制仇恨", "吃药", "主T", "副T", "坦克", "OT",
    "团灭", "拉回来", "奶妈", "抬血", "点名", "极大威胁", "非常危险", "优先处理", "被点名", "远离人群", "压一波血",
    "灌伤害", "压血", "组合技能", "输出效率", "安全", "法力值上升"
],
        ]
    # print("keywords_list: ", list_list_keywords)

    list_keywords = merge_and_deduplicate_keywords(list_list_keywords)
    print("len(list_keywords): ", len(list_keywords))
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


def main():
    list_keywords = get_list_keywords()
    # print("list_keywords: ", list_keywords)

    list_corpus = get_list_corpus()
    # print("list_corpus: ", list_corpus)

    list_valid_keywords_sentences = get_list_valid_keywords_sentences(list_keywords, list_corpus)
    # print("list_valid_keywords_sentences: ", list_valid_keywords_sentences)

    model_path = "/mnt/nfs/zhangqi/zhangqi_nfs/DLM-project/public_models/modelWeights/vicuna-13b-v1.5"
    device = "cuda"
    model, tokenizer = load_llm_model(model_path=model_path, device=device)
    generate_stream_func, repetition_penalty, max_new_tokens, context_len, judge_sent_end = load_llm_setting(model_path, model)

    list_qa = []
    list_corpus_qa = []
    str_date = get_date()
    start_time = last_save_time = time.time()
    epoch = 0
    while time.time() - start_time < 60 * 60 * 24 * 3:
        epoch += 1
        for ketword_i in tqdm.tqdm(range(len(list_valid_keywords_sentences))):
            for sentence_j in range(len(list_valid_keywords_sentences[ketword_i][1])):
                str_prompt = "基于以下语料，请围绕关键词'" + list_valid_keywords_sentences[ketword_i][0] + "'尝试生成1个简洁精简的问题和回答，整理成问答格式，不要胡编乱造内容。语料：" + list_valid_keywords_sentences[ketword_i][1][sentence_j]
                print("str_prompt: ", str_prompt)
                outputs = infer_llm(model_path, device, model, tokenizer, generate_stream_func, repetition_penalty, max_new_tokens, context_len, judge_sent_end, str_prompt)

                qa_pair = extract_one_qa_pair(outputs)
                print("qa_pair: ", qa_pair)

                list_qa.append(qa_pair)
                list_corpus_qa.append(list_valid_keywords_sentences[ketword_i][1][sentence_j])

                if len(list_qa) == 3 \
                        or time.time() - last_save_time > 60 * 60 * 2 \
                        or (epoch==1 and ketword_i==len(list_valid_keywords_sentences)-1 and sentence_j==len(list_valid_keywords_sentences[ketword_i][1])):  # 60 * 60 * 2 60
                    assert len(list_qa)==len(list_corpus_qa)
                    str_data_num = "_dataNum" + str(len(list_qa))
                    output_file = './data/interim/data_vicuna_keyword' + '/data_vicuna_keyword' + str_date + str_data_num + '.json'
                    save_qa_pairs_to_json(list_qa, output_file, list_corpus_qa=list_corpus_qa)
                    last_save_time = time.time()


if __name__ == "__main__":
    main()
