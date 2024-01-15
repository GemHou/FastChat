from datetime import datetime
import json


def split_text_by_dot_and_semicolon(input_str):
    # 使用句号和分号进行切分
    split_chars = ['。', '；']

    # 替换分号，以便更容易切分
    input_str = input_str.replace('；', '。')

    # 根据句号切分文本
    sentences = input_str.split('。')

    # 去除空白项
    sentences = [s.strip() for s in sentences if s.strip()]

    return sentences

def get_date():
    # 获取当前时间
    now = datetime.now()
    # 提取月、日和小时
    month = now.month
    day = now.day
    hour = now.hour
    # 格式化成字符串，保证月、日、小时都是两位数
    str_date = f"_date{month:02d}{day:02d}{hour:02d}"
    return str_date

def save_qa_pairs_to_json(qa_pairs, json_file_path):
    data = []
    for idx, qa_pair in enumerate(qa_pairs):
        human_msg = {"from": "human", "value": qa_pair["question"]}
        gpt_msg = {"from": "gpt", "value": qa_pair["answer"]}
        conversation_data = {"id": f"identity_{idx}", "conversations": [human_msg, gpt_msg]}
        data.append(conversation_data)

    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=2)
