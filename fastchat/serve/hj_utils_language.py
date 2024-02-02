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

def save_qa_pairs_to_json(qa_pairs, json_file_path, list_corpus_qa=None):
    data = []
    for idx, qa_pair in enumerate(qa_pairs):
        human_msg = {"from": "human", "value": qa_pair["question"]}
        gpt_msg = {"from": "gpt", "value": qa_pair["answer"]}
        if list_corpus_qa is None:
            conversation_data = {"id": f"identity_{idx}", 
                                "conversations": [human_msg, gpt_msg]
                                }
        else:
            conversation_data = {"id": f"identity_{idx}", 
                                "conversations": [human_msg, gpt_msg],
                                "corpus": list_corpus_qa[idx]
                                }
        data.append(conversation_data)

    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=2)

def load_qa_pairs_from_json(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    qa_pairs = []
    for conv_data in data:
        human_msg = next(msg["value"] for msg in conv_data["conversations"] if msg["from"] == "human")
        gpt_msg = next(msg["value"] for msg in conv_data["conversations"] if msg["from"] == "gpt")
        corpus_qa = conv_data.get("corpus", None)

        qa_pairs.append({
            "question": human_msg,
            "answer": gpt_msg,
            "corpus": corpus_qa
        })

    return qa_pairs
