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