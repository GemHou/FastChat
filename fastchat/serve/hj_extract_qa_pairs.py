import re


def remove_prefix(input_str, prefixes):
    for prefix in prefixes:
        if input_str.startswith(prefix):
            return input_str[len(prefix):]

    return input_str


def extract_qa_pairs(text_list):
    qa_pairs = []
    current_qa_pair = {"question": "", "answer": ""}

    for text in text_list:
        # 使用正则表达式匹配问答对
        match_1 = re.match(r'(.+)(\n|\n\n)(.+)', text)
        # print("match_1: ", match_1)
        if match_1 is not None:
            match = match_1
        else:
            match_2 = re.match(r'(问|问题|Q|)：(.+)(答|回答|A)：(.+)', text)

            if match_2 is not None:

                match = match_2
            else:
                print("Fail text: ", text)
                continue
        if match:
            # 将匹配的部分提取为问答对
            question = match.group(1).strip()
            question = remove_prefix(question, ["问：", "问题：", "Q：", "1. "])
            answer = match.group(3).strip()
            answer = remove_prefix(answer, ["答：", "回答：", "A："])

            # 如果当前问答对不为空，添加到列表中
            if current_qa_pair["question"] and current_qa_pair["answer"]:
                qa_pairs.append(current_qa_pair)

            # 更新当前问答对
            current_qa_pair = {"question": question, "answer": answer}
        else:
            # 如果没有匹配到问答对，将文本追加到当前答案中
            current_qa_pair["answer"] += "\n\n" + text.strip()

    # 添加最后一个问答对
    if current_qa_pair["question"] and current_qa_pair["answer"]:
        qa_pairs.append(current_qa_pair)

    return qa_pairs


if __name__ == "__main__":
    qa_pairs = extract_qa_pairs([
        '问：在游戏中，角色移动的场景有哪些？\n\n答：在游戏中，角色移动的场景主要包括规避伤害和到达指定地点执行战术两种情况。例如，当BOSS释放一个具有高威胁的大范围伤害技能时，角色需要走到安全位置，等待伤害技能结束，避免受到大量伤害，然后回到输出位置进行攻击；同样，在BOSS战中，当BOSS触发了一些机制时，角色需要移动到指定机关旁，与机关交互，才能继续正常攻略BOSS。',
        '问题：在游戏中，如何提高角色的存活几率？\n\n回答：在游戏中，当坦克倾向的角色生命垂危，治疗角色还在治疗其他人时，D会对坦克角色释放硬化术，为坦克角色提供更多的减伤能力，增加存活几率。这种硬化术可以降低角色受到的伤害，增加角色的耐久性，从而提高角色的存活几率。',
        '问：在游戏中，如何提升团队输出？\n\n回答：在游戏中，当全队开始对BOSS进行输出的时候，D会对BOSS释放脆弱术，使得全团的成员在攻击BOSS时候获得更大的收益，提升团队输出。这种脆弱术可以减少BOSS的防御力，使得全队的攻击更容易成功，从而提高团队的输出效率。'
    ])
    print("qa_pairs: ", qa_pairs)
