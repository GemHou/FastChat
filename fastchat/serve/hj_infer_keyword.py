def merge_and_deduplicate_keywords(list_list_keywords):
    list_keywords = []

    for sublist in list_list_keywords:
        for keyword in sublist:
            if keyword not in list_keywords:
                list_keywords.append(keyword)

    return list_keywords


def main():
    list_list_keywords = [['团战世界', '游戏规则', '团队协同', '角色能力', '敌人击败', '治疗', '输出角色', '大量伤害', '防御', '承伤职业', '仇恨', 'BOSS战', '基础属性', '伤害计算', '技能释放', 'CD', '冷却时间', '单体伤害', '范围伤害', '单体恢复', '范围恢复', '敌对单位', '攻击优先级', '仇恨列表', 'BOSS攻击对象', '增加仇恨', '减少仇恨', '仇恨控制', '伤害预测', 'BOSS伤害对象', '仇恨累积速度', '普通伤害技能', '治疗技能', '高仇恨技能', 'BOSS攻击保护', '无法承受BOSS攻击'],
        ['团战世界', '游戏规则', '游戏胜负判定', '杂兵战', 'BOSS战', '我方人员全部被消灭', '战斗时间', '敌人强化', 'BOSS狂暴化', '生存', '高效输出', '职业倾向', '坦克倾向', '输出倾向', '治疗倾向', '辅助倾向', '角色特性', '防御类属性', '仇恨技能', '自保技能', '破甲', '嘲讽', '盾墙', '火球', '炎爆', '火焰冲击', '小恢复术', '大恢复术', '生命回流', '圣盾术', '硬化术', '回春图腾', '脆弱术', '战斗流程', '仇恨控制', 'CD管理', '输出优化', '治疗策略', '辅助技能'],
        ['团战世界', '游戏规则', '减伤', 'BUFF', '伤害减少', '治疗职业', '阵亡概率', '指挥', '大招', '坦克开减伤', 'AOE', '群体减伤', 'Rush', '固定流程', '切换目标', '敌人掉血', 'AOE技能', 'T', '主T', '副T', '吸引仇恨', '控制仇恨', '吃药', '生命值提升', 'OT', '仇恨失控', '嘲讽技能', '团灭', 'T拉回来', '奶妈抬血', '点名', '压一波血', '灌伤害']]
    print("keywords_list: ", list_list_keywords)

    list_keywords = merge_and_deduplicate_keywords(list_list_keywords)
    print("list_keywords: ", list_keywords)

if __name__ == "__main__":
    main()
