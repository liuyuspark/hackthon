import re
import pandas as pd
import csv
from multiprocessing import Pool
from openai import AzureOpenAI
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

# API密钥和端点
nodes = [
    {
        "deployment_name": "TU-CRMAI-XGZM-EastUS2-GPT35-16k-1",
        "openai_api_key": "edc0a0ec751140048ff1edee9c81c91b",
        "openai_api_base": "https://tp-open-assistant-east-us-2.openai.azure.com/",
        "openai_api_version": "2023-07-01-preview"
    },
    {
        "deployment_name": "TU-CRMAI-XGZM-EastUS2-GPT35-16k-2",
        "openai_api_key": "edc0a0ec751140048ff1edee9c81c91b",
        "openai_api_base": "https://tp-open-assistant-east-us-2.openai.azure.com/",
        "openai_api_version": "2023-07-01-preview"
    },
    {
        "deployment_name": "TU-CRMAI-XGZM-EastUS2-GPT35-16k-3",
        "openai_api_key": "edc0a0ec751140048ff1edee9c81c91b",
        "openai_api_base": "https://tp-open-assistant-east-us-2.openai.azure.com/",
        "openai_api_version": "2023-07-01-preview"
    }
    ,
    {
        "deployment_name": "TU-internalchat-XGZM-JapEst-GPT35-16k",
        "openai_api_key": "6c42db311d084a03ae2ae72b40122bf0",
        "openai_api_base": "https://tu-internalchat-japest.openai.azure.com/",
        "openai_api_version": "2023-12-01-preview"
    }
]

def read_csv(data_path):
    raw_data = []
    with open(data_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for idx, row in enumerate(reader):
            case_id = idx
            # chat作为case_content
            case_content = row['chat']
            # communication_statu/parental_feedback 作为human_label，并转换为整数
            human_label = int(row['communication_statu']) if row['communication_statu'].isdigit() else -1
            raw_data.append([case_id, case_content, human_label])
    return raw_data

def process_entry(index, entry, working_prompt):
    content = entry[1]
    case_id = entry[0]
    combined_text = working_prompt + " " + content
    max_length = 1024 * 16
    if len(combined_text) > max_length:
        content = content[:max_length - len(working_prompt) - 1]

    # 轮换使用节点
    node = nodes[index % len(nodes)]
    client = AzureOpenAI(
        api_key=node["openai_api_key"],
        api_version=node["openai_api_version"],
        azure_endpoint=node["openai_api_base"]
    )

    # 创建完成任务
    completion = client.chat.completions.create(
        model=node["deployment_name"],
        messages=[{"role": "user", "content": combined_text}]
    )

    # 检查响应中是否有选择
    if completion.choices:
        message_content = completion.choices[0].message.content
        print(f"AI Model's response for csv[{case_id}]: {message_content}")
        matches = re.findall(r'[0-1]', message_content)
        gpt_label = int(matches[-1]) if matches else -1
    else:
        gpt_label = -1
    entry.append(gpt_label)
    return entry

def generate_gpt_labels(raw_data, working_prompt):
    with Pool(len(nodes)) as pool:
        results = pool.starmap(process_entry, [(index, entry, working_prompt) for index, entry in enumerate(raw_data)])
    # 根据case_id对结果进行排序
    results.sort(key=lambda x: x[0])
    # # 假设results是generate_gpt_labels的返回结果
    # for row in results:
    #     # 打印每个row的长度和内容，以检查是否有问题
    #     print(f"Length of row: {len(row)}, Content of row: {row}")
    #     if len(row) >= 4:
    #         print(f"case_id: {row[0]}, human_label: {row[2]}, gpt_label: {row[3]}")
    #     else:
    #         print(f"Error: The row does not contain enough elements. Row: {row}")
    return results


def result_judge(data):
    # 如果data不是DataFrame类型，将其转换为DataFrame
    if not isinstance(data, pd.DataFrame):
        data_df = pd.DataFrame(data, columns=['case_id', 'content', 'human_label', 'gpt_label'])
    else:
        data_df = data

    # 将NaN值替换为一个特定的标记（例如-1）
    data_df.fillna(0, inplace=True)

    # 显式转换列的数据类型以避免FutureWarning
    data_df['human_label'] = data_df['human_label'].astype(int)
    data_df['gpt_label'] = data_df['gpt_label'].astype(int)

    # 在计算accuracy和recall之前，确保预测和真实标签都是整数类型
    true_labels = data_df['human_label']
    predicted_labels = data_df['gpt_label']

    # 创建一个透视表，计算不同标签之间的数量
    pivot_table = pd.crosstab(true_labels, predicted_labels, rownames=['True'], colnames=['Predicted'], margins=True)
    print(f"Pivot Table:\n{pivot_table}\n")

    # 计算准确率
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Accuracy: {accuracy:.2f}")

    # 计算召回率
    recall = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)
    print(f"Recall: {recall:.2f}")

    # 计算混淆矩阵
    cm = confusion_matrix(true_labels, predicted_labels)
    print(f"Confusion Matrix:\n{cm}")

if __name__ == '__main__':
    data_path = "JClabel.csv"

    # 调用函数并打印结果
    data = read_csv(data_path)

    working_prompt = """
# 角色
你是一个资深内容质量检测专家，擅长检测文本中是否具有关于**学情反馈**的关键信息。

# **学情反馈**的定义
**学情反馈**是指la（la：辅导老师）与家长之间就学生的学习情况进行的交流。这种沟通可以采取多种形式，包括但不限于老师转述、观察表扬、专题反馈、单课次反馈、电话访谈和综合学情报告。

# 任务说明
你的任务是：对提供文字对话内容进行检测，判断其是否含有关于“学情反馈”的关键信息点，据此进行分类并按要求输出结果。
if [chat]含有关于“学情反馈”的关键信息点，你输出：1
if [chat]不含有关于“学情反馈”的关键信息点，你输出：0

# 执行步骤
Let's think step by step！
1. 你需要仔细思考**学情反馈**的相关定义。
2. 你需要牢固记住检测**学情反馈**的[判断元素]。
3. 你需要仔细阅读[chat]的文本信息。
4. 你需要根据你记住的[判断元素]和[注意事项]从而判断[chat]中是否存在**学情反馈**的相关信息。
5. 你需要严格遵守[任务说明]和[你的输出格式]输出你仔细阅读和思考后的判断结果，要求该结果只能为一个`1`或`0`的整数。

# 判断元素
你需要在[chat]中检查是否包含**学情反馈**信息。判断是否存在**学情反馈**，可以依据以下几个**判断元素**进行评估，只要符合其中任意一条就表示[chat]中包含**学情反馈**信息。
- [老师转述]: la（la：辅导老师）向家长传达学生在课堂上的表现、学习习惯和进步情况，包括但不限于学习效果、专注力、坚持性和课堂参与度。
- [观察表扬]: la（la：辅导老师）对学生的正面行为或学习态度给予肯定和鼓励，包括但不限于认真度、积极性、声音响亮和兴趣高。
- [专题反馈]: 对学生在特定学习主题或难点上的理解和掌握情况进行的详细说明，包括但不限于难点突破和专题自评。
- [单课次反馈]: 对学生在单次课程中的学习表现和理解情况的具体描述，包括但不限于课堂互动和学习目标。
- [电话访谈]: 教育机构通过电话与家长进行的一对一交流，讨论学生的学习进展和需要关注的问题，包括但不限于课后内容和学习难点。
- [综合学情报告]: 对学生在一段时间内的学习表现进行的总结，通常包含学生在不同模块或专题中的表现评估，以及建议和进步。

# 注意事项
除了遵循[判断元素]，你还需要通过[注意事项]判断**学情反馈**的信息。你的**注意事项**如下：
- 当[chat]存在对学生学习态度、行为或成绩的具体评价时，该[chat]被视为包含**学情反馈**。
- 当[chat]存在学习表现描述时，该[chat]被视为包含**学情反馈**。

# 你的输出格式
X (X 为0和1:你根据**判断元素**检测的[chat]中含有关于“学情反馈”的关键信息点则输出1，不含有“学情反馈”的关键信息则输出为0)
- 注意：1.不要漏掉**判断元素**维度，各维度的评判要严谨 2.返回的内容只能为1个0或者1的整数，不要输出其他分析
- 你的输出只能是1个整数：0 or 1，不要输出分析过程和其他内容

# 以下内容是你需要认真检测的**[chat]**：
"""
    results = generate_gpt_labels(data, working_prompt)
    for row in results:
        print(f"case_id: {row[0]}, human_label: {row[2]}, gpt_label: {row[3]}")
    result_judge(results)

