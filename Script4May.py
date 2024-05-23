# 1. 对已有结果进行记录，json保存：son里面几个字段：input prompt, output response, input token, output token（两个token数用client的返回结果），然后是处理时间用两个time.time相减
# 2. 下次运行时，如果json文件存在就不跑了
# 3. 增加一个缓冲机制：累计token到20k，暂停30s，避免token过多导致token limit
import re
import pandas as pd
import csv
from multiprocessing import Pool
from openai import AzureOpenAI
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
import time
import requests
import json
import os
from tqdm import tqdm
import sys


# 假设您已经设置了环境变量或者在代码中直接提供了API密钥和端点
nodes = [
    {
        "deployment_name": "LACopilot-GPT4Turbo0409-20-1st",
        "openai_api_key": "eebab092309841cb992862d1393ccf64",
        "openai_api_base": "https://spark1-techunit-swedencentral.openai.azure.com/openai/deployments/LACopilot-GPT4Turbo0409-20-3rd/chat/completions?api-version=2024-02-15-preview",
        "openai_api_version": "2024-02-15-preview"
    },
    {
        "deployment_name": "LACopilot-GPT4Turbo0409-20-2nd",
        "openai_api_key": "eebab092309841cb992862d1393ccf64",
        "openai_api_base": "https://spark1-techunit-swedencentral.openai.azure.com/openai/deployments/LACopilot-GPT4Turbo0409-20-3rd/chat/completions?api-version=2024-02-15-preview",
        "openai_api_version": "2024-02-15-preview"
    },
    {
        "deployment_name": "LACopilot-GPT4Turbo0409-20-3rd",
        "openai_api_key": "eebab092309841cb992862d1393ccf64",
        "openai_api_base": "https://spark1-techunit-swedencentral.openai.azure.com/openai/deployments/LACopilot-GPT4Turbo0409-20-3rd/chat/completions?api-version=2024-02-15-preview",
        "openai_api_version": "2024-02-15-preview"
    }
]

def read_csv(data_path):
    raw_data = []
    with open(data_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for idx, row in enumerate(reader):
            case_id = idx
            case_content = row['wechat_msgs'] + "\n" + row['callout_msgs'] + "\n" + row['group_wechat_msgs']
            human_label = int(row['t&c_demands']) if row['t&c_demands'].isdigit() else 0
            gpt_json_file_path = f"{case_id}.json"  # 假设json文件的命名规则是case_id.json
            if os.path.exists(gpt_json_file_path):
                with open(gpt_json_file_path, 'r') as json_file:
                    gpt_json = json.load(json_file)
            else:
                gpt_json = None
            raw_data.append([case_id, case_content, human_label, gpt_json])
    return raw_data


def process_entry(index, entry, working_prompt, nodes):
    case_id = entry[0]
    start_time = time.time()  # 开始计时
    content = entry[1]
    combined_text = working_prompt + " " + content

    # Check if the combined text exceeds 4000 tokens
    if len(combined_text.split()) > 4000:
        print(f"The combined text for case {case_id} exceeds 4000 tokens, skipping this line.")
        return entry

    # 轮换使用节点
    node = nodes[index % len(nodes)]
    client = AzureOpenAI(
        api_key=node["openai_api_key"],
        api_version=node["openai_api_version"],
        azure_endpoint=node["openai_api_base"]
    )

    print(f"Processing case {case_id} with node {node['deployment_name']}")  # 添加打印语句

    # 创建完成任务
    completion = client.chat.completions.create(
        model=node["deployment_name"],
        messages=[{"role": "user", "content": combined_text}]
    )

    # 将结果保存到json文件中
    result_data = {
        "input_prompt": combined_text,
        "output_response": completion.choices[0].message.content if completion.choices else None,
        "input_token": len(combined_text),
        "output_token": len(completion.choices[0].message.content) if completion.choices else 0,
        "processing_time": time.time() - start_time
    }

    # 检查响应时间是否超过30秒
    response_time = time.time() - start_time
    if response_time > 10:
        print(f"AI Model's response for csv[{case_id}] took too long: {response_time} seconds")
        gpt_label = 0
    elif completion.choices:
        message_content = completion.choices[0].message.content
        print(f"AI Model's response for csv[{case_id}]: {message_content}")
        matches = re.findall(r'[0-1]', message_content)
        gpt_label = int(matches[-1]) if matches else 0
    else:
        gpt_label = 0

    entry.append(gpt_label)
    entry.append(json.dumps(result_data))  # 将结果数据添加到entry中
    return entry

def generate_gpt_labels(raw_data, working_prompt, nodes):
    total_tokens = 0
    for index, entry in tqdm(enumerate(raw_data), total=len(raw_data), desc="Processing entries"):
        # 如果这一行的gpt_json字段已经有内容，跳过这一行的处理
        if len(entry) > 3 and entry[3]:  # 假设gpt_json字段是entry的第4个元素
            continue

        total_tokens += len(entry[1])
        # 检查累计的token数是否达到20k
        if total_tokens >= 20000:
            print(f"Total tokens reached 20k, sleeping for 30 seconds")  # 添加打印语句
            time.sleep(30)
            total_tokens = 0

    with Pool(len(nodes)) as pool:
        results = pool.starmap(process_entry,
                               [(index, entry, working_prompt, nodes) for index, entry in enumerate(raw_data) if len(entry) <= 3 or not entry[3]])  # 只处理gpt_json字段为空的行
    results.sort(key=lambda x: x[0])
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
    data_path = "Feb.csv"

    # 调用函数并打印结果
    data = read_csv(data_path)

    working_prompt = """
      你是一个资深内容质量检测专家，擅长检测文本中是否具有关于**教师/班型诉求**的关键信息。
     
    # 判断元素
    你需要在[chat]中检查是否包含**教师/班型诉求**信息，只要符合以下任意一条就表示[chat]中包含**教师/班型诉求**信息。
    - [换老师]：提出要求更换老师，不论是什么原因
    - [换班型]：因为难度问题要求换班。注意：因为时间安排要求换班的不算。
    
    # [换老师]和[换班型]的案例
    ## [换老师]的常见案例包括但不限于：
    - 您看看给在提高点、她觉得不难、不好玩
    - 她说可以难一点她试一试
    - 都有哪些老师，麻烦帮忙找位负责任点的
    - 小朋友说她想换回以前那个老师的课
    - 淘宝老师和西子老师两个人的综合评分哪一个更好一些
    - 咱们老师还有和小粉老师差不多的，就是点评小老师视频会把题再讲一遍的，小粉老师特别负责呢，就是发现问题要我们私下多训练，换个时间点就可以
    - 挑战班还有优秀老师没啊
    - 能否找个有经验的老师
    - 我还在想我们还有没有难度提升的班级让他挑战下
    
    ## [换班型]的常见案例包括但不限于：
    - 还是挑战班吧
    - 我觉得还是这个类型班级适合一些
    
    # 任务说明
    你的任务是：对提供文字对话内容进行检测，判断其是否含有关于“教师/班型诉求”的关键信息点，据此进行分类并按要求输出结果。
    if [chat]含有关于“教师/班型诉求”的关键信息点，你输出：1
    if [chat]不含有关于“教师/班型诉求”的关键信息点，你输出：0
     
    # 执行步骤
    Let's think step by step！
    1. 你需要牢固记住检测**教师/班型诉求**的[判断元素]。
    2. 你需要仔细阅读[chat]的文本信息。
    3. 你需要根据你记住的[判断元素]和[注意事项]从而判断[chat]中是否存在**教师/班型诉求**的相关信息。
    4. 你需要严格遵守[任务说明]和[你的输出格式]输出你仔细阅读和思考后的判断结果，要求该结果只能为一个`1`或`0`的整数。
     
    
    # 你的输出格式
    X (X 为0和1:你根据**判断元素**检测的[chat]中含有关于“教师/班型诉求”的关键信息点，如果包含“教师/班型诉求”则输出1，不含有“教师/班型诉求”的关键信息则输出为0)
    - 注意：1.不要漏掉**判断元素**维度，各维度的评判要严谨 2.返回的内容只能为1个0或者1的整数，不要输出其他分析
    - 你的输出只能是1个整数：1 or 0，不要输出文字分析过程和其他内容！
    - 你的输出只能是1个整数：1 or 0，不要输出文字分析过程和其他内容！
    - 你的输出只能是1个整数：1 or 0，不要输出文字分析过程和其他内容！
     
    # 以下内容是你需要认真检测的**[chat]**：
-------------------
    """

    # results = generate_gpt_labels(data, working_prompt, nodes)
    # df = pd.DataFrame(results, columns=['case_id', 'content', 'human_label', 'gpt_label', 'gpt_json'])
    # df.to_csv('Feb.csv', index=False)  # 将结果保存到csv文件中
    results = generate_gpt_labels(data, working_prompt, nodes)
    df = pd.DataFrame(results, columns=['case_id', 'content', 'human_label', 'gpt_label', 'gpt_json'])
    df.to_csv('Feb.csv', index=False)  # 将结果保存到csv文件中
    for row in results:
        print(f"case_id: {row[0]}, human_label: {row[2]}, gpt_label: {row[3]}")
    result_judge(results)