import os
import re
import json
import codecs
import pandas as pd
from openai import AzureOpenAI
# Set environment variables
import sklearn
os.environ["AZURE_OPENAI_API_KEY"] = "6c42db311d084a03ae2ae72b40122bf0"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://tu-internalchat-japest.openai.azure.com/"
# 整体处理脚本
# 数据读取
# 输入jsonl文件，输出如下结构raw_data = [[case_id, case_content, human_label],]
# 其中，case id是jsonl的行数，case content是chat history中user content，
# human label是chat history中assistant content中category的field
import json
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix


def read_jsonl(data_path):
    raw_data = []
    with open(data_path, 'r', encoding='utf-8') as file:
        for idx, line in enumerate(file):
            entry = json.loads(line)
            case_id = idx
            # message是一个list，里面每个元素是一个dict，包含了content和role两个field
            case_content = next((message['content'] for message in entry['messages'] if message['role'] == 'user'), None)
            # human label是chat history中assistant content中category的field
            human_label = next((message['content'] for message in entry['messages'] if message['role'] == 'assistant'), None)
            # # 提取 'assistant' 角色的 'content'
            # assistant_content = next((message['content'] for message in entry['messages'] if message['role'] == 'assistant'), None)
            # # 提取  assistant_content 中的category 字段作为human_label.
            # # assistant_content举例如下：{"role": "assistant", "content": "{'category': 2, 'key_points': ''}"}
            # # Fix the JSON format by replacing single quotes with double quotes and escaping double quotes inside the string
            # assistant_content = assistant_content.replace("'", "\"") # 将单引号替换为双引号
            # # assistant_content = assistant_content.replace('\"{', '{')
            # # assistant_content = assistant_content.replace('}\"', '}')
            # # Parse the JSON string
            # assistant_content = json.loads(assistant_content)
            # human_label = assistant_content['category']
            # raw_data = [[case_id, case_content, human_label], ]
            raw_data.append([case_id, case_content, human_label])
    return raw_data

# def call_gpt_model(data_path):
#     # Read the dataset from the provided data path
#     with codecs.open(data_path, encoding='utf-8-sig') as f:
#         json_dataset = [json.loads(line) for line in f]
#
#     # Set Azure OpenAI client
#     client = AzureOpenAI(
#         api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#         api_version="2023-12-01-preview",
#         azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
#     )
#
#     # Loop through each item in json_dataset and send to AzureOpenAI
#     for index, data in enumerate(json_dataset):
#         messages = data["messages"]
#
#         # Create the completion
#         completion = client.chat.completions.create(
#             model="TU-internalchat-XGZM-JapEst-GPT35-16k",
#             messages=messages
#         )
#
#         # Check if there are any choices in the response
#         if completion.choices:
#             # Extract the message content
#             message_content = completion.choices[0].message.content
#             # Output the AI model's response
#             print(f"AI Model's response for json_dataset[{index}]: {message_content}")


def extract_category(response_text):
    # 假设response_text是一个JSON字符串，包含category字段
    # 这里使用了json.loads来解析字符串，并尝试获取category字段
    import json
    try:
        response_json = json.loads(response_text)
        return response_json.get('category', None)
    except json.JSONDecodeError:
        # 如果response_text不是有效的JSON，这里会捕获异常
        return None

def generate_gpt_label(raw_data, working_prompt):
    # 第一步：处理内容和提示是否超长
    for entry in raw_data:
        # 只读取case_content
        content = entry[1]
        case_id = entry[0]
        # print(content)
        # 将working_prompt和content合并，确保不超过GPT模型的最大输入长度
        combined_text = working_prompt + " " + content
        max_length = 1024 * 16
        if len(combined_text) > max_length:
            content = content[:max_length - len(working_prompt) - 1]  # 截断content
        # 第二步：调用OpenAI Azure SDK完成生成
        # Set Azure OpenAI client
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2023-12-01-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )

        # Create the completion
        completion = client.chat.completions.create(
            model="TU-internalchat-XGZM-JapEst-GPT35-16k",
            messages=[{"role": "user", "content": combined_text}]
        )
        # Check if there are any choices in the response
        if completion.choices:
            # Extract the message content
            message_content = completion.choices[0].message.content
            # Output the AI model's response
            print(f"AI Model's response for jsonl[{case_id}]: {message_content}")
            # 读取AI 模型的响应中的第一个0-2之间的数字作为gpt_label
            match = re.search(r'[0-2]', message_content)
            if match:
                gpt_label = int(match.group(0))
                # 将gpt_label存入raw_data中
                entry.append(gpt_label)
            else:
                entry.append(None)

    return raw_data

# def extract_category(response_text):
#     try:
#         # 假设response_text的格式是JSON，包含一个名为"category"的字段
#         response_json = json.loads(response_text)
#         category = response_json.get("category")
#         if category is not None:
#             return category
#         else:
#             print("No 'category' found in response.")
#             return None
#     except json.JSONDecodeError:
#         print("Error decoding response JSON.")
#         return None


# 结果判断
# data_df = pd.Dataframe(raw_data,..)
# 用pivot table或者sklearn或者confusion matrix
def result_judge(data):
    if not isinstance(data, pd.DataFrame):
        data_df = pd.DataFrame(data)
    else:
        data_df = data

    # 删除包含NaN的行
    data_df = data_df.dropna()

    pivot_table = data_df.pivot_table(index=data_df.columns[2], columns=data_df.columns[3], aggfunc='size', fill_value=0)
    print(pivot_table)

    accuracy = accuracy_score(data_df.iloc[:, 2], data_df.iloc[:, 3])
    print(f"Accuracy: {accuracy}")

    recall = recall_score(data_df.iloc[:, 2], data_df.iloc[:, 3], average='weighted')
    print(f"Recall: {recall}")

    cm = confusion_matrix(data_df.iloc[:, 2], data_df.iloc[:, 3])
    print(f"Confusion Matrix:\n{cm}")



# LLM Classifier
# For row in raw_data:
# gpt_label = generate_gpt_label(row[1])
def LLM_Classifier(raw_data):
    for row in raw_data:
        gpt_label = generate_gpt_label(row[1])
        raw_data.append(gpt_label)
    # # 处理完后的结构是 [[case_id, case_content, human_label, gpt_label],]


# 指定文件路径
data_path = "huohua_data_0308//CategoryOnly//test_sop905_0207.jsonl"

# 调用函数并打印结果
data = read_jsonl(data_path)
# print(data)
# print(data[2])
working_prompt = "作为内容质量检测专家，你的任务是检测文本中的“补差”关键信息，并仅输出相应的数字。\n<输出规则：>\n- 无信息点，输出0。\n- 仅含1个信息点，输出1。\n- 含有补差权益和权益到期，或已经补差，输出2。\n\n\n关键信息点：\n -  补差权益：提及补差金额和获得的权益。\n - 权益到期：提及补差优惠的有效期限。\n - 已经补差：用户已付款，权益已充值。\n\n<**重要：仅输出数字！！！不要包含任何其他说明或文字！！！！！！！！！！！！**>"
generate_gpt_label(data, working_prompt)
# # print(data[2])
# 输出data_path的case_id,human_label, gpt_label
for row in data:
    print(f"case_id: {row[0]}, human_label: {row[2]}, gpt_label: {row[3]}")
result_judge(data)
# print(data[0])