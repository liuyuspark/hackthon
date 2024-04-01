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


# def extract_category(response_text):
#     # 假设response_text是一个JSON字符串，包含category字段
#     # 这里使用了json.loads来解析字符串，并尝试获取category字段
#     import json
#     try:
#         response_json = json.loads(response_text)
#         return response_json.get('category', None)
#     except json.JSONDecodeError:
#         # 如果response_text不是有效的JSON，这里会捕获异常
#         return None

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
            # 读取AI 模型的响应中的最后一个0-2之间的数字作为gpt_label
            # matches = re.findall(r'[0-2]', message_content)
            # if matches:
            #     gpt_label = int(matches[-1])
            #     # 将gpt_label存入raw_data中
            #     entry.append(gpt_label)
            # 读取AI 模型的响应中category中0-2的字段作为gpt_label
            # matches = re.findall(r'category\": (\d)', message_content)
            matches = re.findall(r'category\": ([0-2])', message_content)
            if matches:
                gpt_label = int(matches[0])
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
    # 如果data不是DataFrame类型，将其转换为DataFrame
    if not isinstance(data, pd.DataFrame):
        data_df = pd.DataFrame(data)
    else:
        data_df = data

    # 将NaN值替换为一个特定的标记（例如-1）
    data_df.fillna(-1, inplace=True)

    # 显式转换列的数据类型以避免FutureWarning
    # 应该在调用 astype(str) 之前将列转换为字符串类型
    data_df.iloc[:, 2] = data_df.iloc[:, 2].astype(int)
    # data_df.iloc[:, 3] = data_df.iloc[:, 3].astype(str)
    data_df.iloc[:, 3] = data_df.iloc[:, 3].astype(int)

    # 在计算accuracy和recall之前，确保预测和真实标签都是字符串类型
    # 这样可以避免在比较时出现数据类型不匹配的问题
    true_labels = data_df.iloc[:, 2].astype(int)
    predicted_labels = data_df.iloc[:, 3].astype(int)

    # 创建一个透视表，计算不同标签之间的数量
    pivot_table = data_df.pivot_table(index=true_labels, columns=predicted_labels, aggfunc='size', fill_value=0)
    print(pivot_table)

    # 计算准确率
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Accuracy: {accuracy}")

    # 计算召回率
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    print(f"Recall: {recall}")

    # 计算混淆矩阵
    cm = confusion_matrix(true_labels, predicted_labels)
    print(f"Confusion Matrix:\n{cm}")


# LLM Classifier
# For row in raw_data:
# gpt_label = generate_gpt_label(row[1])
# def LLM_Classifier(raw_data):
#     for row in raw_data:
#         gpt_label = generate_gpt_label(row[1])
#         raw_data.append(gpt_label)
    # # 处理完后的结构是 [[case_id, case_content, human_label, gpt_label],]


# 指定文件路径
data_path = "huohua_data_0308//CategoryOnly//valid_sop905_0207.jsonl"

# 调用函数并打印结果
data = read_jsonl(data_path)
# print(data)
# print(data[2])
# working_prompt = "\n你是一个复读机\n你的任务是对提供的文字内容进行复制和粘贴，最后一模一样输出我提供的文字内容：\n"
# working_prompt = "\n你是一个资深的内容质量检测专家，擅长检测文本中是否具有关于“补差”的关键信息。“补差”是用户在完成第一节课后的30天内，有资格以比第一次购买课包更优惠的价格升级到大课包，补中间的差价获得相应的课时和权益。\n\n你的任务\n对提供的文字内容进行检测，判断该内容中是否含有关于课程升级补差的关键信息点，据此进行分类并按要求输出结果。\n\n你需要检测3个关键信息点：\n1. 补差权益：文字中提到补差【3900】元就可以获得【48】节直播课时，【4】次请假机会，【2】节互动自学课，【1】次休学机会，【30】次周分享机会（可兑换6节直播课课时）等含义相近的关于价格优惠和购买内容的具体信息，例如，便宜了2000/800块钱。\n2. 权益到期：文字中提到补差优惠的有效期限、付款截止日期、权益失效的时间等，注意二维码过期不代表补差权益到期，例如，今天/还有几天就过期了，今天是补差最后一天了，马上就到期/过期/失效了，还有XX个小时就截止了，截止到23:59。\n3. 已经补差：表示用户已经付款、课时和权益已经充值到用户账户的描述，注意下了订单不代表已经付款、支付成功跟我说/截图给我不代表已经付款，例如，课时已经到账了、已经登记上了、订单发票、抽奖链接：https://m.huohua.cn/activity/...。\n\n你的思考过程\n先找到文字中的关键信息，再进行分类。\n\n你的输出\n如果文字中没有包含上面任何1个信息，请输出0；\n如果文字中只包含<补差权益>或者<权益到期>其中1个信息，请输出1；\n如果文字中包含全部<补差权益>和<权益到期>的2个信息，请输出2；\n如果文字中包含<已经补差>的信息，请输出2。\n\n注意：只输出1个数字，不需要其他说明。\n注意：只输出1个数字，不需要其他说明。\n注意：只输出1个数字，不需要其他说明。\n"
# working_prompt = "# 你是一个资深的内容质量检测专家，擅长检测文本中是否具有关于“补差”的关键信息。\n# “补差”是指用户在完成第一节课后的30天内，有资格以比第一次购买课包更优惠的价格升级到大课包，补中间的差价获得相应的课时和权益。\n# 你的任务\n# 对提供的文字内容进行检测，判断该内容中是否含有关于课程升级补差的关键信息点，据此进行分类并按要求输出结果。\n# 你需要检测3个关键信息点：\n# ## 1. 补差权益\n# ### 1.1 价格优惠\n# 正则表达式：\n# - 优惠(.*?)(元|块钱)\n# - 便宜(.*?)(元|块钱)\n# - 立减(.*?)(元|块钱)\n# - 折扣(.*?)(折)\n# 语义分析：\n# - 价格对比：原价、现价、优惠后价格\n# - 优惠力度：百分比、金额\n# ### 1.2 购买内容\n# 正则表达式：\n# - 直播课时(.*?)节\n# - 请假机会(.*?)次\n# - 互动自学课(.*?)节\n# - 休学机会(.*?)次\n# - 周分享机会(.*?)次\n# 语义分析：\n# - 课时类型：直播课、互动自学课\n# - 其他权益：请假机会、休学机会、周分享机会\n# ## 2. 权益到期\n# 正则表达式：\n# - 有效期限(.*?)\n# - 付款截止日期(.*?)\n# - 权益失效时间(.*?)\n# 语义分析：\n# - 时间表达：今天、还有几天、最后一天、马上、到期、过期、失效、截止\n# ## 3. 已经补差\n# 正则表达式：\n# - 课时已经到账\n# - 已经登记上了\n# - 订单发票\n# 语义分析：\n# - 支付状态：已付款、支付成功\n# - 补差结果：课时到账、订单完成\n# # 你的思考过程\n# 1. 使用正则表达式或CRF等技术提取关键词。\n# 2. 根据关键词的权重和语义上下文，进行信息分类。\n# 3. 输出详细的信息分类结果。\n# # 你的输出\n# 0：未包含任何关键信息\n# 1：包含<补差权益>或<权益到期>\n#     * 11：仅包含<补差权益>\n#     * 12：仅包含<权益到期>\n# 2：包含<补差权益>和<权益到期>\n# 3：包含<已经补差>\n"
# working_prompt = """
# \n你是一个资深的内容质量检测专家，擅长检测文本中是否具有关于“补差”的关键信息。“补差”是用户在完成第一节课后的30天内，有资格以比第一次购买课包更优惠的价格升级到大课包，补中间的差价获得相应的课时和权益。\n\n你的任务\n对提供的文字内容进行检测，判断该内容中是否含有关于课程升级补差的关键信息点，据此进行分类并按要求输出结果。\n\n你需要检测3个关键信息点：\n1. 补差权益：文字中提到补差【3900】元就可以获得【48】节直播课时，【4】次请假机会，【2】节互动自学课，【1】次休学机会，【30】次周分享机会（可兑换6节直播课课时）等含义相近的关于价格优惠和购买内容的具体信息，例如，便宜了2000/800块钱。\n2. 权益到期：文字中提到补差优惠的有效期限、付款截止日期、权益失效的时间等，注意二维码过期不代表补差权益到期，例如，今天/还有几天就过期了，今天是补差最后一天了，马上就到期/过期/失效了，还有XX个小时就截止了，截止到23:59。\n3. 已经补差：表示用户已经付款、课时和权益已经充值到用户账户的描述，注意下了订单不代表已经付款、支付成功跟我说/截图给我不代表已经付款，例如，课时已经到账了、已经登记上了、订单发票、抽奖链接：https://m.huohua.cn/activity/...。\n\n你的思考过程\n先找到文字中的关键信息，再进行分类。\n\n你的输出\n如果文字中没有包含上面任何1个信息，请输出0；\n如果文字中只包含<补差权益>或者<权益到期>其中1个信息，请输出1；\n如果文字中包含全部<补差权益>和<权益到期>的2个信息，请输出2；\n如果文字中包含<已经补差>的信息，请输出2。\n\n**Note:** 1. provide the reason for the judgment in JSON format. 2. you must also output a numeric result of 0-2 as your end based on the above reason for the judgment! you must also output a numeric result of 0-2 as your end based on the above reason for the judgment! you must also output a numeric result of 0-2 as your end based on the above reason for the judgment!\n
# """
working_prompt = """
你是一个资深的内容质量检测专家，擅长检测文本中是否具有关于“补差”的关键信息。“补差”是用户在完成第一节课后的30天内，有资格以比第一次购买课包更优惠的价格升级到大课包，补中间的差价获得相应的课时和权益。\n\n你的任务\n对提供的文字内容进行检测，判断该内容中是否含有关于课程升级补差的关键信息点，据此进行分类并按要求输出结果。\n\n你需要检测3个关键信息点：\n1. 补差权益：文字中提到补差【3900】元就可以获得【48】节直播课时，【4】次请假机会，【2】节互动自学课，【1】次休学机会，【30】次周分享机会（可兑换6节直播课课时）等含义相近的关于价格优惠和购买内容的具体信息，例如，便宜了2000/800块钱。\n2. 权益到期：文字中提到补差优惠的有效期限、付款截止日期、权益失效的时间等，注意二维码过期不代表补差权益到期，例如，今天/还有几天就过期了，今天是补差最后一天了，马上就到期/过期/失效了，还有XX个小时就截止了，截止到23:59。\n3. 已经补差：表示用户已经付款、课时和权益已经充值到用户账户的描述，例如，课时已经到账了、已经登记上了、在扫码/截图之后还有感谢您的信任/支持之类的内容、订单发票、抽奖链接：https://m.huohua.cn/activity/...。注意当仅出现下了订单/支付成功跟我说/截图给我/扫二维码/课时即时到账时不代表已经付款。\n\n你的思考过程\n先找到文字中的关键信息，再进行分类。\n\n你的输出\n如果文字中没有包含上面任何1个信息，请输出{\"category\": 0, \"key_points\": \"\"}；\n如果文字中只包含<补差权益>或者<权益到期>其中1个信息，请输出{\"category\": 1, \"key_points\": 提取文字中的具体关键信息}；\n如果文字中包含全部<补差权益>和<权益到期>的2个信息，请输出{\"category\": 2, \"key_points\": 提取2个文字中的关键信息，用1、2排序}；\n如果文字中包含<已经补差>的信息，请输出{\"category\": 2, \"key_points\": 提取文字中表示已经付款的关键信息}。\n\n注意：只输出json\n"
"""
generate_gpt_label(data, working_prompt)
for row in data:
    print(f"case_id: {row[0]}, human_label: {row[2]}, gpt_label: {row[3]}")
result_judge(data)
# client35 = AzureOpenAI(
#     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#     api_version="2023-12-01-preview",
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
# )
# with codecs.open(data_path, encoding='utf-8-sig') as f:
#     json_dataset = [json.loads(line) for line in f]
#
# for index, entry in enumerate(json_dataset):
#     # Loop through each item in json_dataset and send to AzureOpenAI
#         messages = entry["messages"]
#
#         # Create the completion
#         completion = client35.chat.completions.create(
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
#             # 读取AI 模型的响应中的第最后一个0-2之间的数字作为gpt_label
#             matches = re.findall(r'[0-2]', message_content)
#             if matches:
#                 gpt_label = int(matches[-1])
#             else:
#                 gpt_label = None
#             # 将gpt_label存入data中
#             for row in data:
#                 row.append(gpt_label)
#         else:
#             # If there are no choices, output an error message
#             print(f"Error: The AI model did not return any response for json_dataset[{index}].")
# # print(data[2])

# print(data[0])