
from src.utils import pick_a_gpt_conn


def generate_logical_query(cur_date, original_query):
    return f"""
    你会得到一个关于数据查询的口语化表述，你需要将它转换为执行逻辑语句，明确说明选择的字段和筛选条件。

    你需要转化为：
    选择字段：字段1，字段2，字段3
    聚合操作符：distinct, count，count distinct, sum
    条件：条件1，条件2，条件3

    例如：
    假如在2024-04-20输入“过去7天里没有提交小老师视频的用户列表”，你需要转化为：
    选择字段：用户ID
    操作符：distinct
    条件：1. 提交时间在2024-04-13到2024-04-20之间；2. 未提交小老师视频

    今天的日期是{cur_date}
    你要处理的口语化表述是{original_query}

    仅输出logical_query，并以json格式输出
    {{'logical_query':str}}

    """
#     return  f"""
# 你会得到一个关于数据查询的口语化表述，你需要将它转换为执行逻辑语句，明确说明选择的字段和筛选条件。
#
# 你需要转化为：
# 选择字段：字段1，字段2，字段3
# 聚合操作符：distinct, count，count distinct, sum
# 条件：条件1，条件2，条件3
#
# 例如：
# 假如在2024-04-20输入“过去7天里没有提交小老师视频的用户列表”，你需要转化为：
# 选择字段：用户ID
# 操作符：distinct
# 条件：1. 提交时间在2024-04-13到2024-04-20之间；2. 未提交小老师视频
#
# 今天的日期是{cur_date}
# 你要处理的口语化表述是{original_query}
#
# 仅输出logical_query，并以json格式输出
# {{'logical_query':str}}
#
# """
#     return f"""
#     你会得到一个关于数据查询的口语化表述，你需要将它转换为执行逻辑语句，明确说明选择的字段和筛选条件。
#
#     你需要转化为：
#     选择字段：字段1，字段2，字段3
#     聚合操作符：distinct, count，count distinct, sum
#     条件：条件1，条件2，条件3
#
#     例如：
#     假如在2024-04-18输入“昨天有缺课的孩子”，你需要转化为：
#     选择字段：用户ID
#     操作符：distinct
#     条件：1. 提交时间在2024-04-17；2. 缺课
#
#     今天的日期是{cur_date}
#     你要处理的口语化表述是{original_query}
#
#     仅输出logical_query，并以json格式输出
#     {{'logical_query':str}}
# """
#     return f"""
#             你会得到一个关于数据查询的口语化表述，你需要将它转换为执行逻辑语句，明确说明选择的字段和筛选条件。
#
#             你需要转化为：
#             选择字段：字段1，字段2，字段3
#             聚合操作符：distinct, count，count distinct, sum
#             条件：条件1，条件2，条件3
#
#             例如：
#             假如在2024-04-18输入“过去14天小老师视频没上传的孩子和课次名称”，你需要转化为：
#             选择字段：用户ID；课次名称
#             操作符：distinct
#             条件：1. 提交时间在2024-04-12到2024-04-18之间；2. 视频没有上传；
#
#             今天的日期是{cur_date}
#             你要处理的口语化表述是{original_query}
#
#             仅输出logical_query，并以json格式输出
#             {{'logical_query':str}}
#     """

    # return f"""
    #             你会得到一个关于数据查询的口语化表述，你需要将它转换为执行逻辑语句，明确说明选择的字段和筛选条件。
    #
    #             你需要转化为：
    #             选择字段：字段1，字段2，字段3
    #             聚合操作符：distinct, count，count distinct, sum
    #             条件：条件1，条件2，条件3
    #
    #             例如：
    #             假如在2024-04-18输入“过去14天视频没上传的孩子和课次名称”，你需要转化为：
    #             选择字段：1.用户昵称；2.课次名称 3. 是否提交视频
    #             操作符：distinct
    #             条件：1. 提交时间在2024-04-05到2024-04-18之间；2. 视频没有上传；
    #
    #             今天的日期是{cur_date}
    #             你要处理的口语化表述是{original_query}
    #
    #             仅输出logical_query，并以json格式输出
    #             {{'logical_query':str}}
    # """

    # return f"""
    #             你会得到一个关于数据查询的口语化表述，你需要将它转换为执行逻辑语句，明确说明选择的字段和筛选条件。
    #
    #             你需要转化为：
    #             选择字段：字段1，字段2，字段3
    #             聚合操作符：distinct, count，count distinct, sum
    #             条件：条件1，条件2，条件3
    #
    #             例如：
    #             假如在2024-04-18输入“过去14天视频没上传的孩子和课次名称”，你需要转化为：
    #             选择字段：1.用户昵称；2.课次名称 3. 是否提交视频
    #             操作符：distinct
    #             条件：1. 提交时间在2024-04-05到2024-04-18之间；2. 视频没有上传；
    #
    #             今天的日期是{cur_date}
    #             你要处理的口语化表述是{original_query}
    #
    #             仅输出logical_query，并以json格式输出
    #             {{'logical_query':str}}
    # """


def generate_pandas_query(logical_query: str) -> str:
    # return f"""
    # 你有一个pandas的data frame，columns是
    # user_id: int, 用户ID
    # nickname: str, 用户昵称
    # open_dt: date, 开课时间
    # is_homework_done: bool, 是否上传作业
    # is_lesson_join: bool, 是否参加课程
    #
    # 你有一个查询逻辑，请将它转化为pandas的查询语句
    # 你的查询逻辑是：{logical_query}
    #
    # 请注意只输出用户昵称，且做排重
    #
    # 例如，
    # 输入的logical query是'选择字段：用户ID 操作符: distinct 条件：1. 提交时间在2024-04-17；2. 未提交小老师视频'
    # 输出的结果是'df.loc[(df['open_dt'] == '2024-04-17') & (~df['is_homework_done']), 'user_name'].agg({{'column': pd.Series.unique}})'
    #
    # 只输出进行查询的pandas语句，不要提供全部代码上下文。
    # 你的输出格式是:
    # {{"pandas_operator":str}}
    # """
    # return f"""
    # 你有一个pandas的data frame，columns是
    # user_id: int, 用户ID
    # nickname: str, 用户昵称
    # open_dt: date, 开课时间
    # is_lesson_join: bool, 是否参加课程
    #
    # 你有一个查询逻辑，请将它转化为pandas的查询语句
    # 你的查询逻辑是：{logical_query}
    #
    # 请注意只输出用户昵称，且做排重
    #
    # 例如，
    # 输入的logical query是'选择字段：用户ID 操作符: distinct 条件：1. 提交时间在2024-04-17；2. 未提交小老师视频'
    # 输出的结果是'df.loc[(df['open_dt'] == '2024-04-17') & (~df['is_lesson_join']), 'user_name'].agg({{'column': pd.Series.unique}})'
    #
    # 只输出进行查询的pandas语句，不要提供全部代码上下文。
    # 你的输出格式是:
    # {{"pandas_operator":str}}
    # # """
    # return f"""
    # 你有一个pandas的data frame，columns是
    # user_id: int, 用户ID
    # nickname: str, 用户昵称
    # open_dt: date, 开课时间
    # lesson_name: str, 课次名称
    # is_video_done: bool, 是否提交视频
    #
    # 你有一个查询逻辑，请将它转化为pandas的查询语句
    # 你的查询逻辑是：{logical_query}
    #
    # 请注意只输出用户昵称和课次名称
    #
    # 例如，
    # 输入的logical query是'选择字段：用户昵称；课次名称；操作符: distinct 条件：1. 提交时间在2024-04-05到2024-04-18之间；2. 未提交视频'
    # 输出的结果是'df.loc[(df['open_dt'] >= '2024-04-05') & (df['open_dt'] <= '2024-04-18') & (~df['is_homework_done']), 'user_name', 'lesson_name']}})'
    #
    # 只输出进行查询的pandas语句，不要提供全部代码上下文。
    # 你的输出格式是:
    # {{"pandas_operator":str}}
    # """
    # return f"""
    #
    # 你有一个pandas的data frame，columns是
    # user_id: int, 用户ID
    # nickname: str, 用户昵称
    # open_dt: date, 开课时间
    # is_homework_done: bool, 是否上传作业
    # grade: int, Level
    #
    # 你有一个查询逻辑，请将它转化为pandas的查询语句
    # 你的查询逻辑是：{logical_query}
    #
    # 请注意输出用户昵称和课次名称
    #
    # 例如，
    # 输入的logical query是'选择字段：用户昵称；课次名称；操作符: distinct 条件：1. 提交时间在2024-04-08到2024-04-14之间；2. 未提交作业；3. L5'
    # 输出的结果是'df.loc[(df['open_dt'] >= '2024-04-08') & (df['open_dt'] <= '2024-04-14') & (~df['is_homework_done'] & (df['grade'] == '5'), 'user_name' & 'lesson_name']}})'
    #
    # 只输出进行查询的pandas语句，不要提供全部代码上下文。
    # 你的输出格式是:
    # {{"pandas_operator":str}}
    # """
    return f"""
    你有一个pandas的data frame，columns是
    user_id: int, 用户ID
    nickname: str, 用户昵称
    open_dt: date, 开课时间
    is_homework_done: bool, 是否上传作业
    grade: int, Level
    
    你有一个查询逻辑，请将它转化为pandas的查询语句
    你的查询逻辑是：{logical_query}

    请注意输出用户昵称和课次名称

    例如，
    输入的logical query是'选择字段：用户昵称；课次名称；操作符: distinct 条件：1. 提交时间在2024-04-08到2024-04-14之间；2. 未提交作业；3. L5'
    输出的结果是'df.loc[(df['open_dt'] >= '2024-04-08') & (df['open_dt'] <= '2024-04-14') & (~df['is_homework_done'] & (df['grade'] == '5'), 'user_name' & 'lesson_name']}})'

    只输出进行查询的pandas语句，不要提供全部代码上下文。
    你的输出格式是:
    {{"pandas_operator":str}}
    """

if __name__ == "__main__":
    client = pick_a_gpt_conn("la_communication", 'gpt_4_turbo_128k')
    logical_query = client.chat_completion_in_json(generate_logical_query('2024-04-19','过去7天用户未提交作业的次数'))['logical_query']
    # logical_query = '选择字段：用户ID，未提交作业次数 聚合操作符：count 条件：1. 计算时间区间在2024-04-12到2024-04-19之间；2. 未提交作业'
    # logical_query = '选择字段：用户ID，缺课 聚合操作符：count 条件：1. 计算时间区间在2024-04-17；2. 缺课'
    # logical_query = '选择字段：用户昵称，课次名称，未提交作业，grade，聚合操作符：count 条件：1. 在2024-04-08到2024-04-14之间；2. 未提交作业；3. L5'

    print(logical_query)

    pandas_operator = client.chat_completion_in_json(generate_pandas_query(logical_query))['pandas_operator']
    print(pandas_operator)
    import os
    import pandas as pd
    from src.constants import DATA_DIR
    df = pd.read_csv(os.path.join(DATA_DIR, 'src','la copilot demo_1.csv'))
    df['open_dt'] = pd.to_datetime(df['open_dt'])
    df['open_dt'] = df['open_dt'].dt.strftime('%Y-%m-%d')
    res_df = eval(pandas_operator)
    print(res_df)