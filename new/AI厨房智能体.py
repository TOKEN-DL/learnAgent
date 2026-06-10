
"""
通过langchain实现食谱推荐应用
用户拍摄自己家冰箱或者是厨房的食物照片，智能体自动识别图片中的食材，根据食材搜索相关食谱推荐给用户

功能实现
1.图片识别：上传图片识别其中的食材

2.智能搜索：根据识别的食材搜索相关的食谱

3.智能排序：按照营养价值、制作难度对食谱进行排序

4.创意建议：找不到合适食谱时，提供搭配建议

5.对话交互：对话交互,支持图片上传 + 文本对话

"""

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_classic.chains.hyde.prompts import web_search
from langchain_tavily import TavilySearch
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.agents import create_agent
from langchain.messages import HumanMessage
import sqlite3
import os

load_dotenv(verbose=True)

# 初始化模型
model = init_chat_model(
    model = "qwen3.5-plus",
    model_provider="openai",
    base_url = os.getenv("DASHSCOPE_BASE_URL"),
    api_key = os.getenv("DASHSCOPE_API_KEY"),
)

# 定义工具
web_search = TavilySearch(
    max_results=5,
    topic="general"
)

# 记忆管理
## 1.连接sqlite
connection = sqlite3.connect("resources/personal_chief.db", check_same_thread=False)
## 2.初始化
checkpointer = SqliteSaver(connection)
## 3.自动建表
checkpointer.setup()

# 自定义智能体
system_prompt = """
你是一名私人厨师。收到用户提供的食材照片或者清单后，请按以下流程操作：
1.识别和评估食材:若是用户提供照片，首先辨识所有可见食材。基于食材的外观状态，评估其新鲜度和用量，
整理出一份“当前可用食材清单”。
2.智能食谱检索：优先盗用web_search工具，以“可用食材清单”为核心关键词，查找可行的菜谱。
3.多维度评估与排序：从营养价值和制作难度两个维度对检索到的候选食谱进行量化打分，并根据得分排序，制作简单且营养丰富的排名靠前。
4.结构化方案输出：把排序后的食谱整理为一份结构清晰的建议报告，要包含食谱信息、得分、推荐理由、食谱的参考图片，帮助用户快速做出决策.

请严格按照流程，优先使用web_search工具检索食谱，搜索不到的情况下才能自己发挥。
"""

agent = create_agent(
    model=model,
    tools=[web_search],
    system_prompt=system_prompt,
    checkpointer=checkpointer,
)




# 测试
# 设置多模态消息
muti_messages = HumanMessage([
    {"type": "text", "text": "帮我看看可以做什么"},
    {"type": "image", "url": "https://tse3.mm.bing.net/th/id/OIP.BlQbOjiVbuxpaiGv643GAwHaE7?r=0&rs=1&pid=ImgDetMain&o=7&rm=3"},
])

config = {"configurable": {"thread_id":"1"}}

response = agent.invoke({"messages":[muti_messages]},config)

for message in response['messages']:
    message.pretty_print()











