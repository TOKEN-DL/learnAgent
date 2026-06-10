

# 利用postgres实现 langgraph-checkpoint-postgres
from langchain.agents import create_agent
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langchain.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv(verbose=True)

# DB_URI = ""   # 数据库连接地址
# with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
#     checkpointer.setup()  # 创建快照
#     agent = create_agent(model="deepseek-chat",checkpointer=checkpointer)

# 连接sqlite,定义数据库的位置
connection = sqlite3.connect("resources/checkpoint.db", check_same_thread=False)
# 初始化记忆
checkpointer = SqliteSaver(connection)

# 创建agent
agent = create_agent(model="deepseek-chat", checkpointer=checkpointer)

# 设定会话
config = {"configurable":{"thread_id":"thread_1"}}

# 调用智能体，输入消息
response = agent.invoke(
    {"messages":[HumanMessage(content="你好，我叫时雨，我喜欢看小说")]},
    config
)
print(response)