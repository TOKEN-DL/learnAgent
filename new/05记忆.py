from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv

load_dotenv()

agent = create_agent(
    model="deepseek-chat",
    checkpointer=InMemorySaver()
)

config = {"configurable": {"thread_id": "1"}} # 设置一次会话的记忆

response = agent.invoke(
    {"messages": [HumanMessage(content="你好，我时雨")]},
    config
)

print(response)

response = agent.invoke(
    {"messages": [HumanMessage(content="你好，你还知道我叫什么么")]},
    config
)

print(response)