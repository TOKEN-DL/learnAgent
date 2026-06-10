from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv

load_dotenv()
# 初始化checkpointer
checkpointer = InMemorySaver()

# 通过中间件的方式总结之前的上下文内容
middleware=SummarizationMiddleware(
        model="deepseek-chat",
        trigger=("messages",3), # 触发器，当满足条件时，触发总结
        keep=("messages",1)   # 保留20条完整消息
    )

agent = create_agent(
    model="deepseek-chat",
    middleware=[middleware],
    checkpointer=checkpointer
)

config:RunnableConfig = {"configurable":{"thread_id": "thread_3"}}

# 制造长会话历史
agent.invoke({"messages":[HumanMessage(content="你好我是时雨")]}, config)
agent.invoke({"messages":[HumanMessage(content="我是一个程序员")]}, config)
agent.invoke({"messages":[HumanMessage(content="我喜欢的动物是狗狗")]}, config)



response = agent.invoke({"messages":[HumanMessage(content="你还记得我是谁么?")]}, config)


print(response)