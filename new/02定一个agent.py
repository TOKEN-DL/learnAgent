from langchain.agents import create_agent
from dotenv import load_dotenv


load_dotenv()



agent = create_agent(model="deepseek-chat",
                     system_prompt="你是一个智能助手")



# 阻塞式调用
response = agent.invoke({
    "messages": [{"role": "user", "content": "你是谁？"}]
})

# print(response)


# 流式调用
messages = agent.stream(
    {"messages": [{"role": "user", "content": "你是谁？"}]},
    stream_mode="messages"
)

for token, metadata in messages:
    if token.content:
        print(token.content, end="", flush=True)