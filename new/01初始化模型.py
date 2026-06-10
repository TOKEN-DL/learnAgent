from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os

load_dotenv()

# 调用模型
model1 = init_chat_model(model="deepseek-chat")


# 调用模型2
base_url = os.getenv("DASHSCOPE_BASE_URL")
api_key = os.getenv("DASHSCOPE_API_KEY")

model2 = init_chat_model(
    model="qwen-max",
    model_provider="openai",
    base_url=base_url,
    api_key=api_key,

    # 设置文本参数
)


#   模型的调用
response = model1.invoke([
    {"role":"system", "content": "你现在扮演火箭队的武藏，以武藏的口吻回答用户的问题。"},
    {"role":"user", "content": "你是谁"},
])
#print(response.content)

# 流式调用
stream = model1.stream("你是谁？")

for chunk in stream:
    #print(chunk)
    # content='' additional_kwargs={} response_metadata={'model_provider': 'deepseek'}
    # id='lc_run--019e9377-20a2-76c3-a029-065f8a8d0bda'
    # tool_calls=[]
    # invalid_tool_calls=[]
    # tool_call_chunks=[]
    print(chunk.content, end="", flush=True)
