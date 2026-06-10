from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from langchain.agents import create_agent
from dotenv import load_dotenv


# 提示词工程一般包含：身份角色， 指令说明， 对话示例， 背景信息
load_dotenv()


class CapitaInfo(BaseModel):
    name: str
    location: str
    vibe: str
    economy: str

agent = create_agent(
    model="deepseek-chat",
    system_prompt="你是一个科幻作家，根据用户的要求创建一个太空之都",
    response_format=CapitaInfo   # 设置结构化输出格式
)

response = agent.invoke(
    {"messages": [HumanMessage(content="月球的首都是什么？")]}
)

print(response)
