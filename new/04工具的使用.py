from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# 当传入的参数比较复杂可以定义

class WeatherInput(BaseModel):
    """查询天气的输入参数"""
    location: str = Field(default="北京",description="城市的名字")

# 设定工具
@tool(args_schema=WeatherInput)
def get_weather(location: str) -> str:
    """
    这是一个查询天气的工具
    :param location:  地区
    :return:  返回天气情况
    """

    return f"在{location}的天气，现在很晴朗"


# 设定智能体
agent = create_agent(
    model="deepseek-chat",
    tools=[get_weather],
)



# 提问
for token, metadata in agent.stream(
        {"messages":[ HumanMessage(content="现在海南天气怎么样")]},
    stream_mode="messages"
):
    print(token.content, end="", flush=True)

