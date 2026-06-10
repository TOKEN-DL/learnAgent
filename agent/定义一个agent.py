import os
from typing import Any

from langchain_classic.memory import ConversationBufferMemory
from langchain_deepseek import ChatDeepSeek
from langchain.tools import BaseTool
from dotenv import load_dotenv
from langchain_classic import hub
import os
from langchain_classic.agents import AgentExecutor
# 定义模型

load_dotenv()

model = ChatDeepSeek(model="deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY"))


# 定义可用工具
class TextLengthTool(BaseTool):
    name = "文本字数计算工具",
    description = "当你被要求计算文本的字数时调用此工具"

    def _run(self, text):
        return len(text)


tools = [
    TextLengthTool(),
]
# 拉取定制的prompt
prompt = hub.pull("hwchase17/structured-chat-agent")

# 设置记忆
memory = ConversationBufferMemory(
    memory_key='chat_history',
    return_messages=True,
)


# 定义agent执行器
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=model,
    tools=tools,
    memory=memory,
    handle_parsing_errors=True,   #出现错误的话需要自己修理错误，不会报错
    verbose=True,
)

print(agent_executor.invoke({"input": "君不见黄河之水天上来，这个有多少个字"}))

