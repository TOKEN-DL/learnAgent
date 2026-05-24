import os
from urllib import response

from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

# 利用langchain框架调用对应的大模型，并使用消息对象

def init():
    load_dotenv()
    model = ChatOpenAI(model="deepseek-v4-pro", base_url="https://api.deepseek.com", api_key=os.getenv('DEEPSEEK_API_KEY'))

    systemMessage = SystemMessage(content="你是我的物理助教，用通俗易懂的语言解释物理概念")

    human = HumanMessage(content="什么是波粒二象性？")

    message = [
        systemMessage,human
    ]

    response = model.invoke(message)

    print(response.content)

def langChainDeepSeek():
    load_dotenv()
    model = ChatDeepSeek(model="deepseek-v4-pro")

    systemMessage = SystemMessage(content="你是我的物理助教，用通俗易懂的语言解释物理概念")

    human = HumanMessage(content="什么是波粒二象性？")

    message = [
        systemMessage,human
    ]

    response = model.invoke(message)

    print(response.content)


if __name__ == '__main__':
    langChainDeepSeek()

