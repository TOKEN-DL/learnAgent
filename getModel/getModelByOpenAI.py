from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# 接入Deepseek
def getDeeoSeek():
    client = OpenAI(base_url='https://api.deepseek.com',
                    api_key=os.getenv('DEEPSEEK_API_KEY'))
    response = client.chat.completions.create(
        model= 'deepseek-v4-pro',
        messages = [
            {"role": "system", "content": "你是一个智能的人工助手"},
            {"role": "user", "content": "你好，请问你是谁？"},
        ]
    )

    print(response.choices[0].message.content)


# 接入阿里云千问
def getDashScope():
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen3.6-plus",
        messages=[{'role': 'user', 'content': '你是谁？'}]
    )
    print(completion.choices[0].message.content)



if __name__ == '__main__':
    getDashScope()


