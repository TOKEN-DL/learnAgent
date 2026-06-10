from dotenv import load_dotenv
import os
import requests


# 调用DeepSeek
def getDeepSeek():
    # 加载环境变量（读取 .env 里的 API Key）
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")

    # API 地址（和 curl 一致）
    url = "https://api.deepseek.com/chat/completions"

    # 请求头（完全对应 curl 的 -H 参数）
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"  # Bearer + 密钥，格式必须正确
    }

    # 请求体（1:1 复刻 curl 的 -d 所有参数）
    data = {
        "model": "deepseek-v4-pro",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ],
        "thinking": {"type": "enabled"},  # 对应 curl 里的 "enabled"
        "reasoning_effort": "high",
        "stream": False
    }

    # 发送 POST 请求（和 curl 功能完全一致）
    response = requests.post(url, headers=headers, json=data)

    # 打印结果
    print("状态码:", response.status_code)
    print("返回结果:\n", response.json())
    print(response.json()["choices"][0]["message"]["content"])



# 调用DashScope
def getDashScope():
    # 加载环境变量
    load_dotenv()
    api_key = os.getenv("DASHSCOPE_API_KEY")

    # 1:1 对应 curl 的请求地址
    url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

    # 对应 curl 的请求头 -H 参数
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 对应 curl 的请求体 -d 参数
    data = {
        "model": "qwen3.6-plus",
        "messages": [
            {"role": "user", "content": "你是谁？"}
        ]
    }

    # 发送请求 + 提取回复
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()

        # 🔥 精准提取 AI 回复文本（和之前用法一致）
        answer = result["choices"][0]["message"]["content"]
        print("通义千问回复：", answer)

    except Exception as e:
        print("请求失败：", e)
        if 'response' in locals():
            print("错误详情：", response.text)



if __name__ == '__main__':
    getDashScope()