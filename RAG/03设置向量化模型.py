from langchain_community.embeddings import DashScopeEmbeddings
from dotenv import load_dotenv
import os

load_dotenv(verbose=True)

# 设置向量化模型
embedding = DashScopeEmbeddings(
    model="text-embedding-v4",  # 指定阿里向量模型
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)

result = embedding.embed_documents(["hello", "world"])
print(result)