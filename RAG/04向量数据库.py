from langchain_classic.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_deepseek import ChatDeepSeek
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from langchain_classic.memory import ConversationBufferMemory
from pandas.io.formats.format import return_docstring

load_dotenv()

########################  数据导入阶段
# 1.设置文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # 每块文本最大长度
    chunk_overlap=40,       # 分割片段之间重叠的长度
    separators=["\n\n", "\n", "。","!","?",",","、",""]  # 用于分割的字符

)

# 2.txt文件读取
loader = TextLoader("demo.txt", encoding="utf-8")
docs = loader.load()

# 3.进行分割
texts = text_splitter.split_documents(docs)


# 4.设置向量化模型
embedding = DashScopeEmbeddings(
    model="text-embedding-v4",  # 指定阿里向量模型
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)

# 5.把文件存入向量数据库
db = FAISS.from_documents(texts, embedding)

# 6.设置检索器
retriever = db.as_retriever()
# 7.查询数据库
# retrieved_docs = retriever.invoke("长城途经几个城市？")
# print(retrieved_docs[1].page_content)




###########  用户检索阶段
# 设置模型
model = ChatDeepSeek(model="deepseek-chat")
# 设置记忆桶
memory = ConversationBufferMemory(return_messages=True,
                                  memory_key='chat_history',
                                  output_key='answer')

# 把模型，记忆，检索工具带入设定好的工作流，Conversational是支持记忆，Retrieval是支持向量检索
qa = ConversationalRetrievalChain.from_llm(
    llm=model,
    retriever=retriever,
    memory=memory,
)

result = qa.invoke({"chat_history": memory,
           "question": "你还知道我第一个问题是什么么？",})

print(result)