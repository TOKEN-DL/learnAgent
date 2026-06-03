from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 设置文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # 每块文本最大长度
    chunk_overlap=40,       # 分割片段之间重叠的长度
    separators=["\n\n", "\n", "。","!","?",",","、",""]  # 用于分割的字符

)

# 读取文本
loader = TextLoader("demo.txt", encoding="utf-8")
docs = loader.load()

# 进行分割
texts = text_splitter.split_documents(docs)