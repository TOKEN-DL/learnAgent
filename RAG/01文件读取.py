from langchain_community.document_loaders import TextLoader, PyPDFLoader



# txt文件读取
loader = TextLoader("demo.txt", encoding="utf-8")
docs = loader.load()

print(docs)

# pdf文件读取

loader = PyPDFLoader("故宫介绍.pdf")
docs = loader.load()
print(docs)

# 文档加载器还包含了各种各样的文件类型
