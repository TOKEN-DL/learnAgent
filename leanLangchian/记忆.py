from langchain_classic.chains.conversation.base import ConversationChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
import os
from dotenv import load_dotenv


memory = ConversationBufferMemory(return_messages=True)

memory.save_context({"input": "我的名字是时雨"},
                    {"output": "你好，时雨"})

#print(memory.load_memory_variables({}))
# {'history': [HumanMessage(content='我的名字是时雨', additional_kwargs={}, response_metadata={}), AIMessage(content='你好，时雨', additional_kwargs={}, response_metadata={}, tool_calls=[], invalid_tool_calls=[])]}


memory.save_context({"input": "我是一名程序员"},
                    {"output": "好的，我记住了"})

#print(memory.load_memory_variables({}))
# {'history': [HumanMessage(content='我的名字是时雨', additional_kwargs={}, response_metadata={}), AIMessage(content='你好，时雨', additional_kwargs={}, response_metadata={}, tool_calls=[], invalid_tool_calls=[]), HumanMessage(content='我是一名程序员', additional_kwargs={}, response_metadata={}), AIMessage(content='好的，我记住了', additional_kwargs={}, response_metadata={}, tool_calls=[], invalid_tool_calls=[])]}

# 通过设置一个记忆容器去保存记忆，类似于小样本一样创造前置性的记忆


# 设置好记忆后需要，就需要把记忆输入到提示词里去

# 设置一个提示词模板
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个乐于助人的助手。"),
        MessagesPlaceholder(variable_name='history'),
        ("human", "{user_input}"),
    ]
)

# 设置模型
load_dotenv(verbose=True)
model = ChatDeepSeek(model="deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY"))

# 创造基础的工作流
chain = prompt | model


user_input = "你知道我的名字么"
history = memory.load_memory_variables({})["history"]

result = chain.invoke({"user_input": user_input, "history": history})
print(result)

# 回答后需要把新一轮的记忆存入

# 感觉就特别麻烦，每次问答后都要存入一次记忆

memory.save_context({"input": user_input},{"output": result.content})


# 再问一遍问题
user_input = "根据对话历史告诉我，我上一个问你的问题是什么？请重复一遍"
history = memory.load_memory_variables({})["history"]

result = chain.invoke({"user_input": user_input, "history": history})



# 可以调用记忆链，可以有效解决每次对话后都要手动存入记忆
# 就是一个内置好的工作流
memory1 = ConversationBufferMemory(return_messages=True)

chain = ConversationChain(llm=model, memory=memory1)

print(chain.invoke({"input": "你好我是时雨"}))
print(chain.invoke({"input": "我是一名程序员"}))
print(chain.invoke({"input": "你知道我叫什么么"}))



