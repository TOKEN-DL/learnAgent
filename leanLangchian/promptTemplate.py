from langchain_core.prompts import FewShotChatMessagePromptTemplate,ChatPromptTemplate,SystemMessagePromptTemplate, AIMessagePromptTemplate, HumanMessagePromptTemplate
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek


from openai.types.responses import response

# 提示词模板，可以根据系统、用户、AI进行划分
# 模板的作用就是把其中可以替换的变量提取出来进行热插拔对象，比如说某语言转换成某语言

# 系统提示词模板

system_template_text = """你是一位专业的翻译，能够将{input_language}翻译为{output_language},
并且输出文本会根据用户要求的任何语言风格进行调整，请只输出翻译后的文本，不要有任何其他内容
"""

# 通过from_template获取文本模板
system_template_template = SystemMessagePromptTemplate.from_template(system_template_text)


# 用户提示词模板
human_template_text = "文本：{text}\n语言风格：{style}"

human_template_template = HumanMessagePromptTemplate.from_template(human_template_text)


# 利用format定义模板中的变量，根据定义好的模板，生成提示词
system_prompt = system_template_template.format(input_language="英语", output_language="汉语")
human_prompt = human_template_template.format(text="I Love you", style="文言文")



##### Chat模板对之前的三种消息进行总和

# 定义模板
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", """你是一位专业的翻译，能够将{input_language}翻译为{output_language},
并且输出文本会根据用户要求的任何语言风格进行调整，请只输出翻译后的文本，不要有任何其他内容."""),
        ("human", "文本：{text}\n语言风格：{style}")
]
)

# 输入指定信息
prompt_value = prompt_template.invoke({
    "input_language": "英语",
    "output_language": "汉语",
    "text": "I Love you",
    "style": "文言文"
})










# 基础调用系统提示词模板和用户提示词模板
def init():
    load_dotenv()
    model = ChatDeepSeek(model="deepseek-v4-pro")

    response = model.invoke([system_prompt, human_prompt])

    print(response.content)


# 调用Chat提示词模板
def chat():
    load_dotenv()
    model = ChatDeepSeek(model="deepseek-v4-pro")

    response = model.invoke(prompt_value)

    print(response.content)


if __name__ == '__main__':
    #print(system_template_template.input_variables) # ['input_language', 'output_language'] 确认好模板中提出的变量
    chat()