import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch
from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv
from langchain.agents import create_agent

load_dotenv()

llm=ChatDeepSeek(
    model='deepseek-chat',
    temperature=0,
    api_key=os.getenv('DEEPSEEK_KEY')
)

# 1. 设置不同菜系的处理链
sichuan_prompt = ChatPromptTemplate.from_template(
    "你是一位川菜大厨。请用正宗的川菜做法，回答关于「{question}」的问题。"
)
sichuan_chain = sichuan_prompt | llm | StrOutputParser()

cantonese_prompt = ChatPromptTemplate.from_template(
    "你是一位粤菜大厨。请用经典的粤菜做法，回答关于「{question}」的问题。"
)
cantonese_chain = cantonese_prompt | llm | StrOutputParser()

# 定义备用通用链
general_prompt = ChatPromptTemplate.from_template(
    "你是一个美食助手。请回答关于「{question}」的问题。"
)
general_chain = general_prompt | llm | StrOutputParser()

classifier_prompt=ChatPromptTemplate.from_template(
    """根据用户问题中提到的菜品，将其分类为：['川菜', '粤菜', 或 '其他']。
    不要解释你的理由，只返回一个单词的分类结果。
    问题: {question}"""
)

classifier_chain=classifier_prompt | llm | StrOutputParser()

router_branch= RunnableBranch(
    (lambda x: '川菜' in x['topic'],sichuan_chain),
    (lambda x: '粤菜' in x['topic'], cantonese_chain),
    general_chain
)

full_router_chain={'topic':classifier_chain,'question':lambda x : x['question']} | router_branch

result = full_router_chain.invoke({"question": "麻婆豆腐怎么做？"})

print(result)