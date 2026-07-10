import os
from typing import List

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()
llm=ChatDeepSeek(model="deepseek-chat",temperature=0,api_key=os.getenv('DEEPSEEK_API_KEY'))

class PersonInfor(BaseModel):
    name:str = Field (description='任务姓名')
    age: int =Field(description='任务年龄')
    skills:List[str] =Field(description='技能列表')

parser=PydanticOutputParser(pydantic_object=PersonInfor)

prompt=PromptTemplate(
    template='请根据一下文本提取信息。\n{format_instructions}\n{text}\n',
    input_variables=['text'],
    partial_variables={'format_instructions':parser.get_format_instructions()},
)

chain= prompt | llm | parser

text = "张三今年30岁，他擅长Python和Go语言。"
result=chain.invoke({'text':text})

print("\n--- 解析结果 ---")
print(f"结果类型: {type(result)}")
print(result)
print("--------------------\n")

print(f"姓名: {result.name}")
print(f"年龄: {result.age}")
print(f"技能: {result.skills}")