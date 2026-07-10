import os
from typing import List

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

client=OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url='https://api.deepseek.com'
)

def send_message(message,tools=None):
    response=client.chat.completions.create(
        model='deepseek-chat',
        message=message,
        tools=tools,
        tool_choice='auto'
    )
    return response.choices[0].message

tools=[
    {
        'type':'function',
        'function':{
            'name':'get_weather',
            'description':"获取指定地点的天气信息",
            'parameters':{
                'type':'object',
                'properties':{
                    'location':{
                        'type':'string',
                        'description':'城市和省份，例如：杭州市, 浙江'
                    }
                },
                    "required": ["location"]
            }
        }
    }
]

messages=[{'role':'user','content':"杭州今天天气怎么样"}]
print(f"User> {messages[0]['content']}\n")
message=send_message(messages,tools=tools)

if message.tool_calls:
    print("--- 模型发起了工具调用 ---")
    tool_call=message.tool_calls[0]
    function_info=tool_call.function
    print(f"工具名称: {function_info.name}")
    print(f"工具参数: {function_info.arguments}")