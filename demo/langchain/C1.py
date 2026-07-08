import os
import dotenv
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()

from langchain.chat_models import init_chat_model

model=ChatOpenAI(
    model="glm-4.7-flash-free",
    temperature=0.7,
    max_tokens=4096,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://aihubmix.com/v1"
)

response=model.invoke('介绍菜鸟教程 RUNOOB')

print(response)