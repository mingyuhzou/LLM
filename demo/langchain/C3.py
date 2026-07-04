from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel,Field
from langchain.chat_models import init_chat_model

class PersonInfo(BaseModel):
    name: str = Field(description='任务姓名')
    age: str = Field(description='年龄')
    occupation:str = Field(description='职业')
    skills: list[str] = Field(description='技能列表')

model = init_chat_model("deepseek:deepseek-v4-flash",model_kwargs={
        "extra_body": {
            "thinking": {
                "type": "disabled"
            }
        }
    }
                    )
model=model.with_structured_output(PersonInfo)

# 传入非结构化文本，获取结构化数据
text = "张三今年28岁，是一名全栈工程师，精通 Python、React 和 Docker"
result = model.invoke(text)

print(f"姓名: {result.name}")
print(f"年龄: {result.age}")
print(f"职业: {result.occupation}")
print(f"技能: {', '.join(result.skills)}")
print(f"类型: {type(result)}")