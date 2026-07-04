from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel,Field
from langchain.chat_models import init_chat_model

model = init_chat_model("deepseek:deepseek-v4-flash")

class WeatherInput(BaseModel):
    city: str = Field(description='城市名称，如 杭州、北京')
    unit: str = Field(default='celsius',description='温度单位，celsius(摄氏度) 或fahrenheit(华氏度)')

class CalculatorInput(BaseModel):
    expression: str =Field(
        description='要计算的数学表达式，如(3+5)*2'
    )

model=model.bind_tools([WeatherInput,CalculatorInput])

response=model.invoke(
    "北京今天多少度？顺便帮我算一下 123 * 456"
)

print(f'模型请求了{len(response.tool_calls)}个工具调用')
for tc in response.tool_calls:
    print(f' {tc['name']}({tc['args']})')