from langchain_core.messages import HumanMessage, PlainTextContentBlock, ImageContentBlock,SystemMessage
from langchain.chat_models import init_chat_model
import dotenv
dotenv.load_dotenv()

model = init_chat_model("deepseek:deepseek-v4-flash", temperature=0.7)

# 没有系统指令的回复
messages_no_system = [HumanMessage(content="介绍菜鸟教程")]
response = model.invoke(messages_no_system)
print(f"无系统指令: {response.content[:80]}...")

# 有系统指令的回复
messages_with_system = [
    SystemMessage(content="你是一个小红书风格的博主，回复要活泼、使用 emoji、带话题标签"),
    HumanMessage(content="介绍菜鸟教程")
]
response = model.invoke(messages_with_system)
print(f"\n有系统指令: {response.content}")