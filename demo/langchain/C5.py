from langchain.tools import tool
from langchain_core.tools import ToolException
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
import dotenv
dotenv.load_dotenv()

@tool
def book_course(user_name: str, course_name: str) -> str:
    """为用户预订菜鸟教程 RUNOOB 的课程。

    Args:
        user_name: 用户姓名
        course_name: 课程名称
    """
    # 校验用户是否存在
    valid_users = {"张三", "李四", "王五"}
    if user_name not in valid_users:
        raise ToolException(
            f"用户 '{user_name}' 不存在。"
            f"有效用户：{', '.join(sorted(valid_users))}"
        )

    # 校验课程是否存在
    valid_courses = {"Python3 基础教程", "HTML 基础教程", "Java 面向对象"}
    if course_name not in valid_courses:
        raise ToolException(
            f"课程 '{course_name}' 不存在。"
            f"有效课程：{', '.join(sorted(valid_courses))}"
        )

    return f"已为 {user_name} 成功预订《{course_name}》"


model = init_chat_model("deepseek:deepseek-v4-flash")
agent = create_react_agent(
    model=model,
    tools=[book_course],
    handle_tool_errors=True
)

# 错误调用：用户不存在
result = agent.invoke({
    "messages": [HumanMessage(content="帮赵六预订 Python3 基础教程")]
})
print(f"\n错误-用户不存在:")
for msg in result["messages"]:
    if msg.type == "tool":
        print(f"  [{msg.type}] {msg.content[:80]}")
    elif msg.type == "ai" and msg.content:
        print(f"  [{msg.type}] {msg.content[:100]}")