from typing import Annotated, Any
from dotenv import load_dotenv
load_dotenv()

from langchain.tools import tool, InjectedState



@tool
def remember_preference(
    preference:str,
    state: Annotated[dict[str,Any],InjectedState],
):
    """记住用户的偏好设置。

    Args:
        preference: 用户的偏好内容
        state: 系统自动注入的当前 Agent 状态
    """
    message=state.get('message',[])
    message_count=len(message)

    previous_prefs=state.get('user_preferences',[])

    return (
        f'已记住偏好：{preference}',
        f'(当前对话共 {message_count})',
        f'之前偏好: { previous_prefs }',
    )

result = remember_preference.invoke({
    "preference": "暗色主题",
    # state 不需要传，由框架自动注入
})
print(result)