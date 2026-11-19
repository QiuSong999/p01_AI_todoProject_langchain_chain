
"""查询工具函数"""

from langchain_core.tools import tool
from datetime import datetime
from typing import Literal

#用于限定tools传入的参数校验
from schema import (
    SearchTodoByTitleInput,
    SearchAllTodosInput,
    SearchTodoByStatusInput,
    SearchTodoByDeadlineInput,
    AITodoSearch
)

"""
从LangChain工具模块导入tool装饰器。

@tool作用：
1. 把普通Python函数转换成LangChain可以识别的AI工具。
2. 自动读取函数名称作为工具name。
3. 自动读取函数注释(docstring)作为工具description。
4. 根据args_schema指定的Pydantic模型生成parameters参数结构。

例如：
@tool
def search_todos_by_title(title,page,page_size):
#
LangChain会自动转换成AI可以调用的工具：
{
    name: "search_todos_by_title",
    description: "根据函数注释生成",
    parameters: {
        title,
        page,
        page_size
    }
}

后续通过：
llm.bind_tools(ai_tools)

将这些工具提供给大模型，让AI决定调用哪个工具。
"""

# db文件中导入操作数据的函数（用于工具的底层操作数据库）
from db import (
    get_todo_title,
    get_todo_title_count,
    get_todos,
    get_todo_count,
    get_todo_status,
    get_todo_status_count,
    get_todo_deadline,
    get_todo_deadline_count,
    get_todo_advanced,
    get_todo_advanced_count
)


# 以下5个工具都是LangChain Tool
### 1、定义函数，用来当AI调用的工具

# 1.1 定义根据title查询数据的函数（工具1）

@tool(args_schema=SearchTodoByTitleInput)
def search_todos_by_title(
        title: str,             #⭐这些叫“函数签名”
        page: int = 1,          #⭐同时使用详细的函数签名和 args_schema，实际有重复，后续版本选一种即可
        page_size: int = 10
):  # ⭐函数签名定义完整（参数名称、类型、默认值），通常不需要上传 args_schema。
    # ⭐也可以上传 args_schema，由 Pydantic 模型显式定义 Tool 的输入参数结构和校验规则。
    # ⭐使用 args_schema 时，Python 函数参数可以不写类型注解和默认值，但参数名称应与模型字段对应。
    """
    给AI使用的工具函数。
    注意：这个函数不是给用户直接调用的。
    它的作用：
        把一个复杂数据库查询能力，
        包装成AI能够理解的工具。

    根据标题关键词查询Todo。
    """

    todos = get_todo_title(title, page, page_size)      #调用db文件中的“根据title查询数据的函数”
    total = get_todo_title_count(title)

    return {
        "data": todos,
        "total": total,
        "page": page,
        "page_size": page_size
    }


# 1.2 定义查询所有Todo数据的函数 （工具2）
@tool(args_schema=SearchAllTodosInput)
def search_all_todos(
        page: int = 1,
        page_size: int = 10
):
    """
    给AI使用的工具函数。
    作用：
        查询数据库中的所有Todo。
    """

    todos = get_todos(page, page_size)
    total = get_todo_count()

    return {
        "data": todos,
        "total": total,
        "page": page,
        "page_size": page_size
    }


# 1.3 定义根据status查询函数（工具3）
@tool(args_schema=SearchTodoByStatusInput)
def search_todos_by_status(
        status: Literal["pending", "completed"],
        page: int = 1,
        page_size: int = 10
):
    """
    根据任务状态查询Todo。

    参数：
        status:
            pending表示未完成任务。
            completed表示已完成任务。

        page:
            查询页码。

        page_size:
            每页数量。
    """

    todos = get_todo_status(status, page, page_size)    #引用db中的get_todo_status()查询函数
    total = get_todo_status_count(status)

    return {
        "data": todos,
        "total": total,
        "page": page,
        "page_size": page_size
    }


# 1.4 定义根据deadline查询数据的函数（工具4）
@tool(args_schema=SearchTodoByDeadlineInput)
def search_todos_by_deadline(
        start_time: datetime,
        end_time: datetime,
        page: int = 1,
        page_size: int = 10
):
    """
    给AI使用的工具函数。
    作用：
        根据截止时间范围查询Todo。
    """

    todos = get_todo_deadline(
        start_time,
        end_time,
        page,
        page_size
    )

    total = get_todo_deadline_count(start_time, end_time)

    return {
        "data": todos,
        "total": total,
        "page": page,
        "page_size": page_size
    }


# 1.5 复合查询 （工具5）
# 上面4个：一个工具 = 一个条件。例如：status → search_todos_by_status
# 现在：一个工具 = 多个可选条件。例如：title + status + deadline
@tool(args_schema=AITodoSearch)
def search_todos_advanced(
        title: str | None = None,
        status: Literal["pending", "completed"] | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        page: int = 1,
        page_size: int = 10
):
    """
    根据多个条件组合查询Todo。

    支持条件：
        title:
            标题关键词。
        status:
            pending或completed。
        start_time:
            开始时间。
        end_time:
            结束时间。
        page:
            查询页码。
        page_size:
            每页数量。
    """

    todos = get_todo_advanced(      #根据多个可选条件查询具体的 Todo 数据
        title,
        status,
        start_time,
        end_time,
        page,
        page_size
    )

    total = get_todo_advanced_count(    #get_todo_advanced筛选出来的数据的个数
        title,
        status,
        start_time,
        end_time
    )

    return {
        "data": todos,
        "total": total,
        "page": page,
        "page_size": page_size
    }


"""
LangChain 的：
    @tool
    def xxx():
会自动生成：工具的name、description、parameters schema。

这里使用args_schema显式指定Pydantic参数模型，
不再完全依赖函数参数自动推断。
"""

### 2、定义AI工具列表：用于把Python中的查询工具“介绍”给大模型

ai_tools = [
    search_todos_by_title,
    search_all_todos,
    search_todos_by_status,
    search_todos_by_deadline,
    search_todos_advanced
]

# LangChain工具列表
# 把Python函数对象交给大模型
# llm.bind_tools(ai_tools)会自动读取：
# 1.工具名称
# 2.函数说明
# 3.参数结构
