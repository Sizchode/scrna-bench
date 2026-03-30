from __future__ import annotations

from .code_programmer import CodeProgrammer
from .tool_selector import ToolSelector


class Executor:
    def __init__(self, llm, data_file_path: str, global_memory):
        self.llm = llm
        self.data_file_path = data_file_path
        self.global_memory = global_memory
        self.tool_selector = ToolSelector(llm)
        self.code_programmer = CodeProgrammer(llm, data_file_path)
