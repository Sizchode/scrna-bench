from __future__ import annotations

import json

from langchain_core.prompts import PromptTemplate

from .tools.tool_registry import ToolRegistry
from .utils.json_utils import extract_and_parse_json


class ToolSelector:
    def __init__(self, llm):
        self.llm = llm
        self.tool_registry = ToolRegistry()
        self.prompt_template = """You are a bioinformatics tool-selection expert.
Choose the most appropriate tools for the current step from the available tool list.
Return valid JSON.

User request:
{user_requirements}

Current task step:
{step_description}

Available tools:
{available_tools}

Output format:
{{
  "selected_tools": ["Tool1", "Tool2"]
}}

Requirements:
1. The JSON must be valid.
2. Every selected tool must exist in the available tool list.
3. selected_tools must be a list of tool names.
"""

    def select_tools(self, step_description: str, user_requirements: str) -> list[dict]:
        prompt = PromptTemplate(
            input_variables=["user_requirements", "step_description", "available_tools"],
            template=self.prompt_template,
        )
        response = self.llm.invoke(
            prompt.format(
                user_requirements=user_requirements,
                step_description=step_description,
                available_tools=json.dumps(self.tool_registry.get_available_tools(), ensure_ascii=False),
            )
        )
        parsed_json = extract_and_parse_json(response)
        selected_tools = parsed_json.get("selected_tools", []) if parsed_json else []
        return self.tool_registry.get_tools_docs(selected_tools)
