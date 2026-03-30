from __future__ import annotations

from langchain_core.prompts import PromptTemplate

from .utils.json_utils import extract_and_parse_json


class Planner:
    def __init__(self, llm, data_representation: str):
        self.llm = llm
        self.data_representation = data_representation
        self.prompt_template = """You are a planning assistant specialized in bioinformatics.
Create a detailed step-by-step plan for the user's scRNA-seq analysis task.
Return valid JSON so that downstream components can parse the subtasks.

User request:
{user_task}

Data description:
{data_representation}

Output format:
{{
  "steps": [
    {{
      "id": 1,
      "description": "Description of step one"
    }},
    {{
      "id": 2,
      "description": "Description of step two"
    }}
  ]
}}

Make sure the JSON is valid and contains all necessary steps.
"""

    def plan(self, user_task: str) -> list[dict]:
        prompt = PromptTemplate(
            input_variables=["user_task", "data_representation"],
            template=self.prompt_template,
        )
        response = self.llm.invoke(
            prompt.format(
                user_task=user_task,
                data_representation=self.data_representation,
            )
        )
        parsed_json = extract_and_parse_json(response)
        return parsed_json.get("steps", []) if parsed_json else []

    def generate_final_result(self):
        return None
