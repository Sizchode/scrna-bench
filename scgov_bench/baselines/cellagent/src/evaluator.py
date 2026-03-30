from __future__ import annotations

from langchain_core.prompts import PromptTemplate


class Evaluator:
    def __init__(self, llm):
        self.llm = llm
        self.prompt_template = """You are an experienced bioinformatics expert.
Evaluate whether the executed code is correct and reasonable, then provide improvement suggestions.

User request:
{user_requirements}

Data description:
{data_description}

Current task step:
{step_description}

Execution result:
{execution_result}

Return exactly this format:

Evaluation:
[your evaluation]

Suggestions:
[your suggestions]

Do not include anything else.
"""

    def evaluate(self, code, execution_result, step_description, user_requirements, data_description):
        prompt = PromptTemplate(
            input_variables=["user_requirements", "data_description", "step_description", "execution_result"],
            template=self.prompt_template,
        )
        response = self.llm.invoke(
            prompt.format(
                user_requirements=user_requirements,
                data_description=data_description,
                step_description=step_description,
                execution_result=execution_result,
            )
        )
        return response.strip()

    def is_result_satisfactory(self, evaluation: str) -> bool:
        evaluation_lower = evaluation.lower()
        return "suggestions" in evaluation_lower and (
            "none" in evaluation_lower or "no further changes" in evaluation_lower or "not needed" in evaluation_lower
        )
