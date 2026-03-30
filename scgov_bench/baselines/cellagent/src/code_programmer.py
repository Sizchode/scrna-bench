from __future__ import annotations

from pathlib import Path

from langchain_core.prompts import PromptTemplate


class CodeProgrammer:
    def __init__(self, llm, data_file_path: str):
        self.llm = llm
        self.data_file_path = data_file_path
        self.scseq_path = Path(__file__).resolve().parents[2] / "scOmni" / "codes" / "SinglecellSequencing.py"
        self.prompt_template = """You are an expert bioinformatics programmer.
Write Python code that completes the current task step.

User request:
{user_requirements}

Data description:
{data_description}

Historical code:
{historical_code}

Current task step:
{step_description}

Selected tools and docs:
{tools_docs}

Requirements:
- Use the selected tools when appropriate.
- Prefer the real scOmni toolkit when the selected tools mention scOmni.
- The scOmni single-cell toolkit lives at: {scseq_path}
- If you use scOmni, import from that file or from its package location instead of inventing replacement APIs.
- When batch integration, annotation, or trajectory inference are requested, prefer the corresponding scOmni callables before falling back to generic Scanpy code.
- Include the necessary imports and data loading.
- Use this data file path: {data_file_path}
- Do not add explanations or comments outside the code.
- Return the result as a fenced Python code block.
"""

    def generate_code(self, step_description, user_requirements, data_description, global_memory, tools_docs, local_memory):
        prompt = PromptTemplate(
            input_variables=["user_requirements", "data_description", "historical_code", "step_description", "tools_docs", "data_file_path"],
            template=self.prompt_template,
        )
        response = self.llm.invoke(
            prompt.format(
                user_requirements=user_requirements,
                data_description=data_description,
                historical_code=global_memory.get_all_code(),
                step_description=step_description,
                tools_docs=tools_docs,
                data_file_path=self.data_file_path,
                scseq_path=str(self.scseq_path),
            )
        )
        return self.extract_code(response), self.extract_analysis(response)

    def optimize_code(self, evaluation_feedback, local_memory):
        prompt_template = """You are an expert bioinformatics programmer.
Improve the previously generated code using the evaluation feedback below.

Evaluation feedback:
{evaluation_feedback}

Previous attempts:
{previous_attempts}

Rewrite the improved code and return it as a fenced Python code block.
"""
        previous_attempts = "\n".join([f"Attempt {item['attempt']} code:\n{item['code']}" for item in local_memory])
        prompt = PromptTemplate(
            input_variables=["evaluation_feedback", "previous_attempts"],
            template=prompt_template,
        )
        response = self.llm.invoke(
            prompt.format(
                evaluation_feedback=evaluation_feedback,
                previous_attempts=previous_attempts,
            )
        )
        return self.extract_code(response), self.extract_analysis(response)

    def extract_code(self, response: str) -> str:
        start = response.find("```python")
        end = response.find("```", start + 9)
        if start != -1 and end != -1:
            return response[start + 9 : end].strip()
        return response.strip()

    def extract_analysis(self, response: str) -> str:
        return ""
