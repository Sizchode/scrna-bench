#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .code_sandbox import CodeSandbox
from .evaluator import Evaluator
from .executor import Executor
from .memory import GlobalMemory
from .planner import Planner


BASELINE_ROOT = Path(__file__).resolve().parents[1]
SCGOV_BENCH_ROOT = BASELINE_ROOT.parents[1]
DEFAULT_NOTEBOOK_DIR = SCGOV_BENCH_ROOT / "results" / "cellagent" / "notebooks"


def build_llm(args: argparse.Namespace) -> Any:
    if args.backend == "ollama":
        from langchain_community.llms import Ollama

        return Ollama(model=args.model, base_url=args.base_url)

    if args.backend == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=args.model, temperature=0)

    raise ValueError(f"Unsupported backend: {args.backend}")


def read_case(case_file: Path, case_id: str) -> dict[str, Any]:
    with case_file.open() as handle:
        for line in handle:
            record = json.loads(line)
            if record["case_id"] == case_id:
                return record
    raise ValueError(f"Case not found: {case_id}")


def build_trace(
    *,
    case_id: str | None,
    instruction: str,
    data_file_path: Path,
    steps: list[dict[str, Any]],
    trace_steps: list[dict[str, Any]],
    notebook_result: str,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "agent": "cellagent",
        "model": "cellagent-original-loop",
        "instruction": instruction,
        "data_file_path": str(data_file_path),
        "planned_steps": steps,
        "trace": {"steps": trace_steps},
        "notebook_result": notebook_result,
    }


def run_agent(
    *,
    llm: Any,
    instruction: str,
    data_file_path: Path,
    notebook_path: Path,
    max_attempts: int,
    case_id: str | None = None,
) -> dict[str, Any]:
    import scanpy as sc

    global_memory = GlobalMemory()
    code_sandbox = CodeSandbox(notebook_path=str(notebook_path))

    adata = sc.read_h5ad(data_file_path)
    data_representation = str(adata)

    planner = Planner(llm, data_representation)
    executor = Executor(llm, str(data_file_path), global_memory)
    evaluator = Evaluator(llm)

    steps = planner.plan(instruction)
    trace_steps: list[dict[str, Any]] = []
    last_notebook_result = ""

    for index, step in enumerate(steps, start=1):
        local_memory: list[dict[str, Any]] = []
        success = False
        attempt = 0
        step_attempt_limit = max_attempts + (1 if "batch correction" in step["description"].lower() else 0)

        while not success and attempt < step_attempt_limit:
            attempt += 1
            tools = executor.tool_selector.select_tools(step["description"], instruction)
            code, analysis = executor.code_programmer.generate_code(
                step_description=step["description"],
                user_requirements=instruction,
                data_description=data_representation,
                global_memory=global_memory,
                tools_docs=tools,
                local_memory=local_memory,
            )

            local_memory.append({"attempt": attempt, "code": code, "analysis": analysis})
            global_memory.add_code(code)

            code_sandbox.add_code_cell(code)
            execution_result = code_sandbox.execute_notebook()
            last_notebook_result = execution_result

            evaluation = evaluator.evaluate(
                code=code,
                execution_result=execution_result,
                step_description=step["description"],
                user_requirements=instruction,
                data_description=data_representation,
            )

            trace_steps.append(
                {
                    "step_num": len(trace_steps) + 1,
                    "planner_step_id": step.get("id", index),
                    "reasoning": step["description"],
                    "attempt": attempt,
                    "tool_calls": [{"tool_name": item["name"], "params": {}} for item in tools],
                    "code": code,
                    "execution_result": execution_result,
                    "evaluation": evaluation,
                }
            )

            if evaluator.is_result_satisfactory(evaluation):
                success = True
            elif attempt < step_attempt_limit:
                executor.code_programmer.optimize_code(
                    evaluation_feedback=evaluation,
                    local_memory=local_memory,
                )

    return build_trace(
        case_id=case_id,
        instruction=instruction,
        data_file_path=data_file_path,
        steps=steps,
        trace_steps=trace_steps,
        notebook_result=last_notebook_result,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runnable CellAgent baseline using the original planner-executor-evaluator loop.")
    parser.add_argument("--backend", choices=["ollama", "openai"], default="ollama")
    parser.add_argument("--model", default="llama3.1")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--case-file", type=Path, help="Optional scGov-Bench JSONL case file.")
    parser.add_argument("--case-id", help="Case identifier inside the JSONL file.")
    parser.add_argument("--instruction", help="User instruction when not using --case-file.")
    parser.add_argument("--data-file", type=Path, help="Input AnnData file when not using --case-file.")
    parser.add_argument("--output-json", type=Path, help="Optional output path for trace JSON.")
    parser.add_argument("--notebook-path", type=Path, help="Optional notebook output path.")
    parser.add_argument("--max-attempts", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.case_file:
        if not args.case_id:
            raise ValueError("--case-id is required when using --case-file.")
        case = read_case(args.case_file, args.case_id)
        instruction = case["instruction"]
        data_file_path = Path(case["snapshot_path"])
        case_id = case["case_id"]
    else:
        if not args.instruction or not args.data_file:
            raise ValueError("Without --case-file you must provide --instruction and --data-file.")
        instruction = args.instruction
        data_file_path = args.data_file
        case_id = None

    if not data_file_path.exists():
        raise FileNotFoundError(f"Data file does not exist: {data_file_path}")

    notebook_path = args.notebook_path or (DEFAULT_NOTEBOOK_DIR / f"{(case_id or data_file_path.stem)}.ipynb")
    notebook_path.parent.mkdir(parents=True, exist_ok=True)

    llm = build_llm(args)
    trace = run_agent(
        llm=llm,
        instruction=instruction,
        data_file_path=data_file_path,
        notebook_path=notebook_path,
        max_attempts=args.max_attempts,
        case_id=case_id,
    )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as handle:
            json.dump(trace, handle, indent=2)
    else:
        print(json.dumps(trace, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
