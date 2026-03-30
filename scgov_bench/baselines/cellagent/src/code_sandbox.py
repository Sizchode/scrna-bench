from __future__ import annotations

import os
from datetime import datetime

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


class CodeSandbox:
    def __init__(self, notebook_path: str):
        self.notebook_path = notebook_path
        self.nb = nbformat.v4.new_notebook()
        self.nb["cells"] = []

    def add_code_cell(self, code: str) -> None:
        self.nb["cells"].append(nbformat.v4.new_code_cell(code))

    def _generate_unique_filename(self, base_path: str) -> str:
        if not os.path.exists(base_path):
            return base_path
        base, ext = os.path.splitext(base_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base}_{timestamp}{ext}"

    def execute_notebook(self) -> str:
        try:
            notebook_dir = os.path.dirname(self.notebook_path)
            os.makedirs(notebook_dir, exist_ok=True)
            unique_notebook_path = self._generate_unique_filename(self.notebook_path)
            ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
            ep.preprocess(self.nb, {"metadata": {"path": notebook_dir or "./"}})

            with open(unique_notebook_path, "w", encoding="utf-8") as handle:
                nbformat.write(self.nb, handle)

            return f"Execution succeeded. Saved notebook to: {unique_notebook_path}"
        except Exception as exc:
            return f"Execution failed: {exc}"
