from __future__ import annotations


class GlobalMemory:
    def __init__(self):
        self.codes = []

    def add_code(self, code: str) -> None:
        self.codes.append(code)

    def get_all_code(self) -> str:
        return "\n".join(self.codes)
