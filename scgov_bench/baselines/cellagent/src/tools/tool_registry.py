from __future__ import annotations

from pathlib import Path


class ToolRegistry:
    def __init__(self):
        sc_omni_root = Path(__file__).resolve().parents[2] / "scOmni" / "codes"
        self.tools = {
            "Scanpy": "An open-source toolkit for analyzing single-cell gene expression data.",
            "Harmony": "Batch integration available through scOmni SinglecellSequencing.batch_integration(..., method=['harmony']).",
            "LIGER": "Batch integration available through scOmni SinglecellSequencing.batch_integration(..., method=['liger']).",
            "scVI": "Batch integration available through scOmni SinglecellSequencing.batch_integration(..., method=['scvi']).",
            "CellMarker": f"Marker-based annotation backed by {sc_omni_root / 'maker_database' / 'Cell_marker_Seq.xlsx'}.",
            "ACT": f"Marker-based annotation backed by {sc_omni_root / 'maker_database' / 'ACT.csv'}.",
            "TrajectoryInference": f"Trajectory inference wrappers backed by {sc_omni_root / 'TI_R'}.",
        }

    def get_available_tools(self):
        return [{"name": name, "description": desc} for name, desc in self.tools.items()]

    def get_tools_docs(self, tool_names):
        return [{"name": name, "description": self.tools.get(name, "")} for name in tool_names]
