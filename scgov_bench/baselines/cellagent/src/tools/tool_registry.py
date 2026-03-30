from __future__ import annotations

from pathlib import Path


class ToolRegistry:
    def __init__(self):
        sc_omni_root = Path(__file__).resolve().parents[2] / "scOmni" / "codes"
        scseq_path = sc_omni_root / "SinglecellSequencing.py"
        marker_db = sc_omni_root / "maker_database"
        ti_root = sc_omni_root / "TI_R"
        self.tools = {
            "scOmni.SinglecellSequencing_toolkit": (
                f"Primary scOmni toolkit defined in {scseq_path}. "
                "Prefer importing `SinglecellSequencing_toolkit` from this file when you need batch integration, "
                "cell type annotation, or trajectory inference utilities."
            ),
            "scOmni.batch_integration": (
                "Callable: SinglecellSequencing_toolkit().batch_integration(adata, batch_key, method=[...]). "
                "Supported methods include 'harmony', 'liger', and 'scvi'."
            ),
            "scOmni.celltype_annotation": (
                "Callable: SinglecellSequencing_toolkit().celltype_annotation("
                "adata, species=..., tissue_type=..., cancer_type=..., obs_cluster=..., method=[...]). "
                "Supported methods include 'gpt4', 'cellmarker', and 'act'."
            ),
            "scOmni.trajectory_top_k_methods": (
                f"Callable: SinglecellSequencing_toolkit().trajectory_top_k_methods(...). "
                f"R helpers are stored under {ti_root}."
            ),
            "scOmni.trajectory_inference": (
                f"Callable: SinglecellSequencing_toolkit().trajectory_inference(...). "
                f"Backed by the trajectory inference scripts in {ti_root}."
            ),
            "scOmni.CellMarkerDB": f"CellMarker annotation database located at {marker_db / 'Cell_marker_Seq.xlsx'}.",
            "scOmni.ACTDB": f"ACT annotation database located at {marker_db / 'ACT.csv'}.",
            "Scanpy": "Fallback general-purpose toolkit for single-cell gene expression analysis.",
        }

    def get_available_tools(self):
        return [{"name": name, "description": desc} for name, desc in self.tools.items()]

    def get_tools_docs(self, tool_names):
        return [{"name": name, "description": self.tools.get(name, "")} for name in tool_names]
