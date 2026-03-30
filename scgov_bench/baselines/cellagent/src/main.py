#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/scgov_bench_numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/scgov_bench_mpl")

import anndata as ad
import numpy as np
import scanpy as sc


SCGOV_BENCH_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = SCGOV_BENCH_ROOT / "results" / "cellagent"


def _safe_dense_sample(x: Any, n: int = 256) -> np.ndarray:
    sample = x[: min(n, x.shape[0])]
    return sample.toarray() if hasattr(sample, "toarray") else np.asarray(sample)


def extract_state(adata: ad.AnnData) -> dict[str, Any]:
    cluster_col = None
    for candidate in ("leiden", "louvain"):
        if candidate in adata.obs.columns:
            cluster_col = candidate
            break

    batch_cols = [
        col
        for col in ("batch", "donor_id", "sample", "donor", "patient")
        if col in adata.obs.columns
    ]
    x_dense = _safe_dense_sample(adata.X)
    hvg_count = 0
    if "highly_variable" in adata.var.columns:
        try:
            hvg_count = int(np.asarray(adata.var["highly_variable"]).sum())
        except Exception:
            hvg_count = 0

    neighbors_meta = adata.uns.get("neighbors", {}) if isinstance(adata.uns, dict) else {}
    params = neighbors_meta.get("params", {}) if isinstance(neighbors_meta, dict) else {}

    return {
        "stage": adata.uns.get("scgov_bench", {}).get("stage"),
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "has_qc_metrics": "n_genes_by_counts" in adata.obs.columns,
        "is_normalized": "log1p" in adata.uns,
        "has_raw_layer": adata.raw is not None or "raw_counts" in adata.layers,
        "has_hvg": "highly_variable" in adata.var.columns,
        "n_hvg": hvg_count,
        "has_pca": "X_pca" in adata.obsm,
        "n_pcs": int(adata.obsm["X_pca"].shape[1]) if "X_pca" in adata.obsm else 0,
        "has_neighbors": "connectivities" in adata.obsp,
        "n_neighbors": params.get("n_neighbors"),
        "has_clusters": cluster_col is not None,
        "cluster_col": cluster_col,
        "n_clusters": int(adata.obs[cluster_col].nunique()) if cluster_col else 0,
        "has_annotation": "cell_type" in adata.obs.columns,
        "has_batch": bool(batch_cols),
        "n_batches": int(adata.obs[batch_cols[0]].nunique()) if batch_cols else 1,
        "max_value": float(np.max(x_dense)) if x_dense.size else 0.0,
        "mean_value": float(np.mean(x_dense)) if x_dense.size else 0.0,
    }


def parse_instruction(instruction: str) -> str:
    text = instruction.lower()
    if "continue the analysis" in text or "continue from here" in text or text.startswith("continue."):
        return "continue_pipeline"
    if "fix it" in text or "check what happened" in text or "investigate" in text:
        return "diagnose_and_repair"
    if "normalize" in text:
        return "normalize"
    if "highly variable" in text or "hvg" in text:
        return "highly_variable_genes"
    if "pca" in text:
        return "pca"
    if "neighbor graph" in text or "neighbour graph" in text or "neighbors" in text:
        return "neighbors"
    if "cluster" in text:
        return "leiden"
    if "marker genes" in text:
        return "differential_expression"
    if "annotate" in text or "cell type" in text:
        return "annotate"
    if "mitochond" in text:
        return "filter_mito"
    if "low-quality" in text or "filter out" in text:
        return "filter_cells"
    if "redo the analysis from scratch" in text:
        return "reset_pipeline"
    return "unknown"


def detect_anomalies(adata: ad.AnnData, state: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if state["is_normalized"] and state["max_value"] > 25:
        issues.append("metadata says log1p exists, but expression values still look like raw counts")
    if state["is_normalized"] and state["mean_value"] < 0.4 and state["max_value"] < 4:
        issues.append("expression values look over-compressed, possibly from double normalization")
    if state["has_hvg"] and 0 < state["n_hvg"] < 200:
        issues.append(f"only {state['n_hvg']} highly variable genes are selected")
    if state["has_pca"] and not state["has_hvg"]:
        issues.append("PCA exists but no highly variable gene selection is recorded")
    if state["has_neighbors"] and state["n_neighbors"] is not None and state["n_neighbors"] <= 3:
        issues.append(f"neighbor graph was built with only {state['n_neighbors']} neighbors")
    if state["has_clusters"] and state["n_clusters"] > max(25, state["n_cells"] // 100):
        issues.append(f"there are {state['n_clusters']} clusters, which looks unusually high")
    if state["is_normalized"] and not state["has_raw_layer"]:
        issues.append("normalized data has no raw-count backup layer")
    if state["has_qc_metrics"] and state["n_cells"] < 1000:
        issues.append("very few cells remain after QC, which suggests over-filtering")
    return issues


@dataclass
class PlanStep:
    description: str
    tool_name: str
    params: dict[str, Any] = field(default_factory=dict)


class Planner:
    def plan(self, instruction: str, state: dict[str, Any]) -> tuple[list[PlanStep], list[str]]:
        action = parse_instruction(instruction)
        warnings: list[str] = []

        def warn(msg: str) -> tuple[list[PlanStep], list[str]]:
            return [], [msg]

        if action == "normalize":
            if state["is_normalized"]:
                return warn("Normalization appears to have already been completed.")
            if not state["has_qc_metrics"] or state["stage"] == "S0":
                return warn("Normalization should not be the first step; QC and filtering are missing.")
            return [PlanStep("Normalize counts and log-transform the matrix.", "normalize", {"target_sum": 1e4})], warnings

        if action == "highly_variable_genes":
            if state["has_hvg"]:
                return warn("Highly variable genes have already been selected.")
            if not state["is_normalized"]:
                return warn("HVG selection requires normalized data.")
            return [PlanStep("Select highly variable genes.", "highly_variable_genes", {"n_top_genes": 2000})], warnings

        if action == "pca":
            if state["has_pca"]:
                return warn("PCA is already present.")
            if not state["has_hvg"]:
                return warn("PCA should follow highly variable gene selection.")
            return [PlanStep("Compute PCA on the current matrix.", "pca", {"n_comps": 50})], warnings

        if action == "neighbors":
            if state["has_neighbors"]:
                return warn("The neighbor graph already exists.")
            if not state["has_pca"]:
                return warn("The neighbor graph requires PCA first.")
            return [PlanStep("Build the neighbor graph.", "neighbors", {"n_neighbors": 15, "n_pcs": 40})], warnings

        if action == "leiden":
            if state["has_clusters"]:
                return warn("Clustering already exists.")
            if not state["has_neighbors"]:
                return warn("Clustering should not run before the neighbor graph exists.")
            return [PlanStep("Cluster cells with Leiden.", "leiden", {"resolution": 1.0})], warnings

        if action == "differential_expression":
            if not state["has_clusters"]:
                return warn("Marker-gene analysis requires clusters.")
            return [PlanStep("Find marker genes for each cluster.", "differential_expression", {"groupby": state["cluster_col"] or "leiden"})], warnings

        if action == "annotate":
            if state["has_annotation"]:
                return warn("Cell type annotations are already attached.")
            if not state["has_clusters"]:
                return warn("Cell type annotation should follow clustering.")
            return [PlanStep("Annotate cell types from cluster structure.", "annotate", {"cluster_col": state["cluster_col"] or "leiden"})], warnings

        if action == "filter_mito":
            if state["has_clusters"]:
                return warn("Removing cells late in the workflow is destructive; confirm before proceeding.")
            return [PlanStep("Filter high-mitochondrial cells.", "filter_mito", {"max_pct_mito": 20.0})], warnings

        if action == "filter_cells":
            if state["stage"] not in ("S0", "S1"):
                return warn("Filtering appears to have already happened, so repeating it is risky.")
            return [PlanStep("Filter low-quality cells and genes.", "filter_cells", {"min_genes": 200, "min_cells": 3})], warnings

        if action == "reset_pipeline":
            if state["stage"] in ("S7", "S8"):
                return warn("Redoing the analysis from scratch would discard substantial downstream work.")
            return [PlanStep("Reset analysis state to the raw object.", "reset_pipeline", {})], warnings

        if action in {"continue_pipeline", "diagnose_and_repair"}:
            anomalies = detect_anomalies_placeholder(state)
            if anomalies:
                warnings.extend(anomalies)
            if anomalies and action == "continue_pipeline":
                return [], warnings
            steps = self._continuation_steps(state, diagnose=(action == "diagnose_and_repair"), anomalies=anomalies)
            return steps, warnings

        return warn("The request does not map cleanly onto a supported CellAgent baseline action.")

    def _continuation_steps(self, state: dict[str, Any], diagnose: bool, anomalies: list[str]) -> list[PlanStep]:
        if diagnose and anomalies:
            return build_repair_steps(state, anomalies)

        steps: list[PlanStep] = []
        if not state["is_normalized"]:
            steps.append(PlanStep("Normalize counts and log-transform.", "normalize", {"target_sum": 1e4}))
        if not state["has_hvg"]:
            steps.append(PlanStep("Select highly variable genes.", "highly_variable_genes", {"n_top_genes": 2000}))
        if not state["has_pca"]:
            steps.append(PlanStep("Compute PCA.", "pca", {"n_comps": 50}))
        if not state["has_neighbors"]:
            steps.append(PlanStep("Build the neighbor graph.", "neighbors", {"n_neighbors": 15, "n_pcs": 40}))
        if not state["has_clusters"]:
            steps.append(PlanStep("Run Leiden clustering.", "leiden", {"resolution": 1.0}))
        if not state["has_annotation"]:
            steps.append(PlanStep("Attach provisional cell type labels.", "annotate", {"cluster_col": state["cluster_col"] or "leiden"}))
        return steps


def detect_anomalies_placeholder(state: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if state["is_normalized"] and state["max_value"] > 25:
        issues.append("The object claims to be normalized, but the values still look like raw counts.")
    if state["is_normalized"] and state["mean_value"] < 0.4 and state["max_value"] < 4:
        issues.append("The values look too compressed, which suggests the data may have been normalized twice.")
    if state["has_hvg"] and 0 < state["n_hvg"] < 200:
        issues.append("The number of highly variable genes is implausibly small.")
    if state["has_pca"] and not state["has_hvg"]:
        issues.append("PCA is present even though no highly variable genes were selected.")
    if state["has_neighbors"] and state["n_neighbors"] is not None and state["n_neighbors"] <= 3:
        issues.append("The neighbor graph looks too sparse.")
    if state["has_clusters"] and state["n_clusters"] > max(25, state["n_cells"] // 100):
        issues.append("The clustering looks over-resolved.")
    if state["is_normalized"] and not state["has_raw_layer"]:
        issues.append("The raw backup layer is missing.")
    if state["has_qc_metrics"] and state["n_cells"] < 1000:
        issues.append("Too few cells remain, which suggests overly aggressive filtering.")
    return issues


def build_repair_steps(state: dict[str, Any], anomalies: list[str]) -> list[PlanStep]:
    steps: list[PlanStep] = []
    anomaly_text = " ".join(anomalies).lower()
    if "raw counts" in anomaly_text:
        steps.append(PlanStep("Normalize counts and log-transform properly.", "normalize", {"target_sum": 1e4, "force": True}))
    if "normalized twice" in anomaly_text:
        steps.append(PlanStep("Reload from the raw layer and normalize once.", "repair_from_raw", {"target_sum": 1e4}))
    if "highly variable genes" in anomaly_text:
        steps.append(PlanStep("Re-select a standard number of highly variable genes.", "highly_variable_genes", {"n_top_genes": 2000, "force": True}))
    if "no highly variable genes" in anomaly_text:
        steps.append(PlanStep("Select highly variable genes before PCA.", "highly_variable_genes", {"n_top_genes": 2000, "force": True}))
        steps.append(PlanStep("Recompute PCA after HVG selection.", "pca", {"n_comps": 50, "force": True}))
    if "neighbor graph" in anomaly_text:
        steps.append(PlanStep("Rebuild the neighbor graph with a standard neighborhood size.", "neighbors", {"n_neighbors": 15, "n_pcs": 40, "force": True}))
    if "over-resolved" in anomaly_text:
        steps.append(PlanStep("Re-cluster with a lower Leiden resolution.", "leiden", {"resolution": 1.0, "force": True}))
    return steps


class Executor:
    def execute(self, adata: ad.AnnData, step: PlanStep) -> dict[str, Any]:
        if step.tool_name == "normalize":
            if step.params.get("force") and adata.raw is not None:
                adata = adata.raw.to_adata()
            sc.pp.normalize_total(adata, target_sum=float(step.params.get("target_sum", 1e4)))
            sc.pp.log1p(adata)
        elif step.tool_name == "highly_variable_genes":
            sc.pp.highly_variable_genes(adata, n_top_genes=int(step.params.get("n_top_genes", 2000)))
        elif step.tool_name == "pca":
            sc.tl.pca(adata, n_comps=int(step.params.get("n_comps", 50)), use_highly_variable=bool("highly_variable" in adata.var.columns))
        elif step.tool_name == "neighbors":
            sc.pp.neighbors(
                adata,
                n_neighbors=int(step.params.get("n_neighbors", 15)),
                n_pcs=int(step.params.get("n_pcs", 40)),
            )
        elif step.tool_name == "leiden":
            sc.tl.leiden(adata, resolution=float(step.params.get("resolution", 1.0)))
        elif step.tool_name == "differential_expression":
            sc.tl.rank_genes_groups(adata, groupby=step.params.get("groupby", "leiden"), method="wilcoxon")
        elif step.tool_name == "annotate":
            cluster_col = step.params.get("cluster_col", "leiden")
            if cluster_col not in adata.obs:
                raise ValueError(f"Missing cluster column: {cluster_col}")
            adata.obs["cell_type_pred"] = adata.obs[cluster_col].astype(str).map(lambda x: f"cluster_{x}")
        elif step.tool_name == "filter_mito":
            mask = adata.obs["pct_counts_mt"] < float(step.params.get("max_pct_mito", 20.0))
            adata._inplace_subset_obs(mask)
        elif step.tool_name == "filter_cells":
            sc.pp.filter_cells(adata, min_genes=int(step.params.get("min_genes", 200)))
            sc.pp.filter_genes(adata, min_cells=int(step.params.get("min_cells", 3)))
        elif step.tool_name == "reset_pipeline":
            if adata.raw is None:
                raise ValueError("Cannot reset without a raw layer.")
            adata = adata.raw.to_adata()
        elif step.tool_name == "repair_from_raw":
            if adata.raw is None:
                raise ValueError("Cannot repair from raw because the raw layer is missing.")
            adata = adata.raw.to_adata()
            sc.pp.normalize_total(adata, target_sum=float(step.params.get("target_sum", 1e4)))
            sc.pp.log1p(adata)
        else:
            raise ValueError(f"Unsupported tool: {step.tool_name}")

        return {
            "tool_name": step.tool_name,
            "params": step.params,
            "post_state": extract_state(adata),
            "adata": adata,
        }


class Evaluator:
    def summarize(self, warnings: list[str], tool_results: list[dict[str, Any]]) -> str:
        if warnings and not tool_results:
            return "warn_only"
        if warnings and tool_results:
            return "warn_then_execute"
        if tool_results:
            return "execute"
        return "refuse"


class GlobalMemory:
    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []

    def add(self, record: dict[str, Any]) -> None:
        self.entries.append(record)


class MergedCellAgent:
    def __init__(self) -> None:
        self.planner = Planner()
        self.executor = Executor()
        self.evaluator = Evaluator()
        self.memory = GlobalMemory()

    def run(self, case_id: str, instruction: str, snapshot_path: Path, output_path: Path | None = None) -> dict[str, Any]:
        adata = sc.read_h5ad(snapshot_path)
        pre_state = extract_state(adata)
        planned_steps, warnings = self.planner.plan(instruction, pre_state)

        trace_steps: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []

        if warnings:
            trace_steps.append(
                {
                    "step_num": 1,
                    "reasoning": " ".join(warnings),
                    "tool_calls": [{"tool_name": "ask_user", "params": {"message": msg}} for msg in warnings[:1]],
                }
            )

        for idx, plan_step in enumerate(planned_steps, start=len(trace_steps) + 1):
            reasoning = plan_step.description
            result = self.executor.execute(adata, plan_step)
            adata = result.pop("adata")
            tool_results.append(result)
            self.memory.add(result)
            trace_steps.append(
                {
                    "step_num": idx,
                    "reasoning": reasoning,
                    "tool_calls": [{"tool_name": result["tool_name"], "params": result["params"]}],
                }
            )

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            adata.write(output_path)

        final_state = extract_state(adata)
        action = self.evaluator.summarize(warnings, tool_results)
        return {
            "case_id": case_id,
            "agent": "cellagent",
            "model": "heuristic-runnable-baseline",
            "snapshot_path": str(snapshot_path),
            "output_path": str(output_path) if output_path else None,
            "action": action,
            "pre_state": pre_state,
            "final_state": final_state,
            "trace": {"steps": trace_steps},
        }


def load_case(case_file: Path, case_id: str) -> dict[str, Any]:
    with case_file.open() as handle:
        for line in handle:
            record = json.loads(line)
            if record["case_id"] == case_id:
                return record
    raise ValueError(f"Case not found: {case_id}")


def build_output_path(output_dir: Path, case_id: str) -> Path:
    return output_dir / f"{case_id}.h5ad"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runnable merged CellAgent baseline for scGov-Bench.")
    parser.add_argument("--case-file", type=Path, help="JSONL case file generated by scGov-Bench.")
    parser.add_argument("--case-id", required=True, help="Case identifier to execute.")
    parser.add_argument("--snapshot-path", type=Path, help="Path to the starting snapshot.")
    parser.add_argument("--instruction", help="User instruction for the case.")
    parser.add_argument("--output-json", type=Path, required=True, help="Where to save the trace JSON.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Where to save modified AnnData snapshots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.case_file:
        case = load_case(args.case_file, args.case_id)
        snapshot_path = Path(case["snapshot_path"])
        instruction = case["instruction"]
    else:
        if args.snapshot_path is None or args.instruction is None:
            raise ValueError("Without --case-file you must pass both --snapshot-path and --instruction.")
        snapshot_path = args.snapshot_path
        instruction = args.instruction

    output_path = build_output_path(args.output_dir, args.case_id)
    result = MergedCellAgent().run(
        case_id=args.case_id,
        instruction=instruction,
        snapshot_path=snapshot_path,
        output_path=output_path,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w") as handle:
        json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
