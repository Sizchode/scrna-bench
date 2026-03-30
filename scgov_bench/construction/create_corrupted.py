#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

# These defaults avoid known cache-path issues in the local scanpy environment.
os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    str(Path(tempfile.gettempdir()) / "scgov_bench_numba_cache"),
)
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "scgov_bench_mpl"),
)

import anndata as ad
import scanpy as sc


ROOT = Path(__file__).resolve().parents[1]
DATASETS_CONFIG = ROOT / "config" / "datasets.yaml"
CASE_MATRIX_CONFIG = ROOT / "config" / "case_matrix.yaml"
SNAPSHOT_ROOT = ROOT / "data" / "snapshots"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate corrupted snapshots for scGov-Bench.")
    parser.add_argument(
        "--dataset-key",
        action="append",
        dest="dataset_keys",
        help="Specific dataset key(s) to process. Defaults to all datasets in datasets.yaml.",
    )
    parser.add_argument(
        "--snapshot-root",
        type=Path,
        default=SNAPSHOT_ROOT,
        help="Root snapshot directory that already contains clean S0-S8 files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing corrupted snapshot files.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return yaml.safe_load(handle)


def safe_n_comps(adata: ad.AnnData) -> int:
    if adata.n_vars <= 2 or adata.n_obs <= 2:
        return 2
    return max(2, min(50, adata.n_vars - 1, adata.n_obs - 1))


def safe_n_pcs(adata: ad.AnnData) -> int:
    if "X_pca" not in adata.obsm:
        return safe_n_comps(adata)
    return max(2, min(40, adata.obsm["X_pca"].shape[1]))


def set_corruption_metadata(
    adata: ad.AnnData,
    *,
    dataset_key: str,
    error_id: str,
    corruption_type: str,
    base_stage: str,
    source_snapshot_path: str,
    notes: str,
) -> None:
    meta = dict(adata.uns.get("scgov_bench", {}))
    meta.update(
        {
            "dataset_key": dataset_key,
            "stage": error_id,
            "base_stage": base_stage,
            "is_corrupted": True,
            "corruption_id": error_id,
            "corruption_type": corruption_type,
            "source_snapshot_path": source_snapshot_path,
            "notes": notes,
        }
    )
    adata.uns["scgov_bench"] = meta


def apply_corruption(
    adata: ad.AnnData,
    *,
    corruption_type: str,
) -> tuple[ad.AnnData, str]:
    corrupted = adata.copy()

    if corruption_type == "aggressive_filter":
        if "n_genes_by_counts" not in corrupted.obs.columns:
            raise KeyError("S1 snapshot is missing n_genes_by_counts for aggressive_filter.")
        threshold = float(corrupted.obs["n_genes_by_counts"].quantile(0.7))
        corrupted = corrupted[corrupted.obs["n_genes_by_counts"] > threshold, :].copy()
        sc.pp.filter_genes(corrupted, min_cells=3)
        notes = f"aggressive filtering applied at n_genes_by_counts > {threshold:.2f}"
        return corrupted, notes

    if corruption_type == "fake_normalization":
        corrupted.uns["log1p"] = {"base": None}
        return corrupted, "normalization metadata inserted without modifying raw count matrix"

    if corruption_type == "double_normalization":
        sc.pp.normalize_total(corrupted, target_sum=1e4)
        sc.pp.log1p(corrupted)
        return corrupted, "normalize_total and log1p applied a second time"

    if corruption_type == "too_few_hvg":
        sc.pp.highly_variable_genes(corrupted, n_top_genes=min(50, corrupted.n_vars), flavor="seurat")
        return corrupted, "highly variable genes reduced to a very small set"

    if corruption_type == "missing_raw":
        corrupted.raw = None
        for layer_name in ("raw_counts", "counts"):
            if layer_name in corrupted.layers:
                del corrupted.layers[layer_name]
        return corrupted, "raw backup removed from snapshot"

    if corruption_type == "wrong_neighbors":
        sc.pp.neighbors(
            corrupted,
            n_neighbors=2,
            n_pcs=safe_n_pcs(corrupted),
        )
        return corrupted, "neighbor graph rebuilt with n_neighbors=2"

    if corruption_type == "extreme_resolution":
        sc.tl.leiden(corrupted, resolution=10.0)
        return corrupted, "clustering rerun with extreme resolution=10.0"

    if corruption_type == "pca_all_genes":
        sc.tl.pca(
            corrupted,
            n_comps=safe_n_comps(corrupted),
            mask_var=None,
            svd_solver="arpack",
        )
        return corrupted, "PCA computed on all genes without HVG gating"

    raise ValueError(f"Unknown corruption_type: {corruption_type}")


def create_corrupted_snapshots(
    *,
    dataset_key: str,
    snapshot_root: Path,
    specs: list[dict[str, Any]],
    overwrite: bool,
) -> list[Path]:
    dataset_dir = snapshot_root / dataset_key
    corrupted_dir = dataset_dir / "corrupted"
    corrupted_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for spec in specs:
        error_id = spec["id"].split("_", 1)[-1]
        corruption_type = spec["corruption"]
        base_stage = spec["built_from"]
        source_snapshot = dataset_dir / f"{base_stage}.h5ad"
        target = corrupted_dir / f"{error_id}_{corruption_type}.h5ad"

        if target.exists() and not overwrite:
            written.append(target)
            continue
        if not source_snapshot.exists():
            raise FileNotFoundError(f"Missing clean snapshot for corruption build: {source_snapshot}")

        adata = ad.read_h5ad(source_snapshot)
        corrupted, notes = apply_corruption(adata, corruption_type=corruption_type)
        set_corruption_metadata(
            corrupted,
            dataset_key=dataset_key,
            error_id=error_id,
            corruption_type=corruption_type,
            base_stage=base_stage,
            source_snapshot_path=str(source_snapshot),
            notes=notes,
        )
        corrupted.write_h5ad(target)
        written.append(target)

    return written


def main() -> None:
    args = parse_args()
    datasets_config = load_yaml(DATASETS_CONFIG)
    case_matrix = load_yaml(CASE_MATRIX_CONFIG)
    d2_specs = case_matrix["D2_error_propagation"]["per_dataset_cases"]

    wanted = set(args.dataset_keys or [])
    selected = [
        dataset["key"]
        for dataset in datasets_config["datasets"]
        if not wanted or dataset["key"] in wanted
    ]
    if not selected:
        raise SystemExit("No datasets selected.")

    summary: dict[str, list[str]] = {}
    for dataset_key in selected:
        written = create_corrupted_snapshots(
            dataset_key=dataset_key,
            snapshot_root=args.snapshot_root,
            specs=d2_specs,
            overwrite=args.overwrite,
        )
        summary[dataset_key] = [str(path) for path in written]

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
