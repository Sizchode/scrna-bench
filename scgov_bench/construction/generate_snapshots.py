#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import sparse

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
BENCHMARK_CONFIG = ROOT / "config" / "benchmark.yaml"
OUTPUT_ROOT = ROOT / "data" / "snapshots"

QC_DROP_PATTERNS = (
    "ncount",
    "nfeature",
    "pct_",
    "percent",
    "mito",
    "cluster",
    "annotation",
    "rank",
    "louvain",
    "leiden",
)
UNS_KEEP_KEYS = {
    "citation",
    "organism",
    "organism_ontology_term_id",
    "schema_reference",
    "schema_version",
    "title",
}
GOLD_COLUMNS = ("cell_type", "cell_type_ontology_term_id")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate S0-S8 snapshots for scGov-Bench.")
    parser.add_argument(
        "--dataset-key",
        action="append",
        dest="dataset_keys",
        help="Specific dataset key(s) to process. Defaults to all datasets in datasets.yaml.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Root folder where per-dataset snapshot directories will be created.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing snapshot files.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return yaml.safe_load(handle)


def sample_is_integer(matrix: Any, sample_rows: int = 128, sample_cols: int = 512) -> bool:
    if matrix is None:
        return False
    view = matrix[:sample_rows, :sample_cols]
    if sparse.issparse(view):
        arr = view.toarray()
    else:
        arr = np.asarray(view)
    if arr.size == 0:
        return True
    return bool(np.allclose(arr, np.round(arr)))


def pick_counts_source(adata: ad.AnnData) -> tuple[ad.AnnData, str]:
    if adata.raw is not None and sample_is_integer(adata.raw.X):
        counts = adata.raw.to_adata()
        return counts.copy(), "raw"

    for layer_name in ("raw_counts", "counts"):
        if layer_name in adata.layers and sample_is_integer(adata.layers[layer_name]):
            counts = adata.copy()
            counts.X = counts.layers[layer_name].copy()
            return counts, f"layer:{layer_name}"

    counts = adata.copy()
    return counts, "X"


def as_text_list(values: pd.Series) -> list[str]:
    return values.astype(str).fillna("").tolist()


def extract_hidden_gold(adata: ad.AnnData) -> dict[str, dict[str, str]]:
    gold_maps: dict[str, dict[str, str]] = {}
    for column in GOLD_COLUMNS:
        if column in adata.obs.columns:
            gold_maps[column] = adata.obs[column].astype(str).to_dict()
            del adata.obs[column]
    return gold_maps


def attach_hidden_gold(adata: ad.AnnData, gold_maps: dict[str, dict[str, str]]) -> None:
    payload: dict[str, Any] = {"obs_names": list(map(str, adata.obs_names))}
    for column, mapping in gold_maps.items():
        payload[column] = [mapping.get(str(obs_name), "") for obs_name in adata.obs_names]
    adata.uns["scgov_gold"] = payload


def reveal_visible_annotation(adata: ad.AnnData, gold_maps: dict[str, dict[str, str]]) -> None:
    for column, mapping in gold_maps.items():
        values = [mapping.get(str(obs_name), "") for obs_name in adata.obs_names]
        adata.obs[column] = pd.Categorical(values)


def prune_obs_columns(adata: ad.AnnData) -> None:
    to_drop: list[str] = []
    for column in adata.obs.columns:
        lower = column.lower()
        if any(token in lower for token in QC_DROP_PATTERNS):
            to_drop.append(column)
    for column in to_drop:
        del adata.obs[column]


def prune_var_columns(adata: ad.AnnData) -> None:
    to_drop = [
        column
        for column in adata.var.columns
        if column.lower() in {"highly_variable", "means", "dispersions", "dispersions_norm"}
    ]
    for column in to_drop:
        del adata.var[column]


def clean_precomputed_state(adata: ad.AnnData) -> None:
    adata.obsm.clear()
    adata.obsp.clear()
    adata.varm.clear()
    adata.varp.clear()
    adata.layers.clear()

    for key in list(adata.uns.keys()):
        if key not in UNS_KEEP_KEYS:
            del adata.uns[key]

    prune_obs_columns(adata)
    prune_var_columns(adata)


def pick_gene_symbols(adata: ad.AnnData) -> pd.Index:
    for column in ("feature_name", "gene_symbols", "gene_name"):
        if column in adata.var.columns:
            return pd.Index(adata.var[column].astype(str))
    return pd.Index(adata.var_names.astype(str))


def add_mt_flag(adata: ad.AnnData) -> None:
    gene_names = pick_gene_symbols(adata)
    adata.var["mt"] = gene_names.str.startswith("MT-") | gene_names.str.startswith("mt-")


def set_stage_metadata(
    adata: ad.AnnData,
    *,
    dataset_key: str,
    source_file: str,
    counts_source: str,
    stage: str,
    notes: str,
) -> None:
    adata.uns["scgov_bench"] = {
        "dataset_key": dataset_key,
        "source_file": source_file,
        "counts_source": counts_source,
        "stage": stage,
        "notes": notes,
    }


def prepare_s0(dataset_key: str, dataset_file: Path) -> tuple[ad.AnnData, dict[str, dict[str, str]], str]:
    original = ad.read_h5ad(dataset_file)
    counts_adata, counts_source = pick_counts_source(original)
    clean_precomputed_state(counts_adata)
    gold_maps = extract_hidden_gold(counts_adata)
    attach_hidden_gold(counts_adata, gold_maps)
    set_stage_metadata(
        counts_adata,
        dataset_key=dataset_key,
        source_file=str(dataset_file),
        counts_source=counts_source,
        stage="S0",
        notes="raw counts baseline with hidden gold labels removed from obs",
    )
    return counts_adata, gold_maps, counts_source


def write_stage(
    adata: ad.AnnData,
    *,
    dataset_key: str,
    output_dir: Path,
    stage: str,
    gold_maps: dict[str, dict[str, str]],
    visible_annotation: bool = False,
    overwrite: bool = False,
) -> Path:
    target = output_dir / f"{stage}.h5ad"
    if target.exists() and not overwrite:
        return target

    stage_adata = adata.copy()
    attach_hidden_gold(stage_adata, gold_maps)
    if visible_annotation:
        reveal_visible_annotation(stage_adata, gold_maps)

    target.parent.mkdir(parents=True, exist_ok=True)
    stage_adata.write_h5ad(target)
    return target


def safe_n_comps(adata: ad.AnnData) -> int:
    if adata.n_vars <= 2 or adata.n_obs <= 2:
        return 2
    return max(2, min(50, adata.n_vars - 1, adata.n_obs - 1))


def run_pipeline(
    *,
    dataset_key: str,
    dataset_file: Path,
    output_root: Path,
    overwrite: bool,
) -> list[Path]:
    output_dir = output_root / dataset_key
    s0, gold_maps, counts_source = prepare_s0(dataset_key, dataset_file)
    written: list[Path] = []

    written.append(
        write_stage(
            s0,
            dataset_key=dataset_key,
            output_dir=output_dir,
            stage="S0",
            gold_maps=gold_maps,
            overwrite=overwrite,
        )
    )

    s1 = s0.copy()
    add_mt_flag(s1)
    if bool(s1.var["mt"].sum()):
        sc.pp.calculate_qc_metrics(s1, qc_vars=["mt"], inplace=True, percent_top=None)
    else:
        sc.pp.calculate_qc_metrics(s1, inplace=True, percent_top=None)
    set_stage_metadata(
        s1,
        dataset_key=dataset_key,
        source_file=str(dataset_file),
        counts_source=counts_source,
        stage="S1",
        notes="QC metrics added",
    )
    written.append(
        write_stage(
            s1,
            dataset_key=dataset_key,
            output_dir=output_dir,
            stage="S1",
            gold_maps=gold_maps,
            overwrite=overwrite,
        )
    )

    s2 = s1.copy()
    sc.pp.filter_cells(s2, min_genes=200)
    sc.pp.filter_genes(s2, min_cells=3)
    if "pct_counts_mt" in s2.obs.columns:
        s2 = s2[s2.obs["pct_counts_mt"] < 20, :].copy()
    set_stage_metadata(
        s2,
        dataset_key=dataset_key,
        source_file=str(dataset_file),
        counts_source=counts_source,
        stage="S2",
        notes="cells and genes filtered",
    )
    written.append(
        write_stage(
            s2,
            dataset_key=dataset_key,
            output_dir=output_dir,
            stage="S2",
            gold_maps=gold_maps,
            overwrite=overwrite,
        )
    )

    s3 = s2.copy()
    s3.raw = s3.copy()
    sc.pp.normalize_total(s3, target_sum=1e4)
    sc.pp.log1p(s3)
    set_stage_metadata(
        s3,
        dataset_key=dataset_key,
        source_file=str(dataset_file),
        counts_source=counts_source,
        stage="S3",
        notes="normalized and log-transformed",
    )
    written.append(
        write_stage(
            s3,
            dataset_key=dataset_key,
            output_dir=output_dir,
            stage="S3",
            gold_maps=gold_maps,
            overwrite=overwrite,
        )
    )

    s4 = s3.copy()
    sc.pp.highly_variable_genes(s4, n_top_genes=min(2000, s4.n_vars), flavor="seurat")
    set_stage_metadata(
        s4,
        dataset_key=dataset_key,
        source_file=str(dataset_file),
        counts_source=counts_source,
        stage="S4",
        notes="highly variable genes selected",
    )
    written.append(
        write_stage(
            s4,
            dataset_key=dataset_key,
            output_dir=output_dir,
            stage="S4",
            gold_maps=gold_maps,
            overwrite=overwrite,
        )
    )

    s5 = s4.copy()
    sc.tl.pca(
        s5,
        n_comps=safe_n_comps(s5),
        use_highly_variable=True,
        svd_solver="arpack",
    )
    set_stage_metadata(
        s5,
        dataset_key=dataset_key,
        source_file=str(dataset_file),
        counts_source=counts_source,
        stage="S5",
        notes="PCA computed",
    )
    written.append(
        write_stage(
            s5,
            dataset_key=dataset_key,
            output_dir=output_dir,
            stage="S5",
            gold_maps=gold_maps,
            overwrite=overwrite,
        )
    )

    s6 = s5.copy()
    sc.pp.neighbors(
        s6,
        n_neighbors=max(2, min(15, s6.n_obs - 1)),
        n_pcs=max(2, min(40, s6.obsm["X_pca"].shape[1])),
    )
    set_stage_metadata(
        s6,
        dataset_key=dataset_key,
        source_file=str(dataset_file),
        counts_source=counts_source,
        stage="S6",
        notes="neighbor graph computed",
    )
    written.append(
        write_stage(
            s6,
            dataset_key=dataset_key,
            output_dir=output_dir,
            stage="S6",
            gold_maps=gold_maps,
            overwrite=overwrite,
        )
    )

    s7 = s6.copy()
    sc.tl.leiden(s7, resolution=1.0)
    set_stage_metadata(
        s7,
        dataset_key=dataset_key,
        source_file=str(dataset_file),
        counts_source=counts_source,
        stage="S7",
        notes="leiden clustering computed",
    )
    written.append(
        write_stage(
            s7,
            dataset_key=dataset_key,
            output_dir=output_dir,
            stage="S7",
            gold_maps=gold_maps,
            overwrite=overwrite,
        )
    )

    s8 = s7.copy()
    set_stage_metadata(
        s8,
        dataset_key=dataset_key,
        source_file=str(dataset_file),
        counts_source=counts_source,
        stage="S8",
        notes="visible annotation restored from hidden gold",
    )
    written.append(
        write_stage(
            s8,
            dataset_key=dataset_key,
            output_dir=output_dir,
            stage="S8",
            gold_maps=gold_maps,
            visible_annotation=True,
            overwrite=overwrite,
        )
    )

    return written


def main() -> None:
    args = parse_args()
    datasets_config = load_yaml(DATASETS_CONFIG)
    _ = load_yaml(BENCHMARK_CONFIG)

    wanted = set(args.dataset_keys or [])
    selected = [
        dataset
        for dataset in datasets_config["datasets"]
        if not wanted or dataset["key"] in wanted
    ]
    if not selected:
        raise SystemExit("No datasets selected.")

    summary: dict[str, list[str]] = {}
    for dataset in selected:
        dataset_key = dataset["key"]
        dataset_file = Path(dataset["local_file"])
        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
        written = run_pipeline(
            dataset_key=dataset_key,
            dataset_file=dataset_file,
            output_root=args.output_root,
            overwrite=args.overwrite,
        )
        summary[dataset_key] = [str(path) for path in written]

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
