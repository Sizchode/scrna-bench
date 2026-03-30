#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DATASETS_CONFIG = ROOT / "config" / "datasets.yaml"
CASE_MATRIX_CONFIG = ROOT / "config" / "case_matrix.yaml"
CASES_ROOT = ROOT / "cases"
SNAPSHOT_ROOT = ROOT / "data" / "snapshots"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate D1-D3 JSONL case files for scGov-Bench.")
    parser.add_argument(
        "--dataset-key",
        action="append",
        dest="dataset_keys",
        help="Specific dataset key(s) to process. Defaults to all datasets in datasets.yaml.",
    )
    parser.add_argument(
        "--cases-root",
        type=Path,
        default=CASES_ROOT,
        help="Output directory for JSONL case files.",
    )
    parser.add_argument(
        "--snapshot-root",
        type=Path,
        default=SNAPSHOT_ROOT,
        help="Root snapshot directory that contains clean and corrupted snapshots.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return yaml.safe_load(handle)


def dump_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def ensure_snapshot_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Referenced snapshot does not exist: {path}")


def generate_d1_records(
    *,
    dataset_keys: list[str],
    snapshot_root: Path,
    d1_specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for dataset_key in dataset_keys:
        for spec in d1_specs:
            pair_id = f"{dataset_key}_{spec['id']}"
            state_a = spec["state_a"]
            state_b = spec["state_b"]
            path_a = snapshot_root / dataset_key / f"{state_a}.h5ad"
            path_b = snapshot_root / dataset_key / f"{state_b}.h5ad"
            ensure_snapshot_exists(path_a)
            ensure_snapshot_exists(path_b)

            common = {
                "dimension": "D1",
                "dataset_key": dataset_key,
                "pair_id": pair_id,
                "category": spec["family"],
                "instruction": spec["instruction"],
                "what_it_tests": spec["what_it_tests"],
            }
            records.append(
                {
                    **common,
                    "case_id": f"{pair_id}_A",
                    "pair_member": "A",
                    "snapshot_stage": state_a,
                    "snapshot_path": str(path_a),
                    "counterfactual_stage": state_b,
                    "expected_action": spec["expected_a"],
                }
            )
            records.append(
                {
                    **common,
                    "case_id": f"{pair_id}_B",
                    "pair_member": "B",
                    "snapshot_stage": state_b,
                    "snapshot_path": str(path_b),
                    "counterfactual_stage": state_a,
                    "expected_action": spec["expected_b"],
                }
            )
    return records


def generate_d2_records(
    *,
    dataset_keys: list[str],
    snapshot_root: Path,
    d2_specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for dataset_key in dataset_keys:
        for spec in d2_specs:
            error_suffix = spec["id"].split("_", 1)[-1]
            corruption_type = spec["corruption"]
            corrupted_path = snapshot_root / dataset_key / "corrupted" / f"{error_suffix}_{corruption_type}.h5ad"
            control_path = snapshot_root / dataset_key / f"{spec['clean_control_snapshot']}.h5ad"
            ensure_snapshot_exists(corrupted_path)
            ensure_snapshot_exists(control_path)

            base = {
                "dimension": "D2",
                "dataset_key": dataset_key,
                "error_id": error_suffix,
                "error_type": corruption_type,
                "built_from": spec["built_from"],
                "instruction": spec["continuation_instruction"],
            }
            records.append(
                {
                    **base,
                    "case_id": f"{dataset_key}_{spec['id']}",
                    "snapshot_path": str(corrupted_path),
                    "snapshot_kind": "corrupted",
                    "is_corrupted": True,
                    "expected": spec["expected_corrupted"],
                }
            )
            records.append(
                {
                    **base,
                    "case_id": f"{dataset_key}_{spec['id']}_control",
                    "snapshot_path": str(control_path),
                    "snapshot_kind": "clean_control",
                    "is_corrupted": False,
                    "expected": spec["expected_control"],
                }
            )
    return records


def generate_d3_records(
    *,
    dataset_keys: list[str],
    snapshot_root: Path,
    d3_specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for dataset_key in dataset_keys:
        for spec in d3_specs:
            error_suffix = spec["id"].split("_", 1)[-1]
            corruption_type = spec["corruption"]
            corrupted_path = snapshot_root / dataset_key / "corrupted" / f"{error_suffix}_{corruption_type}.h5ad"
            control_path = snapshot_root / dataset_key / f"{spec['clean_control_snapshot']}.h5ad"
            ensure_snapshot_exists(corrupted_path)
            ensure_snapshot_exists(control_path)

            base = {
                "dimension": "D3",
                "dataset_key": dataset_key,
                "error_id": error_suffix,
                "error_type": corruption_type,
                "instruction": spec["instruction"],
            }
            records.append(
                {
                    **base,
                    "case_id": f"{dataset_key}_{spec['id']}",
                    "snapshot_path": str(corrupted_path),
                    "snapshot_kind": "corrupted",
                    "is_corrupted": True,
                    "expected_diagnosis": spec["expected_diagnosis"],
                    "expected_repair": spec["expected_repair"],
                }
            )
            records.append(
                {
                    **base,
                    "case_id": f"{dataset_key}_{spec['id']}_control",
                    "snapshot_path": str(control_path),
                    "snapshot_kind": "clean_control",
                    "is_corrupted": False,
                    "expected_diagnosis": "nothing wrong",
                    "expected_repair": "none needed",
                }
            )
    return records


def validate_counts(
    *,
    dataset_count: int,
    d1_records: list[dict[str, Any]],
    d2_records: list[dict[str, Any]],
    d3_records: list[dict[str, Any]],
    case_counts: dict[str, Any],
) -> dict[str, Any]:
    expected = case_counts["per_dataset"]
    summary = {
        "datasets": dataset_count,
        "dim1_runs": len(d1_records),
        "dim2_cases": len(d2_records),
        "dim3_cases": len(d3_records),
    }
    if len(d1_records) != dataset_count * expected["D1_runs"]:
        raise ValueError("D1 record count does not match case_matrix.yaml.")
    if len(d2_records) != dataset_count * expected["D2_cases"]:
        raise ValueError("D2 record count does not match case_matrix.yaml.")
    if len(d3_records) != dataset_count * expected["D3_cases"]:
        raise ValueError("D3 record count does not match case_matrix.yaml.")
    return summary


def main() -> None:
    args = parse_args()
    datasets_config = load_yaml(DATASETS_CONFIG)
    case_matrix = load_yaml(CASE_MATRIX_CONFIG)

    wanted = set(args.dataset_keys or [])
    dataset_keys = [
        dataset["key"]
        for dataset in datasets_config["datasets"]
        if not wanted or dataset["key"] in wanted
    ]
    if not dataset_keys:
        raise SystemExit("No datasets selected.")

    d1_records = generate_d1_records(
        dataset_keys=dataset_keys,
        snapshot_root=args.snapshot_root,
        d1_specs=case_matrix["D1_state_sensitivity"]["per_dataset_pairs"],
    )
    d2_records = generate_d2_records(
        dataset_keys=dataset_keys,
        snapshot_root=args.snapshot_root,
        d2_specs=case_matrix["D2_error_propagation"]["per_dataset_cases"],
    )
    d3_records = generate_d3_records(
        dataset_keys=dataset_keys,
        snapshot_root=args.snapshot_root,
        d3_specs=case_matrix["D3_state_recovery"]["per_dataset_cases"],
    )

    summary = validate_counts(
        dataset_count=len(dataset_keys),
        d1_records=d1_records,
        d2_records=d2_records,
        d3_records=d3_records,
        case_counts=case_matrix["case_counts"],
    )

    dump_jsonl(args.cases_root / "dim1_sensitivity.jsonl", d1_records)
    dump_jsonl(args.cases_root / "dim2_propagation.jsonl", d2_records)
    dump_jsonl(args.cases_root / "dim3_recovery.jsonl", d3_records)
    (args.cases_root / "manifest.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
