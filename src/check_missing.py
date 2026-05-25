from __future__ import annotations

import argparse
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import quapy as qp

from uncawe import env
from uncawe.config import get_acc_names, get_filtered_datasets, get_method_names
from uncawe.pretrain.data import PretrainInfo, load_info_paths

DEFAULT_DOMAIN = "classic"
DEFAULT_EXPERIMENT = "main"
DEFAULT_SAMPLE_SIZE = 1000

qp.environ["SAMPLE_SIZE"] = DEFAULT_SAMPLE_SIZE


@dataclass(frozen=True, order=True)
class ExpectedResult:
    domain: str
    acc_name: str
    dataset: str
    classifier: str
    method: str
    path: Path


def result_path(
    output_root: Path,
    experiment: str,
    domain: str,
    dataset: str,
    classifier: str,
    method: str,
    acc_name: str,
) -> Path:
    return output_root / experiment / domain / acc_name / dataset / classifier / f"{method}.parquet"


def expected_results_from_pretrain(domain: str, output_root: Path, experiment: str) -> list[ExpectedResult]:
    filtered_datasets = set(get_filtered_datasets())
    method_names = get_method_names()
    acc_names = get_acc_names()

    expected = []
    for info_path in load_info_paths(domain=domain):
        p_info = PretrainInfo.load(info_path, fast=True)
        dataset = p_info.d_info.name
        if dataset not in filtered_datasets:
            continue

        classifier = p_info.h_info.full_name
        for acc_name in acc_names:
            for method in method_names:
                expected.append(
                    ExpectedResult(
                        domain=domain,
                        acc_name=acc_name,
                        dataset=dataset,
                        classifier=classifier,
                        method=method,
                        path=result_path(
                            output_root,
                            experiment,
                            domain,
                            dataset,
                            classifier,
                            method,
                            acc_name,
                        ),
                    )
                )

    return sorted(expected)


def expected_results_from_output(domain: str, output_root: Path, experiment: str) -> list[ExpectedResult]:
    method_names = get_method_names()
    acc_names = get_acc_names()
    observed_pairs = set()
    domain_root = output_root / experiment / domain

    for path in domain_root.glob("*/*/*/*.parquet"):
        acc_name, dataset, classifier = path.relative_to(domain_root).parts[:3]
        if acc_name in acc_names:
            observed_pairs.add((dataset, classifier))

    expected = []
    for dataset, classifier in sorted(observed_pairs):
        for acc_name in acc_names:
            for method in method_names:
                expected.append(
                    ExpectedResult(
                        domain=domain,
                        acc_name=acc_name,
                        dataset=dataset,
                        classifier=classifier,
                        method=method,
                        path=result_path(
                            output_root,
                            experiment,
                            domain,
                            dataset,
                            classifier,
                            method,
                            acc_name,
                        ),
                    )
                )

    return sorted(expected)


def expected_results(domain: str, output_root: Path, experiment: str) -> tuple[list[ExpectedResult], str]:
    if load_info_paths(domain=domain):
        return expected_results_from_pretrain(domain, output_root, experiment), "pretrain metadata"

    return expected_results_from_output(domain, output_root, experiment), "existing output tree"


def print_missing_results(missing: list[ExpectedResult], root: Path) -> None:
    grouped = defaultdict(list)
    for result in missing:
        grouped[(result.dataset, result.classifier, result.method)].append(result.acc_name)

    print("Missing experiments:")
    for dataset, classifier, method in sorted(grouped):
        acc_names = ", ".join(sorted(grouped[(dataset, classifier, method)]))
        print(f"- dataset={dataset} classifier={classifier} method={method} missing_accs={acc_names}")

    print()
    print("Missing result files:")
    for result in missing:
        path = result.path
        try:
            path = path.relative_to(root)
        except ValueError:
            pass
        print(f"- {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show missing output files for the experiments defined in src/uncawe/main.py."
    )
    parser.add_argument(
        "--domain",
        default=DEFAULT_DOMAIN,
        choices=("classic", "text", "image"),
        help=f"Experiment domain to check (default: {DEFAULT_DOMAIN}).",
    )
    parser.add_argument(
        "--experiment",
        default=DEFAULT_EXPERIMENT,
        help=f"Experiment name under the output directory (default: {DEFAULT_EXPERIMENT}).",
    )
    parser.add_argument(
        "--output-root",
        default=env.root_dir,
        help=f"Root directory for uncawe outputs (default: {env.root_dir}).",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print only counts, without listing missing experiments and files.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    expected, source = expected_results(args.domain, output_root, args.experiment)
    missing = [result for result in expected if not result.path.exists()]

    existing_count = len(expected) - len(missing)
    print(f"Experiment: {args.experiment}")
    print(f"Domain: {args.domain}")
    print(f"Output root: {output_root}")
    print(f"Expected grid source: {source}")
    print(f"Expected result files: {len(expected)}")
    print(f"Existing result files: {existing_count}")
    print(f"Missing result files: {len(missing)}")

    missing_experiments = {
        (result.dataset, result.classifier, result.method) for result in missing
    }
    print(f"Missing dataset/classifier/method experiments: {len(missing_experiments)}")

    if missing and not args.summary_only:
        print()
        print_missing_results(missing, Path(os.getcwd()))


if __name__ == "__main__":
    main()
