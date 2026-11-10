"""Rename a method in stored results: ``<old>.parquet`` becomes ``<new>.parquet`` and its ``method`` column is rewritten.

Used by FIX-2 + FIX-3 (docs/work_plan.md) to rename ``RQBS`` to ``SQS-Hbs`` in ``main``, ``main_pre_fix3`` and
``binform``. Only the file name and the ``method`` column change; every other column is copied as stored (the table is
read and written with pyarrow, zstd compression as ``results.py:ResultsDataFrame.save_result``). Each file is written
to a temporary name, then moved into place, and the old file is removed only after that. A file whose target exists, or
whose ``method`` column is not ``<old>`` throughout, is reported and left untouched.

Usage: ``uv run python scripts/migrate/rename_method.py RQBS SQS-Hbs <root> [<root> ...]`` (``--dry-run`` lists the
files only).
"""

import argparse
import os
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def rename_file(path: Path, old: str, new: str, dry_run: bool) -> str:
    target = path.with_name(f"{new}.parquet")
    if target.exists():
        return f"skip (target exists): {path}"
    table = pq.read_table(path)
    if "method" not in table.column_names:
        return f"skip (no method column): {path}"
    methods = table.column("method")
    if not pc.all(pc.equal(methods, old)).as_py():
        return f"skip (method column is not {old!r} throughout): {path}"
    if dry_run:
        return f"would rename: {path} -> {target.name}"
    index = table.column_names.index("method")
    table = table.set_column(index, table.schema.field(index), pa.array([new] * len(table), type=methods.type))
    tmp = target.with_name(f".{target.name}.tmp")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, target)
    path.unlink()
    return f"renamed: {path} -> {target.name}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("old")
    parser.add_argument("new")
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    skipped = renamed = 0
    for root in args.roots:
        for path in sorted(root.rglob(f"{args.old}.parquet")):
            outcome = rename_file(path, args.old, args.new, args.dry_run)
            print(outcome)
            if outcome.startswith("skip"):
                skipped += 1
            else:
                renamed += 1
    print(f"{renamed} renamed, {skipped} skipped")
    return 1 if skipped else 0


if __name__ == "__main__":
    sys.exit(main())
