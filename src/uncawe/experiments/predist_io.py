"""Read, validate, and atomically persist incremental PrediQuant experiment results."""

import os
import tempfile

import pandas as pd

import uncawe.env as env
from uncawe.experiments.predist_config import EXPERIMENT, METHOD_NAME, RESULT_SCHEMA_VERSION


def result_path(domain: str, dataset: str, classifier: str, acc_name: str) -> str:
    parent = os.path.join(env.root_dir, EXPERIMENT, domain, acc_name, dataset, classifier)
    os.makedirs(parent, exist_ok=True)
    return os.path.join(parent, f"{METHOD_NAME}.parquet")


def atomic_save_parquet(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".parquet", dir=os.path.dirname(path))
    os.close(fd)
    try:
        df.to_parquet(tmp_path, compression="zstd", index=False)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def load_existing(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "schema_version" not in df.columns:
        return pd.DataFrame()
    if not (df["schema_version"] >= RESULT_SCHEMA_VERSION).all():
        return pd.DataFrame()
    return df


def completed_keys(df: pd.DataFrame) -> set[tuple[str, int]]:
    if df.empty:
        return set()
    return set(zip(df["distance"].astype(str), df["sample_id"].astype(int)))
