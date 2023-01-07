from pathlib import Path

import pandas as pd

from src.config.settings import get_config


def load_dataset(force_refresh: bool = False) -> pd.DataFrame:
    config = get_config().data
    local_path: Path = config.local_dataset_path

    if local_path.exists() and not force_refresh:
        return pd.read_csv(local_path)

    local_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(config.dataset_url)
    df.to_csv(local_path, index=False)
    return df


def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    config = get_config()
    out = df.copy()

    out.columns = [c.strip() for c in out.columns]
    for col in config.numeric_coercion_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if config.data.id_column in out.columns:
        out = out.drop(columns=[config.data.id_column])

    if config.data.target_column in out.columns:
        out[config.data.target_column] = out[config.data.target_column].astype(str).str.strip()

    return out
