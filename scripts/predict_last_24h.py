from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd

from src.modeling.features import FeatureConfig, add_lag_and_rolling_features


@dataclass(frozen=True)
class InferenceConfig:
    """Configuration for rolling inference on the most recent hours."""
    horizon_hours: int = 24
    model_path: Path = Path("data/models/huber.joblib")
    feature_schema_path: Path = Path("data/models/feature_columns.joblib")
    output_path: Path = Path("data/processed/predictions_last_24h.parquet")


def _load_parquet(path: Path) -> pd.DataFrame:
    """Load a parquet file with a clear error if missing."""
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path.resolve()}")
    return pd.read_parquet(path)


def _load_training_feature_cols(path: Path) -> List[str]:
    """Load the list of feature columns saved during training."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing training feature schema file: {path.resolve()}. "
            "Run training first and ensure it saves feature_columns.joblib."
        )
    cols = joblib.load(path)
    if not isinstance(cols, list) or not all(isinstance(c, str) for c in cols):
        raise TypeError("feature_columns.joblib must contain a list[str].")
    return cols


def _build_inference_frame(
    vault_hourly_df: pd.DataFrame,
    market_features_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge vault + market features for inference (no target required).

    This intentionally does NOT require 'return_24h' because we are predicting it.
    """
    required_vault_cols = {"timestamp", "share_price", "apy"}
    required_market_cols = {"timestamp", "weighted_utilization", "weighted_supply_apy"}

    missing_vault = required_vault_cols - set(vault_hourly_df.columns)
    missing_market = required_market_cols - set(market_features_df.columns)

    if missing_vault:
        raise ValueError(f"vault_hourly_df missing columns: {sorted(missing_vault)}")
    if missing_market:
        raise ValueError(f"market_features_df missing columns: {sorted(missing_market)}")

    inference_df = (
        vault_hourly_df[["timestamp", "share_price", "apy"]]
        .merge(
            market_features_df[["timestamp", "weighted_utilization", "weighted_supply_apy"]],
            on="timestamp",
            how="inner",
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return inference_df


def _ensure_feature_schema(
    features_df: pd.DataFrame,
    expected_feature_cols: List[str],
) -> pd.DataFrame:
    """
    Ensure inference features match the training schema.

    - checks missing columns
    - selects the exact same column order as training
    - drops rows with NaNs (introduced by lags/rolling)
    """
    missing_cols = [c for c in expected_feature_cols if c not in features_df.columns]
    if missing_cols:
        raise ValueError(f"Inference frame missing feature columns: {missing_cols}")

    X_future = features_df[expected_feature_cols].copy()
    X_future = X_future.dropna(axis=0, how="any")

    if X_future.empty:
        raise RuntimeError(
            "No valid inference rows after dropping NaNs. "
            "This usually means the last horizon window does not have enough history for rolling features."
        )

    return X_future


def main() -> None:
    """
    Generate rolling 24h-ahead predictions for the most recent horizon window.

    Notes:
    - These are forward-looking predictions: y(t) = return over [t, t+24h]
    - Ground truth for these timestamps is typically not available yet.
    """
    repo_root = Path.cwd().resolve()
    data_dir = repo_root / "data" / "processed"

    vault_path = data_dir / "vault_hourly.parquet"
    market_path = data_dir / "market_weighted_features.parquet"

    vault_hourly_df = _load_parquet(vault_path)
    market_features_df = _load_parquet(market_path)

    inference_config = InferenceConfig()
    feature_config = FeatureConfig()

    if not inference_config.model_path.exists():
        raise FileNotFoundError(
            f"Missing trained model: {inference_config.model_path.resolve()}. "
            "Run training first to create huber.joblib."
        )

    # ---- 1) Build inference base frame (no target)
    base_df = _build_inference_frame(
        vault_hourly_df=vault_hourly_df,
        market_features_df=market_features_df,
    )

    # ---- 2) Add lag/rolling features
    full_features_df = add_lag_and_rolling_features(model_df=base_df, config=feature_config)

    # ---- 3) Take last N hours
    horizon_df = full_features_df.tail(inference_config.horizon_hours).copy()

    # ---- 4) Load model + training schema
    huber_model = joblib.load(inference_config.model_path)
    expected_feature_cols = _load_training_feature_cols(inference_config.feature_schema_path)

    X_future = _ensure_feature_schema(horizon_df, expected_feature_cols)

    # Align timestamps with rows kept after dropna
    kept_index = X_future.index
    timestamps = horizon_df.loc[kept_index, "timestamp"].reset_index(drop=True)

    # ---- 5) Predict
    preds_return_24h = huber_model.predict(X_future.to_numpy(dtype=float))
    preds_return_24h = np.asarray(preds_return_24h, dtype=float)

    # ---- 6) Build results (annualized for readability)
    results_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "predicted_return_24h": preds_return_24h,
            "predicted_apy": preds_return_24h * 365.0,
            "market_apy_proxy": horizon_df.loc[kept_index, "weighted_supply_apy"].reset_index(drop=True),
        }
    )

    inference_config.output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(inference_config.output_path, index=False)

    print(f"Wrote {inference_config.output_path} ✅ rows={len(results_df)}")
    print(results_df)


if __name__ == "__main__":
    main()
