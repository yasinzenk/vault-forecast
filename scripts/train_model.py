from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import joblib

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.modeling.features import FeatureConfig, make_xy


@dataclass(frozen=True)
class TrainConfig:
    """Training configuration for time-series split and baselines."""
    embargo_hours: int = 24
    test_size_fraction: float = 0.25
    ridge_alpha: float = 1.0


def _load_parquet(path: Path) -> pd.DataFrame:
    """Load a parquet file with a helpful error message."""
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path.resolve()}")
    return pd.read_parquet(path)


def _add_forward_return_24h(vault_hourly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute forward 24h return from share_price.
    Note: last 24 rows will be NaN by construction.
    """
    df = vault_hourly_df.sort_values("timestamp").reset_index(drop=True).copy()
    if "share_price" not in df.columns:
        raise ValueError("vault_hourly_df must contain a 'share_price' column.")

    df["return_24h"] = df["share_price"].shift(-24) / df["share_price"] - 1.0
    return df


def _time_split_with_embargo(
    X: pd.DataFrame,
    y: pd.Series,
    timestamps: pd.Series,
    config: TrainConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Time-based train/test split with an embargo to reduce leakage from overlapping targets.
    """
    n = len(X)
    if n < 50:
        raise ValueError(f"Not enough samples after feature generation. n={n}")

    test_size = int(np.floor(n * config.test_size_fraction))
    test_size = max(test_size, 24)

    test_start_idx = n - test_size
    embargo = config.embargo_hours

    train_end_idx = test_start_idx - embargo
    if train_end_idx <= 0:
        raise ValueError(
            f"Embargo too large for dataset size. n={n}, test_size={test_size}, embargo={embargo}"
        )

    X_train = X.iloc[:train_end_idx].reset_index(drop=True)
    y_train = y.iloc[:train_end_idx].reset_index(drop=True)

    X_test = X.iloc[test_start_idx:].reset_index(drop=True)
    y_test = y.iloc[test_start_idx:].reset_index(drop=True)

    # Optional: print split dates for sanity
    print("Train:", timestamps.iloc[0], "->", timestamps.iloc[train_end_idx - 1])
    print("Embargo gap ends at:", timestamps.iloc[test_start_idx - 1])
    print("Test :", timestamps.iloc[test_start_idx], "->", timestamps.iloc[-1])

    return X_train, X_test, y_train, y_test


def _evaluate_regression(y_true: pd.Series, y_pred: np.ndarray, label: str) -> None:
    """Print regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    print(f"{label:<18} MAE={mae:.10f}  RMSE={rmse:.10f}")


def main() -> None:
    repo_root = Path.cwd().resolve()
    data_dir = repo_root / "data" / "processed"

    vault_path = data_dir / "vault_hourly.parquet"
    market_path = data_dir / "market_weighted_features.parquet"

    vault_hourly_df = _load_parquet(vault_path)
    market_features_df = _load_parquet(market_path)

    # ---- 1) Target construction (forward 24h return)
    vault_hourly_df = _add_forward_return_24h(vault_hourly_df)

    # ---- 2) Build X/y (lags + rolling are backward-looking only)
    feature_config = FeatureConfig(lags_hours=(1, 3, 6), rolling_windows_hours=(6, 24))
    X, y, timestamps = make_xy(vault_hourly_df, market_features_df, config=feature_config)

    print(f"Model frame ✅ X={X.shape} y={y.shape}")

    # ---- 3) Time split with embargo
    train_config = TrainConfig(embargo_hours=24, test_size_fraction=0.25, ridge_alpha=1.0)
    X_train, X_test, y_train, y_test = _time_split_with_embargo(X, y, timestamps, train_config)

    # ---- 4) Baselines
    baseline_mean = np.full(shape=len(y_test), fill_value=float(y_train.mean()))
    _evaluate_regression(y_test, baseline_mean, label="Baseline (mean)")

    # APY-implied baseline (deployable): r_24h ≈ APY_t / 365
    if "apy" not in X_test.columns:
        raise ValueError("Missing 'apy' in X_test. Check features.make_xy() includes it.")

    baseline_apy = X_test["apy"].to_numpy(dtype=float) / 365.0
    _evaluate_regression(y_test, baseline_apy, label="Baseline (apy/365)")

    # ---- 5) Ridge model
    model = Ridge(alpha=train_config.ridge_alpha)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    _evaluate_regression(y_test, y_pred, label="Ridge")

    # Quick coefficient sanity (top absolute)
    coef_series = pd.Series(model.coef_, index=X_train.columns).abs().sort_values(ascending=False)
    print("\nTop coefficients (abs):")
    print(coef_series.head(10))

    # Huber regression (robust to outliers)
    huber = HuberRegressor(
        epsilon=1.35,      # standard choice
        alpha=0.0001,      # L2 regularization
        max_iter=1000,
    )
    huber.fit(X_train, y_train)
    y_pred_huber = huber.predict(X_test)
    _evaluate_regression(y_test, y_pred_huber, label="Huber")

    huber_coefs = pd.Series(huber.coef_,index=X_train.columns,).abs().sort_values(ascending=False)
    print("\nTop Huber coefficients (abs):")
    print(huber_coefs.head(10))

    models_dir = Path("data/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(huber, models_dir / "huber.joblib")
    joblib.dump(list(X_train.columns), models_dir / "feature_columns.joblib")
    print("Saved model + feature schema ✅")

     # ---- 6) Export results for notebook analysis
    results_df = pd.DataFrame(
        {
            "timestamp": timestamps.loc[y_test.index].reset_index(drop=True),
            "y_true_return_24h": y_test.reset_index(drop=True),
            "pred_ridge": y_pred,
            "pred_huber": y_pred_huber,
            "baseline_mean": baseline_mean,
            "baseline_apy_365": baseline_apy,
        }
    )

    results_path = Path("data/processed/model_results.parquet")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(results_path, index=False)

    print(f"Saved model results for notebook → {results_path} ✅")

if __name__ == "__main__":
    main()
