from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for lag/rolling feature generation."""
    lags_hours: Tuple[int, ...] = (1, 3, 6)
    rolling_windows_hours: Tuple[int, ...] = (6, 24)


def build_model_frame(
    vault_hourly_df: pd.DataFrame,
    market_features_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge vault + market features on timestamp and ensure clean, sorted hourly frame.
    """
    required_vault_cols = {"timestamp", "share_price", "return_24h", "apy"}
    required_market_cols = {"timestamp", "weighted_utilization", "weighted_supply_apy"}

    missing_vault = required_vault_cols - set(vault_hourly_df.columns)
    missing_market = required_market_cols - set(market_features_df.columns)

    if missing_vault:
        raise ValueError(f"vault_hourly_df missing columns: {sorted(missing_vault)}")
    if missing_market:
        raise ValueError(f"market_features_df missing columns: {sorted(missing_market)}")

    model_df = (
        vault_hourly_df[["timestamp", "share_price", "return_24h", "apy"]]
        .merge(
            market_features_df[["timestamp", "weighted_utilization", "weighted_supply_apy"]],
            on="timestamp",
            how="inner",
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    return model_df


def add_lag_and_rolling_features(
    model_df: pd.DataFrame,
    config: FeatureConfig = FeatureConfig(),
) -> pd.DataFrame:
    """
    Add backward-looking lag and rolling features for time-series modeling.
    """
    df = model_df.copy()

    # We intentionally keep 'apy' as an instantaneous feature (no lag/rolling)
    # so it can be used as a deployable APY-implied baseline (apy/365).
    base_feature_cols = ["weighted_utilization", "weighted_supply_apy"]

    for col in base_feature_cols:
        for lag in config.lags_hours:
            df[f"{col}_lag_{lag}h"] = df[col].shift(lag)

    for col in base_feature_cols:
        for window in config.rolling_windows_hours:
            roll = df[col].rolling(window=window, min_periods=window)
            df[f"{col}_roll_mean_{window}h"] = roll.mean()
            df[f"{col}_roll_std_{window}h"] = roll.std()

    return df


def make_xy(
    vault_hourly_df: pd.DataFrame,
    market_features_df: pd.DataFrame,
    config: FeatureConfig = FeatureConfig(),
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Build feature matrix X and target y aligned on timestamp.
    Returns: X, y, timestamps
    """
    base_df = build_model_frame(vault_hourly_df, market_features_df)
    feats_df = add_lag_and_rolling_features(base_df, config=config)

    y = feats_df["return_24h"]

    non_feature_cols = {"timestamp", "share_price", "return_24h"}
    feature_cols = [c for c in feats_df.columns if c not in non_feature_cols]

    X = feats_df[feature_cols]
    timestamps = feats_df["timestamp"]

    valid_mask = X.notna().all(axis=1) & y.notna()
    X = X.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True)
    timestamps = timestamps.loc[valid_mask].reset_index(drop=True)

    return X, y, timestamps
