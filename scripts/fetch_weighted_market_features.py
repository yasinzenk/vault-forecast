from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.config import SETTINGS
from src.morpho.gql_client import MorphoGraphQLClient
from src.morpho.queries import MARKET_HOURLY_QUERY, VAULT_ALLOCATIONS_QUERY


@dataclass(frozen=True)
class TimeRange:
    """UTC time range expressed as unix timestamps (seconds)."""

    start_ts: int
    end_ts: int

    @staticmethod
    def last_days(days: int) -> "TimeRange":
        """Build a [now-days, now] UTC time range."""
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=days)
        return TimeRange(start_ts=int(start.timestamp()), end_ts=int(now.timestamp()))


def _floor_to_hour(dt: datetime) -> datetime:
    """Floor a UTC datetime to the hour."""
    return dt.replace(minute=0, second=0, microsecond=0, tzinfo=timezone.utc)


def _timeseries_to_df(points: List[Dict[str, Any]], column_name: str) -> pd.DataFrame:
    """Convert Morpho [{x,y}] timeseries into a DataFrame (null-safe)."""
    timestamps: List[datetime] = []
    values: List[float] = []

    for point in points:
        ts = datetime.fromtimestamp(int(point["x"]), tz=timezone.utc)
        timestamps.append(_floor_to_hour(ts))
        y = point.get("y")
        values.append(float(y) if y is not None else 0.0)

    return pd.DataFrame({"timestamp": timestamps, column_name: values})


def _fetch_allocation_snapshot(client: MorphoGraphQLClient) -> pd.DataFrame:
    """Fetch a vault allocation snapshot and return a DataFrame of market weights."""
    result = client.execute(
        query=VAULT_ALLOCATIONS_QUERY,
        variables={"address": SETTINGS.vault_address, "chainId": SETTINGS.chain_id},
    )

    state: Dict[str, Any] = result["vaultByAddress"]["state"]
    total_assets_usd = float(state["totalAssetsUsd"] or 0.0)

    if total_assets_usd <= 0:
        raise RuntimeError("Vault totalAssetsUsd is 0. Cannot compute allocation weights safely.")

    allocations: List[Dict[str, Any]] = []
    for alloc in state["allocation"]:
        supply_assets_usd = float(alloc["supplyAssetsUsd"] or 0.0)
        if supply_assets_usd <= 0:
            continue

        allocations.append(
            {
                "market_id": alloc["market"]["uniqueKey"],
                "weight": supply_assets_usd / total_assets_usd,
                "supply_assets_usd": supply_assets_usd,
            }
        )

    if not allocations:
        raise RuntimeError("No active allocations found (all supplyAssetsUsd are 0).")

    allocations_df = pd.DataFrame(allocations).sort_values("weight", ascending=False).reset_index(drop=True)

    # Optional: renormalize in case totalAssetsUsd includes idle/unallocated portions
    weight_sum = float(allocations_df["weight"].sum())
    if weight_sum > 0:
        allocations_df["weight"] = allocations_df["weight"] / weight_sum

    return allocations_df


def main() -> None:
    """
    Fetch weighted underlying market features using an allocation snapshot:
    - weighted_utilization
    - weighted_supply_apy
    - weighted_borrow_apy

    The allocation weights are computed once from vault state (snapshot).
    """
    client = MorphoGraphQLClient(base_url=SETTINGS.morpho_graphql_url)

    # ---- 1) Allocation snapshot (weights)
    allocations_df = _fetch_allocation_snapshot(client)

    # ---- 2) Hourly market series for the last 7 days
    time_range = TimeRange.last_days(days=7)
    options: Dict[str, Any] = {
        "startTimestamp": time_range.start_ts,
        "endTimestamp": time_range.end_ts,
        "interval": "HOUR",
    }

    per_market_frames: List[pd.DataFrame] = []

    for _, alloc_row in allocations_df.iterrows():
        market_id = str(alloc_row["market_id"])
        weight = float(alloc_row["weight"])

        market_result = client.execute(
            query=MARKET_HOURLY_QUERY,
            variables={"marketId": market_id, "options": options},
        )

        historical_state: Dict[str, Any] = market_result["marketByUniqueKey"]["historicalState"]

        utilization_df = _timeseries_to_df(historical_state["utilization"], "utilization")
        supply_apy_df = _timeseries_to_df(historical_state["supplyApy"], "supply_apy")
        borrow_apy_df = _timeseries_to_df(historical_state["borrowApy"], "borrow_apy")

        market_hourly_df = (
            utilization_df.merge(supply_apy_df, on="timestamp", how="inner")
            .merge(borrow_apy_df, on="timestamp", how="inner")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        market_hourly_df["market_id"] = market_id
        market_hourly_df["weight"] = weight

        # Precompute weighted contributions (simplifies aggregation)
        market_hourly_df["w_utilization"] = market_hourly_df["utilization"] * weight
        market_hourly_df["w_supply_apy"] = market_hourly_df["supply_apy"] * weight
        market_hourly_df["w_borrow_apy"] = market_hourly_df["borrow_apy"] * weight

        per_market_frames.append(market_hourly_df)

    all_markets_hourly_df = pd.concat(per_market_frames, ignore_index=True)

    # ---- 3) Weighted aggregation by timestamp
    weighted_features_df = (
        all_markets_hourly_df.groupby("timestamp", as_index=False)[["w_utilization", "w_supply_apy", "w_borrow_apy"]]
        .sum()
        .rename(
            columns={
                "w_utilization": "weighted_utilization",
                "w_supply_apy": "weighted_supply_apy",
                "w_borrow_apy": "weighted_borrow_apy",
            }
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # ---- 4) Save
    output_path = Path("data/processed/market_weighted_features.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    alloc_snapshot_ts = int(datetime.now(timezone.utc).timestamp())
    weighted_features_df["alloc_snapshot_ts"] = alloc_snapshot_ts

    weighted_features_df.to_parquet(output_path, index=False)
    print(f"Wrote {output_path} ✅ rows={len(weighted_features_df)} snapshot_ts={alloc_snapshot_ts}")


if __name__ == "__main__":
    main()
