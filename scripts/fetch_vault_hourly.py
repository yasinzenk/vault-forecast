from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from src.config import SETTINGS
from src.morpho.gql_client import MorphoGraphQLClient
from src.morpho.models import Vault
from src.morpho.queries import VAULT_BY_ADDRESS_QUERY
from src.morpho.queries import VAULT_TOTALS_AND_APY_HOURLY_QUERY

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


def _timeseries_points_to_dataframe(points: List[Dict[str, Any]], value_name: str) -> pd.DataFrame:
    """Convert Morpho [{x,y}, ...] points into a timestamped DataFrame."""
    rows: List[Dict[str, Any]] = []
    for point in points:
        rows.append(
            {
                "timestamp": datetime.fromtimestamp(int(point["x"]), tz=timezone.utc),
                value_name: float(point["y"]),
            }
        )
    return pd.DataFrame(rows)


def _fetch_reference_snapshot(client: MorphoGraphQLClient) -> Tuple[float, int]:
    """Fetch the latest snapshot share price and asset decimals to normalize historical series."""
    variables: Dict[str, Any] = {"address": SETTINGS.vault_address, "chainId": SETTINGS.chain_id}
    result = client.execute(query=VAULT_BY_ADDRESS_QUERY, variables=variables)
    vault = Vault.model_validate(result["vaultByAddress"])

    reference_share_price = float(vault.state.sharePriceNumber)
    asset_decimals = int(vault.asset.decimals)
    return reference_share_price, asset_decimals


def main() -> None:
    """Fetch >=7 days of hourly totals + APY and write a clean parquet dataset."""
    client = MorphoGraphQLClient(base_url=SETTINGS.morpho_graphql_url)

    reference_share_price, asset_decimals = _fetch_reference_snapshot(client)
    time_range = TimeRange.last_days(days=7)

    variables: Dict[str, Any] = {
        "address": SETTINGS.vault_address,
        "chainId": SETTINGS.chain_id,
        "options": {
            "startTimestamp": time_range.start_ts,
            "endTimestamp": time_range.end_ts,
            "interval": "HOUR",
        },
    }

    result = client.execute(query=VAULT_TOTALS_AND_APY_HOURLY_QUERY, variables=variables)
    historical_state = result["vaultByAddress"]["historicalState"]

    total_assets_points: List[Dict[str, Any]] = historical_state["totalAssets"]
    total_supply_points: List[Dict[str, Any]] = historical_state["totalSupply"]
    apy_points: List[Dict[str, Any]] = historical_state["apy"]
    net_apy_points: List[Dict[str, Any]] = historical_state["netApy"]

    if not total_assets_points or not total_supply_points:
        raise RuntimeError("Empty totals timeseries returned. Check the selected time range/options.")

    vault_total_assets_df = _timeseries_points_to_dataframe(total_assets_points, "total_assets_raw")
    vault_total_supply_df = _timeseries_points_to_dataframe(total_supply_points, "total_supply_raw")

    # APY series can occasionally have missing points; we merge outer and keep what we have.
    vault_apy_df = _timeseries_points_to_dataframe(apy_points, "apy") if apy_points else pd.DataFrame(columns=["timestamp", "apy"])
    vault_net_apy_df = (
        _timeseries_points_to_dataframe(net_apy_points, "net_apy")
        if net_apy_points
        else pd.DataFrame(columns=["timestamp", "net_apy"])
    )

    vault_hourly_df = vault_total_assets_df.merge(vault_total_supply_df, on="timestamp", how="inner")
    vault_hourly_df = vault_hourly_df.merge(vault_apy_df, on="timestamp", how="left")
    vault_hourly_df = vault_hourly_df.merge(vault_net_apy_df, on="timestamp", how="left")

    vault_hourly_df = vault_hourly_df.sort_values("timestamp").reset_index(drop=True)

    # Normalize assets into human units (USDC has 6 decimals).
    vault_hourly_df["total_assets"] = vault_hourly_df["total_assets_raw"] / (10**asset_decimals)

    # Rescale ratio to match Morpho's canonical snapshot sharePriceNumber.
    last_assets = float(vault_hourly_df["total_assets"].iloc[-1])
    last_supply_raw = float(vault_hourly_df["total_supply_raw"].iloc[-1])

    if last_assets <= 0.0 or last_supply_raw <= 0.0:
        raise RuntimeError("Invalid last assets/supply values; cannot calibrate share price scaling.")

    raw_ratio_last = last_assets / last_supply_raw
    scale_factor = reference_share_price / raw_ratio_last

    vault_hourly_df["share_price"] = (vault_hourly_df["total_assets"] / vault_hourly_df["total_supply_raw"]) * scale_factor

    # Sanity check: last computed share price should align with snapshot sharePriceNumber.
    last_share_price = float(vault_hourly_df["share_price"].iloc[-1])
    if abs(last_share_price - reference_share_price) > 1e-6:
        raise RuntimeError(
            f"Share price scaling mismatch: computed={last_share_price}, reference={reference_share_price}"
        )

    # TVL in asset units (USDC).
    vault_hourly_df["tvl_usdc_proxy"] = vault_hourly_df["total_assets"]

    output_path = Path("data/processed/vault_hourly.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vault_hourly_df.to_parquet(output_path, index=False)
    print(f"Wrote {output_path} ✅ rows={len(vault_hourly_df)}")


if __name__ == "__main__":
    main()
