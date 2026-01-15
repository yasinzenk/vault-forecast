from __future__ import annotations

VAULT_BY_ADDRESS_QUERY: str = """
query SmokehouseVault($address: String!, $chainId: Int!) {
  vaultByAddress(address: $address, chainId: $chainId) {
    address
    name
    symbol
    listed
    asset {
      address
      symbol
      decimals
      chain { id network }
    }
    state {
      totalAssets
      totalAssetsUsd
      totalSupply
      sharePriceNumber
      sharePriceUsd
      apy
      netApy
      avgApy
      avgNetApy
    }
  }
}
"""

VAULT_TOTALS_AND_APY_HOURLY_QUERY: str = """
query VaultTotalsAndApyHourly($address: String!, $chainId: Int!, $options: TimeseriesOptions) {
  vaultByAddress(address: $address, chainId: $chainId) {
    historicalState {
      totalAssets(options: $options) { x y }
      totalSupply(options: $options) { x y }
      apy(options: $options) { x y }
      netApy(options: $options) { x y }
    }
  }
}
"""

VAULT_ALLOCATIONS_QUERY: str = """
query VaultAllocations($address: String!, $chainId: Int!) {
  vaultByAddress(address: $address, chainId: $chainId) {
    state {
      totalAssetsUsd
      allocation {
        market {
          uniqueKey
        }
        supplyAssetsUsd
      }
    }
  }
}
"""

MARKET_HOURLY_QUERY: str = """
query MarketHourly($marketId: String!, $options: TimeseriesOptions) {
  marketByUniqueKey(uniqueKey: $marketId) {
    historicalState {
      utilization(options: $options) { x y }
      supplyApy(options: $options) { x y }
      borrowApy(options: $options) { x y }
    }
  }
}
"""