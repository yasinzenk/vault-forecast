from __future__ import annotations
from pydantic import BaseModel

class ChainInfo(BaseModel):
    """Chain metadata for the asset."""
    id: int
    network: str

class AssetInfo(BaseModel):
    """Vault underlying asset metadata."""
    address: str
    symbol: str
    decimals: int
    chain: ChainInfo

class VaultState(BaseModel):
    """Current vault state snapshot returned by Morpho API."""
    totalAssets: str | int | float
    totalAssetsUsd: str | int | float | None = None
    totalSupply: str | int | float
    sharePriceNumber: float | None = None
    sharePriceUsd: float | None = None
    apy: float | None = None
    netApy: float | None = None
    avgApy: float | None = None
    avgNetApy: float | None = None
    
class Vault(BaseModel):
    """Vault metadata + state."""
    address: str
    name: str
    symbol: str
    listed: bool
    asset: AssetInfo
    state: VaultState
