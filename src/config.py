from __future__ import annotations
from pydantic import BaseModel, Field

class Settings(BaseModel):
    """Centralized configuration for data collection scripts."""

    morpho_graphql_url: str = Field(default="https://api.morpho.org/graphql")
    chain_id: int = Field(default=1)
    vault_address: str = Field(default="0xBEeFFF209270748ddd194831b3fa287a5386f5bC")

    @property
    def vault_address_lower(self) -> str:
        """Return the vault address normalized as lowercase hex string."""
        return self.vault_address.lower()

SETTINGS = Settings()