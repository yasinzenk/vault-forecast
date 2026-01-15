from __future__ import annotations
from typing import Any, Dict, Optional
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential


class MorphoGraphQLClient:
    """GraphQL client with retries for the Morpho API."""

    def __init__(self, base_url: str, timeout_seconds: float = 30.0) -> None:
        self._base_url = base_url
        self._timeout_seconds = timeout_seconds

    @retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(5))

    def execute(self, query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a GraphQL query and return the JSON payload as a dict."""
        payload: Dict[str, Any] = {"query": query, "variables": variables or {}}
        with httpx.Client(timeout=self._timeout_seconds) as client:
            response = client.post(self._base_url, json=payload)
            response.raise_for_status()
            data: Dict[str, Any] = response.json()

        if "errors" in data:
            raise RuntimeError(f"GraphQL errors: {data['errors']}")
        if "data" not in data:
            raise RuntimeError(f"Unexpected GraphQL response: {data}")

        return data["data"]

