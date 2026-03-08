"""Shared OpenRouter LLM client factory."""

from openai import OpenAI
from backend.config import OPENROUTER_KEY

_client: OpenAI | None = None


def get_openrouter_client() -> OpenAI:
    """Get or create the singleton OpenRouter client."""
    global _client
    if _client is None:
        if not OPENROUTER_KEY:
            raise ValueError("OPENROUTER_KEY not configured in backend/.env")
        _client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_KEY,
        )
    return _client
