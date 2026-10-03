from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    """Shared paths, compact-memory settings, and model configurations."""
    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def _provider_config(prefix: str, fallback: ProviderConfig | None = None) -> ProviderConfig:
    provider = normalize_provider(os.getenv(f"{prefix}_PROVIDER", fallback.provider if fallback else "openai"))
    same_provider = fallback is not None and provider == fallback.provider
    key_names = {
        "openai": "OPENAI_API_KEY", "custom": "CUSTOM_API_KEY",
        "gemini": "GEMINI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
        "ollama": "OLLAMA_API_KEY", "openrouter": "OPENROUTER_API_KEY",
    }
    return ProviderConfig(
        provider=provider,
        model_name=os.getenv(f"{prefix}_MODEL", fallback.model_name if same_provider else "").strip(),
        temperature=float(os.getenv(f"{prefix}_TEMPERATURE", str(fallback.temperature if same_provider else 0.0))),
        api_key=os.getenv(f"{prefix}_API_KEY") or os.getenv(key_names[provider]) or (
            os.getenv("GOOGLE_API_KEY") if provider == "gemini" else None
        ) or (fallback.api_key if same_provider else None),
        base_url=os.getenv(f"{prefix}_BASE_URL") or os.getenv(f"{provider.upper()}_BASE_URL") or (
            fallback.base_url if same_provider else None
        ),
    )


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load root/.env without overriding existing environment variables.

    LLM_* configures the main model; JUDGE_* overrides its defaults for the judge.
    No configured model means offline mode. Compact defaults: 2000 tokens, 6 messages.
    """
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    load_dotenv(root / ".env", override=False)
    threshold = int(os.getenv("COMPACT_THRESHOLD_TOKENS", "2000"))
    keep_messages = int(os.getenv("COMPACT_KEEP_MESSAGES", "6"))
    if threshold <= 0 or keep_messages <= 0:
        raise ValueError("Compact threshold and kept-message count must be positive.")
    model = _provider_config("LLM")
    judge_model = _provider_config("JUDGE", model)
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return LabConfig(root, root / "data", state_dir, threshold, keep_messages, model, judge_model)
