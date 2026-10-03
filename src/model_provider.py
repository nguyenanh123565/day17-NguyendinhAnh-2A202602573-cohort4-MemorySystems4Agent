from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module


@dataclass
class ProviderConfig:
    """Provider settings; an empty model name disables live model creation."""
    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    provider = value.strip().lower()
    provider = {"anthorpic": "anthropic"}.get(provider, provider)
    if provider not in {"openai", "custom", "gemini", "anthropic", "ollama", "openrouter"}:
        raise ValueError(f"Unsupported model provider: {value!r}")
    return provider


def build_chat_model(config: ProviderConfig):
    """Return a live client, or None when live settings or SDKs are absent.

    Imports are lazy so offline use needs no provider SDKs. Construction does not
    contact a server; callers handle failures during live invocation.
    """
    provider = normalize_provider(config.provider)
    if not config.model_name.strip():
        return None
    if provider not in {"ollama", "custom"} and not config.api_key:
        return None
    if provider == "custom" and not config.base_url:
        return None
    integrations = {
        "openai": ("langchain_openai", "ChatOpenAI"),
        "custom": ("langchain_openai", "ChatOpenAI"),
        "gemini": ("langchain_google_genai", "ChatGoogleGenerativeAI"),
        "anthropic": ("langchain_anthropic", "ChatAnthropic"),
        "ollama": ("langchain_ollama", "ChatOllama"),
        "openrouter": ("langchain_openrouter", "ChatOpenRouter"),
    }
    module_name, class_name = integrations[provider]
    try:
        model_class = getattr(import_module(module_name), class_name)
    except ImportError:
        return None
    kwargs = {"model": config.model_name, "temperature": config.temperature}
    if provider == "custom":
        kwargs["api_key"] = config.api_key or "local-no-key"
    elif config.api_key and provider != "ollama":
        kwargs["api_key"] = config.api_key
    if config.base_url:
        kwargs["openrouter_api_base" if provider == "openrouter" else "base_url"] = config.base_url
    return model_class(**kwargs)
