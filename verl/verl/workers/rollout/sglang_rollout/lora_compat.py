"""Small, dependency-free helpers for selecting SGLang's LoRA rollout mode."""

from typing import Any

SGLANG_LORA_NAME = "verl_actor_lora_name"


def lora_adapter_enabled(model_config: Any) -> bool:
    """Whether SGLang should load a dynamic adapter instead of merged weights."""
    rank = model_config.lora_rank or model_config.lora.get("rank", 0)
    return rank > 0 and not model_config.lora.get("merge", False)


def configure_lora_server_args(args: dict[str, Any], model_config: Any) -> dict[str, Any]:
    """Add SGLang adapter flags only for the unmerged-adapter rollout path."""
    if lora_adapter_enabled(model_config):
        rank = model_config.lora_rank or model_config.lora.get("rank", 0)
        args.update(
            {
                "enable_lora": True,
                "max_lora_rank": rank,
                "lora_target_modules": model_config.target_modules,
            }
        )
    return args


def lora_adapter_name(model_config: Any) -> str | None:
    """Return the request adapter name only when SGLang loaded a separate adapter."""
    return SGLANG_LORA_NAME if lora_adapter_enabled(model_config) else None
