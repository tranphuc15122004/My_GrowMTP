def resolve_sglang_attention_backend(
    configured_backend: str | None,
    compute_capability: tuple[int | None, int | None],
) -> str:
    """Choose a safe SGLang attention backend for the current GPU generation."""
    if configured_backend is not None:
        return configured_backend

    major, _minor = compute_capability
    if major is not None and major >= 10:
        return "flashinfer"
    return "fa3"
