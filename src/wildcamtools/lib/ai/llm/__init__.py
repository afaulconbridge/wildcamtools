from wildcamtools.lib.ai.llm.abstract import AbstractLlm
from wildcamtools.lib.ai.llm.llamacpp import LlamaCppLlm
from wildcamtools.lib.ai.llm.ollama import OllamaLlm
from wildcamtools.lib.ai.types import Backend


def create_analyser(
    backend: Backend,
    model: str,
    url: str,
    api_key: str | None,
) -> AbstractLlm:
    """Create an AbstractLlm instance for the specified backend.

    Args:
        backend: The LLM backend to use (OLLAMA or LLAMACPP).
        model: The model name/identifier.
        url: The base URL for the LLM service.
        api_key: Optional API key for authentication.

    Returns:
        AbstractLlm: An instance of the configured LLM backend.

    Raises:
        NotImplementedError: If the backend is not supported.
    """
    match backend:
        case Backend.LLAMACPP:
            return LlamaCppLlm(model=model, base_url=url, api_key=api_key)
        case Backend.OLLAMA:
            return OllamaLlm(model=model, host=url, api_key=api_key)
        case _:
            raise NotImplementedError(f"Unsupported backend: {backend}")
