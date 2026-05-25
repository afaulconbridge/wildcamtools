from wildcamtools.lib.ai import Backend
from wildcamtools.lib.ai.llm.abstract import AbstractLlm
from wildcamtools.lib.ai.llm.llamacpp import LlamaCppLlm
from wildcamtools.lib.ai.llm.ollama import OllamaLlm


def create_analyser(
    backend: Backend,
    model: str,
    url: str,
    api_key: str | None,
) -> AbstractLlm:

    match backend:
        case Backend.LLAMACPP:
            return LlamaCppLlm(model=model, base_url=url, api_key=api_key)
        case Backend.OLLAMA:
            return OllamaLlm(model=model, host=url, api_key=api_key)
        case _:
            raise ValueError(f"Unsupported backend: {backend}")
