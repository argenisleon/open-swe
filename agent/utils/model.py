import os

from langchain.chat_models import init_chat_model

OPENAI_RESPONSES_WS_BASE_URL = "wss://api.openai.com/v1"
MINIMAX_BASE_URL = "https://api.minimax.io/v1"

DEFAULT_MODEL = "anthropic:claude-opus-4-6"


def get_model_id() -> str:
    """Return the model identifier from the AGENT_MODEL env var or the default."""
    return os.environ.get("AGENT_MODEL", DEFAULT_MODEL)


def make_model(model_id: str, **kwargs: dict):
    model_kwargs = kwargs.copy()

    if model_id.startswith("openai:"):
        model_kwargs["base_url"] = OPENAI_RESPONSES_WS_BASE_URL
        model_kwargs["use_responses_api"] = True
    elif model_id.startswith("minimax:"):
        # MiniMax uses an OpenAI-compatible API
        model_name = model_id.removeprefix("minimax:")
        model_kwargs["base_url"] = MINIMAX_BASE_URL
        model_kwargs["api_key"] = os.environ["MINIMAX_API_KEY"]
        return init_chat_model(model=f"openai:{model_name}", **model_kwargs)

    return init_chat_model(model=model_id, **model_kwargs)
