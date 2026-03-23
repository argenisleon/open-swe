"""Tests for the make_model utility and get_model_id helper.

These tests cover the configurable model feature added in
feat/configurable-model, including MiniMax routing and the
AGENT_MODEL env var.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# get_model_id
# ---------------------------------------------------------------------------


class TestGetModelId:
    def test_default_model(self):
        """Returns the default model when AGENT_MODEL is not set."""
        from agent.utils.model import DEFAULT_MODEL, get_model_id

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AGENT_MODEL", None)
            assert get_model_id() == DEFAULT_MODEL

    def test_env_override(self):
        """AGENT_MODEL env var overrides the default."""
        from agent.utils.model import get_model_id

        with patch.dict(os.environ, {"AGENT_MODEL": "minimax:MiniMax-M2.7"}):
            assert get_model_id() == "minimax:MiniMax-M2.7"


# ---------------------------------------------------------------------------
# make_model
# ---------------------------------------------------------------------------


class TestMakeModel:
    @patch("agent.utils.model.init_chat_model")
    def test_openai_model_sets_responses_api(self, mock_init):
        """OpenAI models get base_url and use_responses_api kwargs."""
        from agent.utils.model import OPENAI_RESPONSES_WS_BASE_URL, make_model

        make_model("openai:gpt-4o", temperature=0)

        mock_init.assert_called_once_with(
            model="openai:gpt-4o",
            temperature=0,
            base_url=OPENAI_RESPONSES_WS_BASE_URL,
            use_responses_api=True,
        )

    @patch("agent.utils.model.init_chat_model")
    def test_anthropic_model_passthrough(self, mock_init):
        """Anthropic models are passed directly to init_chat_model."""
        from agent.utils.model import make_model

        make_model("anthropic:claude-opus-4-6", temperature=0)

        mock_init.assert_called_once_with(
            model="anthropic:claude-opus-4-6",
            temperature=0,
        )

    @patch("agent.utils.model.init_chat_model")
    def test_minimax_model_routing(self, mock_init):
        """MiniMax models are routed through OpenAI-compatible API."""
        from agent.utils.model import MINIMAX_BASE_URL, make_model

        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-123"}):
            make_model("minimax:MiniMax-M2.7", temperature=0)

        mock_init.assert_called_once_with(
            model="openai:MiniMax-M2.7",
            temperature=0,
            base_url=MINIMAX_BASE_URL,
            api_key="test-key-123",
        )

    def test_minimax_missing_api_key_raises(self):
        """MiniMax model without MINIMAX_API_KEY raises KeyError."""
        from agent.utils.model import make_model

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MINIMAX_API_KEY", None)
            with pytest.raises(KeyError, match="MINIMAX_API_KEY"):
                make_model("minimax:MiniMax-M2.7")

    @patch("agent.utils.model.init_chat_model")
    def test_kwargs_not_mutated(self, mock_init):
        """make_model should not mutate the caller's kwargs."""
        from agent.utils.model import make_model

        original = {"temperature": 0}
        make_model("anthropic:claude-opus-4-6", **original)

        # Original dict unchanged
        assert original == {"temperature": 0}
