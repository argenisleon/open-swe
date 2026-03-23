"""Tests for the auto_github_comment middleware."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from agent.middleware.auto_github_comment import (
    _agent_already_commented,
    _collect_ai_texts,
    _extract_ai_text,
    _messages_since_last_human,
    auto_github_comment,
)


# ---------------------------------------------------------------------------
# _extract_ai_text
# ---------------------------------------------------------------------------


class TestExtractAiText:
    def test_plain_string(self):
        msg = {"type": "ai", "content": "Hello world"}
        assert _extract_ai_text(msg) == "Hello world"

    def test_strips_think_blocks(self):
        msg = {"type": "ai", "content": "<think>reasoning here</think>\n\nThe answer is 42."}
        assert _extract_ai_text(msg) == "The answer is 42."

    def test_strips_multiline_think_blocks(self):
        msg = {
            "type": "ai",
            "content": "<think>\nline1\nline2\n</think>\n\nResult here.",
        }
        assert _extract_ai_text(msg) == "Result here."

    def test_content_list_with_text_blocks(self):
        msg = {
            "type": "ai",
            "content": [
                {"type": "text", "text": "Part one."},
                {"type": "text", "text": "Part two."},
            ],
        }
        assert _extract_ai_text(msg) == "Part one.\nPart two."

    def test_empty_content(self):
        msg = {"type": "ai", "content": ""}
        assert _extract_ai_text(msg) == ""

    def test_only_think_block_returns_empty(self):
        msg = {"type": "ai", "content": "<think>just thinking</think>"}
        assert _extract_ai_text(msg) == ""


# ---------------------------------------------------------------------------
# _agent_already_commented
# ---------------------------------------------------------------------------


class TestAgentAlreadyCommented:
    def test_returns_true_when_github_comment_present(self):
        messages = [
            {"type": "ai", "content": "response"},
            {"type": "tool", "name": "github_comment", "content": '{"success": true}'},
        ]
        assert _agent_already_commented(messages) is True

    def test_returns_false_when_no_github_comment(self):
        messages = [
            {"type": "ai", "content": "response"},
            {"type": "tool", "name": "ls", "content": "file1.py"},
        ]
        assert _agent_already_commented(messages) is False

    def test_returns_false_for_empty_messages(self):
        assert _agent_already_commented([]) is False


# ---------------------------------------------------------------------------
# _messages_since_last_human
# ---------------------------------------------------------------------------


class TestMessagesSinceLastHuman:
    def test_returns_messages_after_last_human(self):
        messages = [
            {"type": "human", "content": "old question"},
            {"type": "ai", "content": "old answer"},
            {"type": "human", "content": "new question"},
            {"type": "ai", "content": "new answer"},
        ]
        result = _messages_since_last_human(messages)
        assert len(result) == 2
        assert result[0]["content"] == "new question"
        assert result[1]["content"] == "new answer"

    def test_returns_all_when_no_human(self):
        messages = [
            {"type": "ai", "content": "answer"},
        ]
        result = _messages_since_last_human(messages)
        assert result == messages

    def test_returns_from_only_human(self):
        messages = [
            {"type": "human", "content": "question"},
        ]
        result = _messages_since_last_human(messages)
        assert len(result) == 1

    def test_multiple_runs_only_returns_last(self):
        messages = [
            {"type": "human", "content": "run 1"},
            {"type": "ai", "content": "response 1"},
            {"type": "human", "content": "run 2"},
            {"type": "ai", "content": "response 2a"},
            {"type": "ai", "content": "response 2b"},
        ]
        result = _messages_since_last_human(messages)
        assert len(result) == 3
        assert result[0]["content"] == "run 2"


# ---------------------------------------------------------------------------
# _collect_ai_texts
# ---------------------------------------------------------------------------


class TestCollectAiTexts:
    def test_collects_ai_texts_from_current_run(self):
        messages = [
            {"type": "human", "content": "old question"},
            {"type": "ai", "content": "old answer"},
            {"type": "human", "content": "new question"},
            {"type": "ai", "content": "Here's the folder structure:\n```\nsrc/\nlib/\n```"},
            {"type": "tool", "name": "ls", "content": "file list"},
            {"type": "ai", "content": "Done."},
        ]
        result = _collect_ai_texts(messages)
        assert "folder structure" in result
        assert "Done." in result
        assert "old answer" not in result

    def test_strips_think_blocks(self):
        messages = [
            {"type": "human", "content": "question"},
            {"type": "ai", "content": "<think>reasoning</think>\n\nActual answer."},
        ]
        result = _collect_ai_texts(messages)
        assert result == "Actual answer."
        assert "reasoning" not in result

    def test_skips_non_ai_messages(self):
        messages = [
            {"type": "human", "content": "question"},
            {"type": "tool", "name": "ls", "content": "files"},
            {"type": "ai", "content": "The answer."},
        ]
        result = _collect_ai_texts(messages)
        assert result == "The answer."

    def test_empty_ai_text_skipped(self):
        messages = [
            {"type": "human", "content": "question"},
            {"type": "ai", "content": "<think>only thinking</think>"},
            {"type": "ai", "content": "Real answer."},
        ]
        result = _collect_ai_texts(messages)
        assert result == "Real answer."

    def test_combines_multiple_ai_messages(self):
        messages = [
            {"type": "human", "content": "question"},
            {"type": "ai", "content": "Part 1."},
            {"type": "tool", "name": "ls", "content": "files"},
            {"type": "ai", "content": "Part 2."},
        ]
        result = _collect_ai_texts(messages)
        assert result == "Part 1.\n\nPart 2."


# ---------------------------------------------------------------------------
# auto_github_comment (integration-level)
# ---------------------------------------------------------------------------


class TestAutoGithubComment:
    """Integration tests for the auto_github_comment middleware.

    The @after_agent decorator creates an AgentMiddleware instance with an
    aafter_agent method.  We call that method directly.
    """

    def _make_config(self, **overrides):
        config = {
            "configurable": {
                "source": "github",
                "repo": {"owner": "test-org", "name": "test-repo"},
                "github_issue": {"number": 42},
            },
        }
        config["configurable"].update(overrides)
        return config

    def _run(self, state):
        """Helper to call the async aafter_agent method synchronously."""
        return asyncio.run(auto_github_comment.aafter_agent(state, None))

    def test_skips_non_github_source(self):
        config = self._make_config(source="linear")
        state = {"messages": [{"type": "ai", "content": "response"}]}

        with patch("agent.middleware.auto_github_comment.get_config", return_value=config):
            result = self._run(state)
        assert result is None

    def test_skips_when_agent_already_commented(self):
        config = self._make_config()
        state = {
            "messages": [
                {"type": "human", "content": "question"},
                {"type": "ai", "content": "response", "tool_calls": []},
                {"type": "tool", "name": "github_comment", "content": '{"success": true}'},
            ]
        }

        with patch("agent.middleware.auto_github_comment.get_config", return_value=config):
            result = self._run(state)
        assert result is None

    def test_skips_when_no_issue_number(self):
        config = self._make_config()
        config["configurable"]["github_issue"] = {}
        config["configurable"].pop("pr_number", None)
        state = {"messages": [{"type": "ai", "content": "response"}]}

        with patch("agent.middleware.auto_github_comment.get_config", return_value=config):
            result = self._run(state)
        assert result is None

    def test_posts_comment_when_agent_did_not(self):
        config = self._make_config()
        state = {
            "messages": [
                {"type": "human", "content": "question"},
                {"type": "ai", "content": "Here is the answer.", "tool_calls": []},
            ]
        }

        mock_token = AsyncMock(return_value="test-token")
        mock_post = AsyncMock(return_value=True)

        with (
            patch("agent.middleware.auto_github_comment.get_config", return_value=config),
            patch(
                "agent.middleware.auto_github_comment.get_github_app_installation_token",
                mock_token,
            ),
            patch("agent.middleware.auto_github_comment.post_github_comment", mock_post),
        ):
            result = self._run(state)

        assert result is None
        mock_post.assert_called_once_with(
            {"owner": "test-org", "name": "test-repo"},
            42,
            "Here is the answer.",
            token="test-token",
        )

    def test_uses_pr_number_when_no_github_issue(self):
        config = self._make_config(pr_number=99)
        config["configurable"].pop("github_issue")
        state = {
            "messages": [
                {"type": "human", "content": "question"},
                {"type": "ai", "content": "PR response.", "tool_calls": []},
            ]
        }

        mock_token = AsyncMock(return_value="test-token")
        mock_post = AsyncMock(return_value=True)

        with (
            patch("agent.middleware.auto_github_comment.get_config", return_value=config),
            patch(
                "agent.middleware.auto_github_comment.get_github_app_installation_token",
                mock_token,
            ),
            patch("agent.middleware.auto_github_comment.post_github_comment", mock_post),
        ):
            self._run(state)

        mock_post.assert_called_once()
        assert mock_post.call_args[0][1] == 99

    def test_skips_when_no_ai_text(self):
        config = self._make_config()
        state = {
            "messages": [
                {"type": "human", "content": "question"},
            ]
        }

        with patch("agent.middleware.auto_github_comment.get_config", return_value=config):
            result = self._run(state)
        assert result is None
