"""After-agent middleware that posts a GitHub comment if the agent didn't.

Safety net for models that don't reliably call the ``github_comment`` tool.
If the run was triggered from GitHub and the agent produced a response but
never called ``github_comment``, this middleware posts the last AI message
as a comment on the originating issue or PR.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from langchain.agents.middleware import AgentState, after_agent
from langgraph.config import get_config

from ..utils.github_app import get_github_app_installation_token
from ..utils.github_comments import post_github_comment

logger = logging.getLogger(__name__)


def _agent_already_commented(messages: list) -> bool:
    """Return True if any tool message in the conversation is from github_comment."""
    for msg in messages:
        name = msg.get("name", "") if isinstance(msg, dict) else getattr(msg, "name", "")
        if name == "github_comment":
            return True
    return False


def _extract_ai_text(msg: dict | object) -> str:
    """Extract visible text from a single AI message, stripping think blocks."""
    content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")

    if isinstance(content, str):
        text = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return text
    elif isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts).strip()
    return ""


def _messages_since_last_human(messages: list) -> list:
    """Return only the messages from the last human message onward.

    This scopes the middleware to the current run so it doesn't pick up
    AI responses from previous runs on the same thread.
    """
    last_human_idx = -1
    for i, msg in enumerate(messages):
        msg_type = msg.get("type", "") if isinstance(msg, dict) else getattr(msg, "type", "")
        if msg_type == "human":
            last_human_idx = i
    if last_human_idx == -1:
        return messages
    return messages[last_human_idx:]


def _collect_ai_texts(messages: list) -> str:
    """Collect all AI response texts from the current run and combine them.

    Only considers messages after the last human message (i.e. the current
    run).  No filtering — all AI text is included so we don't accidentally
    drop the actual answer.
    """
    current_run_msgs = _messages_since_last_human(messages)

    parts: list[str] = []
    for msg in current_run_msgs:
        msg_type = msg.get("type", "") if isinstance(msg, dict) else getattr(msg, "type", "")
        if msg_type != "ai":
            continue
        text = _extract_ai_text(msg)
        if text:
            parts.append(text)

    return "\n\n".join(parts)


@after_agent
async def auto_github_comment(
    state: AgentState,
    runtime: object,
) -> dict[str, Any] | None:
    """Post the agent's response as a GitHub comment if it wasn't already posted."""
    logger.info("auto_github_comment middleware started")

    try:
        config = get_config()
        configurable = config.get("configurable", {})

        source = configurable.get("source", "")
        if source != "github":
            logger.debug("Source is '%s', not github — skipping", source)
            return None

        # Determine the issue/PR number
        github_issue = configurable.get("github_issue", {})
        issue_number = github_issue.get("number") if github_issue else None
        if not issue_number:
            issue_number = configurable.get("pr_number")
        if not issue_number:
            logger.info("No issue/PR number found in config, skipping auto-comment")
            return None

        repo_config = configurable.get("repo", {})
        if not repo_config.get("owner") or not repo_config.get("name"):
            logger.info("No repo config found, skipping auto-comment")
            return None

        messages = state.get("messages", [])

        current_run_msgs = _messages_since_last_human(messages)

        if _agent_already_commented(current_run_msgs):
            logger.info("Agent already called github_comment, skipping auto-comment")
            return None

        text = _collect_ai_texts(messages)
        if not text:
            logger.info("No AI response text found, skipping auto-comment")
            return None

        token = await get_github_app_installation_token()
        if not token:
            logger.warning("Failed to get GitHub App token for auto-comment")
            return None

        logger.info("Posting auto-comment to %s/%s#%s", repo_config.get("owner"), repo_config.get("name"), issue_number)
        success = await post_github_comment(repo_config, issue_number, text, token=token)

        if success:
            logger.info("Auto-comment posted successfully")
        else:
            logger.warning("Failed to post auto-comment")

    except Exception:
        logger.exception("Error in auto_github_comment middleware")

    return None
