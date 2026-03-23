"""Tests for the stale SANDBOX_CREATING sentinel detection logic.

When sandbox_id is SANDBOX_CREATING but no in-memory backend exists
(e.g. after a server restart), the server should reset the metadata
and create a new sandbox rather than waiting indefinitely.

These tests validate the detection logic without invoking the full
get_agent function, which has many heavy dependencies.
"""

from __future__ import annotations

import pytest


class TestStaleSandboxDetection:
    """Test the conditions for detecting a stale SANDBOX_CREATING sentinel."""

    def test_stale_when_creating_but_no_backend(self):
        """SANDBOX_CREATING + no cached backend = stale sentinel."""
        from agent.server import SANDBOX_CREATING
        from agent.utils.sandbox_state import SANDBOX_BACKENDS

        thread_id = "test-thread-stale"
        sandbox_id = SANDBOX_CREATING
        sandbox_backend = SANDBOX_BACKENDS.get(thread_id)

        # This is exactly the condition checked in get_agent
        is_stale = sandbox_id == SANDBOX_CREATING and not sandbox_backend
        assert is_stale is True

    def test_not_stale_when_creating_with_backend(self):
        """SANDBOX_CREATING + cached backend = actively being created, not stale."""
        from agent.server import SANDBOX_CREATING
        from agent.utils.sandbox_state import SANDBOX_BACKENDS

        thread_id = "test-thread-active"
        SANDBOX_BACKENDS[thread_id] = object()  # simulate a real backend
        try:
            sandbox_id = SANDBOX_CREATING
            sandbox_backend = SANDBOX_BACKENDS.get(thread_id)

            is_stale = sandbox_id == SANDBOX_CREATING and not sandbox_backend
            assert is_stale is False
        finally:
            SANDBOX_BACKENDS.pop(thread_id, None)

    def test_not_stale_when_real_sandbox_id(self):
        """A real sandbox_id (not the sentinel) is never stale."""
        from agent.server import SANDBOX_CREATING

        sandbox_id = "sandbox-abc123"
        sandbox_backend = None  # no backend

        is_stale = sandbox_id == SANDBOX_CREATING and not sandbox_backend
        assert is_stale is False

    def test_not_stale_when_none(self):
        """sandbox_id=None means no sandbox exists yet, not stale."""
        from agent.server import SANDBOX_CREATING

        sandbox_id = None
        sandbox_backend = None

        is_stale = sandbox_id == SANDBOX_CREATING and not sandbox_backend
        assert is_stale is False

    def test_sentinel_value(self):
        """SANDBOX_CREATING sentinel has the expected value."""
        from agent.server import SANDBOX_CREATING

        assert SANDBOX_CREATING == "__creating__"
