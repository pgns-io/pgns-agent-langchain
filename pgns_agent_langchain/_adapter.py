# Copyright (c) 2026 PGNS LLC
# SPDX-License-Identifier: Apache-2.0

"""LangChain/LangGraph adapters for pgns-agent."""

from __future__ import annotations

__all__ = ["LangChainAdapter", "LangChainStreamAdapter"]

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from pgns_agent import Adapter

if TYPE_CHECKING:
    from langchain_core.runnables import Runnable, RunnableConfig


def _normalize_output(result: Any) -> Any:
    """Convert a LangChain output to a JSON-serializable value.

    Handles the common return types from LangChain/LangGraph runnables:

    * ``dict`` — passed through as-is (e.g. chain output, LangGraph state).
    * ``str``, ``int``, ``float``, ``bool``, ``None`` — scalar primitives.
    * ``list`` — passed through (assumed serializable).
    * Objects with a ``.content`` attribute — ``BaseMessage`` and subclasses;
      extracts the ``.content`` string.
    * Everything else — ``str()`` fallback.
    """
    if isinstance(result, dict | str | int | float | bool | list | type(None)):
        return result
    if hasattr(result, "content"):
        return result.content
    return str(result)


_SAFE_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "finish_reason",
        "model_name",
        "logprobs",
    }
)


def _filter_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return only allowlisted keys from LangChain response_metadata.

    Prevents leaking token usage, rate-limit headers, or provider
    fingerprints into the relay payload.
    """
    return {k: v for k, v in metadata.items() if k in _SAFE_METADATA_KEYS}


def _build_result(result: Any) -> dict[str, Any]:
    """Build a pgns-agent result dict from a LangChain output.

    If the result is a ``BaseMessage`` with ``response_metadata``, only
    safe metadata keys are forwarded under the ``"metadata"`` key.
    """
    out: dict[str, Any] = {"output": _normalize_output(result)}
    if hasattr(result, "response_metadata") and result.response_metadata:
        filtered = _filter_metadata(result.response_metadata)
        if filtered:
            out["metadata"] = filtered
    return out


class LangChainAdapter(Adapter):
    """Wraps a LangChain/LangGraph ``Runnable`` for single-shot execution.

    Calls the runnable's :meth:`~langchain_core.runnables.Runnable.ainvoke`
    method and returns the result as a pgns-agent task result dict.

    Example::

        from langchain_openai import ChatOpenAI
        from pgns_agent_langchain import LangChainAdapter

        llm = ChatOpenAI(model="gpt-4o")
        agent.use(LangChainAdapter(llm))

    Args:
        runnable: Any LangChain ``Runnable`` — an LLM, chain, agent, or
            compiled LangGraph graph.
        config: Optional ``RunnableConfig`` forwarded to every
            ``ainvoke`` call (callbacks, tags, metadata, etc.).
    """

    def __init__(
        self,
        runnable: Runnable[Any, Any],
        *,
        config: RunnableConfig | None = None,
    ) -> None:
        self._runnable = runnable
        self._config = config

    async def handle(self, task_input: dict[str, Any]) -> dict[str, Any]:
        result = await self._runnable.ainvoke(task_input, config=self._config)
        return _build_result(result)


class LangChainStreamAdapter(Adapter):
    """Wraps a LangChain/LangGraph ``Runnable`` for streaming execution.

    Calls the runnable's :meth:`~langchain_core.runnables.Runnable.astream`
    method and yields each chunk as a pgns-agent result dict.  The
    **last yielded dict** becomes the final task result.

    Example::

        from langchain_openai import ChatOpenAI
        from pgns_agent_langchain import LangChainStreamAdapter

        llm = ChatOpenAI(model="gpt-4o", streaming=True)
        agent.use(LangChainStreamAdapter(llm))

    Args:
        runnable: Any LangChain ``Runnable`` — an LLM, chain, agent, or
            compiled LangGraph graph.
        config: Optional ``RunnableConfig`` forwarded to every
            ``astream`` call (callbacks, tags, metadata, etc.).
    """

    def __init__(
        self,
        runnable: Runnable[Any, Any],
        *,
        config: RunnableConfig | None = None,
    ) -> None:
        self._runnable = runnable
        self._config = config

    async def handle(self, task_input: dict[str, Any]) -> AsyncGenerator[dict[str, Any], None]:
        """Process a task input and yield streaming results.

        Returns an async generator (a subtype of ``AsyncIterator``) — compatible
        with the base class union return type.
        """
        async for chunk in self._runnable.astream(task_input, config=self._config):
            yield _build_result(chunk)
