# Copyright (c) 2026 PGNS LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for LangChainAdapter and LangChainStreamAdapter."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from starlette.testclient import TestClient

from pgns_agent import Adapter, AgentServer
from pgns_agent_langchain import LangChainAdapter, LangChainStreamAdapter

# ---------------------------------------------------------------------------
# Mock runnables — lightweight fakes matching the Runnable protocol
# ---------------------------------------------------------------------------


class MockRunnable:
    """Runnable that returns a dict."""

    def __init__(self, output: Any = None) -> None:
        self._output = output or {"answer": "42"}

    async def ainvoke(self, input: Any, **kwargs: Any) -> Any:
        return self._output

    async def astream(self, input: Any, **kwargs: Any) -> AsyncIterator[Any]:
        words = str(self._output).split()
        for word in words:
            yield word


class MockStringRunnable:
    """Runnable that returns a plain string."""

    async def ainvoke(self, input: Any, **kwargs: Any) -> str:
        return f"echo: {input}"

    async def astream(self, input: Any, **kwargs: Any) -> AsyncIterator[str]:
        for word in f"echo: {input}".split():
            yield word


class MockMessageRunnable:
    """Runnable that returns a fake BaseMessage-like object."""

    async def ainvoke(self, input: Any, **kwargs: Any) -> Any:
        return FakeMessage(content="Hello from LLM", response_metadata={"model_name": "test-llm"})

    async def astream(self, input: Any, **kwargs: Any) -> AsyncIterator[Any]:
        yield FakeMessage(content="Hello", response_metadata={})
        yield FakeMessage(content=" from", response_metadata={})
        yield FakeMessage(content=" LLM", response_metadata={"model_name": "test-llm"})


class FakeMessage:
    """Mimics langchain_core.messages.BaseMessage."""

    def __init__(self, content: str, response_metadata: dict[str, Any] | None = None) -> None:
        self.content = content
        self.response_metadata = response_metadata or {}


class MockConfigCapture:
    """Runnable that captures the config kwarg for assertion."""

    def __init__(self) -> None:
        self.last_config: Any = None

    async def ainvoke(self, input: Any, **kwargs: Any) -> dict[str, Any]:
        self.last_config = kwargs.get("config")
        return {"received_config": self.last_config is not None}

    async def astream(self, input: Any, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        self.last_config = kwargs.get("config")
        yield {"received_config": self.last_config is not None}


class MockErrorRunnable:
    """Runnable that raises during ainvoke/astream."""

    async def ainvoke(self, input: Any, **kwargs: Any) -> Any:
        raise RuntimeError("runnable exploded")

    async def astream(self, input: Any, **kwargs: Any) -> AsyncIterator[Any]:
        raise RuntimeError("stream exploded")
        yield  # pragma: no cover


class MockEmptyStreamRunnable:
    """Runnable whose astream yields nothing."""

    async def ainvoke(self, input: Any, **kwargs: Any) -> dict[str, str]:
        return {"result": "ok"}

    async def astream(self, input: Any, **kwargs: Any) -> AsyncIterator[Any]:
        return
        yield  # makes this an async generator


class MockListRunnable:
    """Runnable that returns a list."""

    async def ainvoke(self, input: Any, **kwargs: Any) -> list[str]:
        return ["a", "b", "c"]

    async def astream(self, input: Any, **kwargs: Any) -> AsyncIterator[list[str]]:
        yield ["a"]
        yield ["a", "b"]
        yield ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Adapter base class compliance
# ---------------------------------------------------------------------------


class TestAdapterSubclass:
    def test_langchain_adapter_is_adapter(self) -> None:
        adapter = LangChainAdapter(MockRunnable())
        assert isinstance(adapter, Adapter)

    def test_stream_adapter_is_adapter(self) -> None:
        adapter = LangChainStreamAdapter(MockRunnable())
        assert isinstance(adapter, Adapter)


# ---------------------------------------------------------------------------
# LangChainAdapter — ainvoke() path
# ---------------------------------------------------------------------------


class TestLangChainAdapterInvoke:
    def test_dict_output(self) -> None:
        agent = AgentServer("test", "test agent")
        agent.use(LangChainAdapter(MockRunnable({"answer": "42"})))

        client = TestClient(agent.app())
        resp = client.post("/", json={"id": "t1", "input": {"question": "meaning"}})
        assert resp.status_code == 200
        result = resp.json()["result"]
        assert result["output"] == {"answer": "42"}

    def test_string_output(self) -> None:
        agent = AgentServer("test", "test agent")
        agent.use(LangChainAdapter(MockStringRunnable()))

        client = TestClient(agent.app())
        resp = client.post("/", json={"id": "t1", "input": "hello"})
        assert resp.status_code == 200
        result = resp.json()["result"]
        assert result["output"] == "echo: hello"

    def test_message_output_extracts_content(self) -> None:
        agent = AgentServer("test", "test agent")
        agent.use(LangChainAdapter(MockMessageRunnable()))

        client = TestClient(agent.app())
        resp = client.post("/", json={"id": "t1", "input": {}})
        assert resp.status_code == 200
        result = resp.json()["result"]
        assert result["output"] == "Hello from LLM"

    def test_message_metadata_passed_through(self) -> None:
        agent = AgentServer("test", "test agent")
        agent.use(LangChainAdapter(MockMessageRunnable()))

        client = TestClient(agent.app())
        resp = client.post("/", json={"id": "t1", "input": {}})
        result = resp.json()["result"]
        assert result["metadata"] == {"model_name": "test-llm"}

    def test_list_output(self) -> None:
        agent = AgentServer("test", "test agent")
        agent.use(LangChainAdapter(MockListRunnable()))

        client = TestClient(agent.app())
        resp = client.post("/", json={"id": "t1", "input": {}})
        assert resp.status_code == 200
        assert resp.json()["result"]["output"] == ["a", "b", "c"]

    def test_null_input_normalized(self) -> None:
        agent = AgentServer("test", "test agent")
        agent.use(LangChainAdapter(MockRunnable()))

        client = TestClient(agent.app())
        resp = client.post("/", json={"id": "t1"})
        assert resp.status_code == 200

    def test_config_forwarded(self) -> None:
        capture = MockConfigCapture()
        agent = AgentServer("test", "test agent")
        agent.use(LangChainAdapter(capture, config={"tags": ["test"]}))

        client = TestClient(agent.app())
        resp = client.post("/", json={"id": "t1", "input": {}})
        assert resp.status_code == 200
        assert capture.last_config == {"tags": ["test"]}

    def test_config_none_by_default(self) -> None:
        capture = MockConfigCapture()
        agent = AgentServer("test", "test agent")
        agent.use(LangChainAdapter(capture))

        client = TestClient(agent.app())
        resp = client.post("/", json={"id": "t1", "input": {}})
        assert resp.status_code == 200
        assert capture.last_config is None

    def test_named_skill(self) -> None:
        agent = AgentServer("test", "test agent")
        agent.use(LangChainAdapter(MockRunnable({"qa": True})), skill="qa")

        client = TestClient(agent.app())
        resp = client.post("/", json={"id": "t1", "input": {}, "skill": "qa"})
        assert resp.status_code == 200
        assert resp.json()["result"]["output"]["qa"] is True

    def test_error_returns_500(self) -> None:
        agent = AgentServer("test", "test agent")
        agent.use(LangChainAdapter(MockErrorRunnable()))

        client = TestClient(agent.app(), raise_server_exceptions=False)
        resp = client.post("/", json={"id": "t1", "input": {}})
        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# LangChainStreamAdapter — astream() path
# ---------------------------------------------------------------------------


class TestLangChainStreamAdapter:
    def test_stream_returns_last_chunk(self) -> None:
        agent = AgentServer("test", "test agent")
        agent.use(LangChainStreamAdapter(MockRunnable({"final": "result"})))

        client = TestClient(agent.app())
        resp = client.post("/", json={"id": "t1", "input": {}})
        assert resp.status_code == 200
        # MockRunnable.astream splits str({"final": "result"}) by whitespace,
        # yielding ["{'final':", "'result'}"]. The last chunk becomes the result.
        result = resp.json()["result"]
        assert result["output"] == "'result'}"

    def test_message_stream_extracts_content(self) -> None:
        agent = AgentServer("test", "test agent")
        agent.use(LangChainStreamAdapter(MockMessageRunnable()))

        client = TestClient(agent.app())
        resp = client.post("/", json={"id": "t1", "input": {}})
        assert resp.status_code == 200
        result = resp.json()["result"]
        # Last chunk is FakeMessage(content=" LLM", ...)
        assert result["output"] == " LLM"
        assert result["metadata"] == {"model_name": "test-llm"}

    def test_empty_stream_returns_null(self) -> None:
        agent = AgentServer("test", "test agent")
        agent.use(LangChainStreamAdapter(MockEmptyStreamRunnable()))

        client = TestClient(agent.app())
        resp = client.post("/", json={"id": "t1", "input": {}})
        assert resp.status_code == 200
        assert resp.json()["result"] is None

    def test_stream_named_skill(self) -> None:
        agent = AgentServer("test", "test agent")
        agent.use(LangChainStreamAdapter(MockListRunnable()), skill="lists")

        client = TestClient(agent.app())
        resp = client.post("/", json={"id": "t1", "input": {}, "skill": "lists"})
        assert resp.status_code == 200
        assert resp.json()["result"]["output"] == ["a", "b", "c"]

    def test_stream_config_forwarded(self) -> None:
        capture = MockConfigCapture()
        agent = AgentServer("test", "test agent")
        agent.use(LangChainStreamAdapter(capture, config={"tags": ["stream"]}))

        client = TestClient(agent.app())
        resp = client.post("/", json={"id": "t1", "input": {}})
        assert resp.status_code == 200
        assert capture.last_config == {"tags": ["stream"]}

    def test_stream_error_returns_500(self) -> None:
        agent = AgentServer("test", "test agent")
        agent.use(LangChainStreamAdapter(MockErrorRunnable()))

        client = TestClient(agent.app(), raise_server_exceptions=False)
        resp = client.post("/", json={"id": "t1", "input": {}})
        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# Multiple adapters on one server
# ---------------------------------------------------------------------------


class TestMultipleAdapters:
    def test_invoke_and_stream_coexist(self) -> None:
        agent = AgentServer("test", "test agent")
        agent.use(LangChainAdapter(MockRunnable({"mode": "invoke"})), skill="invoke")
        agent.use(LangChainStreamAdapter(MockListRunnable()), skill="stream")

        client = TestClient(agent.app())

        invoke_resp = client.post("/", json={"id": "t1", "input": {}, "skill": "invoke"})
        assert invoke_resp.json()["result"]["output"]["mode"] == "invoke"

        stream_resp = client.post("/", json={"id": "t2", "input": {}, "skill": "stream"})
        assert stream_resp.json()["result"]["output"] == ["a", "b", "c"]

    def test_adapter_appears_in_agent_card(self) -> None:
        agent = AgentServer("test", "test agent")
        agent.use(LangChainAdapter(MockRunnable()), skill="qa")
        agent.use(LangChainStreamAdapter(MockRunnable()), skill="chat")

        client = TestClient(agent.app())
        resp = client.get("/.well-known/agent.json")
        skill_ids = {s["id"] for s in resp.json()["skills"]}
        assert skill_ids == {"qa", "chat"}


# ---------------------------------------------------------------------------
# pgns_agent.adapters re-export
# ---------------------------------------------------------------------------


class TestAdaptersReExport:
    def test_import_langchain_adapter(self) -> None:
        from pgns_agent.adapters import LangChainAdapter as ReExported

        assert ReExported is LangChainAdapter

    def test_import_stream_adapter(self) -> None:
        from pgns_agent.adapters import LangChainStreamAdapter as ReExported

        assert ReExported is LangChainStreamAdapter

    def test_unknown_attr_raises(self) -> None:
        from pgns_agent import adapters

        with pytest.raises(AttributeError, match="no attribute"):
            _ = adapters.NoSuchAdapter  # type: ignore[attr-defined]
