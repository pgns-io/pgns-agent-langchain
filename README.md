# pgns-agent-langchain

LangChain/LangGraph adapter for [pgns-agent](https://pypi.org/project/pgns-agent/). Wrap any LangChain runnable or LangGraph agent in a production-ready A2A server.

## Installation

```bash
pip install pgns-agent-langchain
```

## Quick Start

```python
from langchain_openai import ChatOpenAI
from pgns_agent import AgentServer
from pgns_agent_langchain import LangChainAdapter

llm = ChatOpenAI(model="gpt-4o")

server = AgentServer("my-agent", "A LangChain-powered agent")
server.use(LangChainAdapter(llm))
server.listen(3000)
```

## Streaming

```python
from pgns_agent_langchain import LangChainStreamAdapter

server.use(LangChainStreamAdapter(llm))
```

## License

Apache-2.0
