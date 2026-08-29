# Tiny Agent — Zero-Dependency Agent Framework

> **Like LangChain, but in one file. Zero dependencies. Built for autonomous agents.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen.svg)](tiny_agent.py)
[![Part of the tiny-* ecosystem](https://img.shields.io/badge/tiny--*-ecosystem-purple.svg)](#ecosystem)

Tiny Agent is a **production-quality, zero-dependency** Python agent framework in a single file. Full ReAct loop, streaming, checkpoint/suspend, budget enforcement, idempotency guards, multi-agent delegation — all using only Python's standard library.

## What's New (v0.3.0)

- **Streaming ReAct loop** — stream the reasoning process, not just the final answer
- **Checkpoint integration** — `checkpoint_every=N` decorator auto-saves state every N steps
- **Better MCP tool schemas** — JSON schema auto-generation now matches MCP protocol
- **Budget governor** — hard USD limits with warning callbacks
- **Idempotency guards** — prevent duplicate side-effects from re-runs
- **5 real-world examples** — bounty hunter, code reviewer, data pipeline, meeting summarizer, incident responder

## ✨ Features

| | |
|---|---|
| 🔄 **ReAct Loop** | Full reasoning → action → observation cycle |
| 🔧 **Tool Registry** | Decorator-based registration with MCP-compatible JSON schema |
| 📡 **Streaming** | Stream the full ReAct reasoning process in real-time |
| 💾 **Checkpoint/Suspend** | Auto-save every N steps; resume on crash |
| 🛡️ **Budget Governor** | Hard USD limits; warning callbacks at 70% spend |
| 🔁 **Idempotency Guards** | SHA-1 dedup keys prevent duplicate side-effects |
| 🧠 **Memory** | In-memory with optional SnapDB persistence |
| 🤖 **LLM Agnostic** | OpenAI, Anthropic, Ollama, HTTP endpoints, custom |
| 🤝 **Multi-Agent** | Delegate tasks to sub-agents with result passing |
| 📊 **Observable** | Event hooks: `on("thinking")`, `on("action")`, `on("observation")` |

## 🚀 Quick Start

```python
from tiny_agent import Agent, OpenAIAdapter, tool, AgentConfig

# Define tools
@tool()
def calculator(expression: str) -> str:
    """Evaluate a math expression and return the result."""
    result = eval(expression)
    return str(result)

@tool()
def web_search(query: str) -> str:
    """Search the web for information."""
    return f"Found results for: {query}"

# Configure LLM
llm = OpenAIAdapter(
    api_key=os.environ["OPENAI_API_KEY"],
    model="gpt-4o-mini",
)

# Create agent
agent = Agent(
    llm=llm,
    config=AgentConfig(
        system_prompt="You are a helpful assistant. Use tools when needed.",
        max_iterations=10,
    )
)

# Register tools
agent.tools.register(calculator._tiny_agent_tool)
agent.tools.register(web_search._tiny_agent_tool)

# Run
result = agent.run("What is (15 * 23) + 7?")
print(result)  # → "382"
```

## Quick Reference

```python
# One-liner agent
from tiny_agent import Agent, OpenAIAdapter, tool, AgentConfig
import os

@tool()
def search(query: str) -> str:
    """Web search tool."""
    return f"Results for: {query}"

agent = Agent(
    llm=OpenAIAdapter(api_key=os.environ["OPENAI_API_KEY"], model="gpt-4o-mini"),
    config=AgentConfig(max_iterations=5),
)
agent.tools.register(search._tiny_agent_tool)
result = agent.run("Search for the latest AI news")
```

### CLI Usage

```bash
# Install
pip install tiny-agent

# Or from source
pip install git+https://github.com/hussain-alsaibai/tiny-agent.git

# Run examples
python -m tiny_agent.example
```

## 💡 Real-World Examples

### Example 1: Bounty Hunter Agent

```python
"""Autonomous bug bounty hunting agent."""
import os
from tiny_agent import Agent, OpenAIAdapter, tool, AgentConfig

@tool()
def search_code(query: str, language: str = "python") -> str:
    """Search code on GitHub for vulnerability patterns."""
    return f"Found repos matching: {query} in {language}"

@tool()
def analyze_repo(repo_url: str) -> str:
    """Clone and analyze a repo for security issues."""
    return "Analyzing repo for XSS, SQL injection, SSRF patterns..."

@tool()
def report_bug(domain: str, finding: str, severity: str) -> str:
    """File a bug report. Returns the report URL."""
    return f"https://bugbounty.example.com/reports/new?domain={domain}&severity={severity}"

agent = Agent(
    llm=OpenAIAdapter(api_key=os.environ["OPENAI_API_KEY"], model="gpt-4o-mini"),
    config=AgentConfig(
        system_prompt=(
            "You are an elite bug bounty hunter. Search for high-value targets, "
            "find real vulnerabilities, and file quality reports. "
            "Focus on IDOR, XSS, and SSRF in fintech targets."
        ),
        max_iterations=20,
        budget_limit=2.00,
    ),
    name="bounty-hunter",
)

for tool_obj in [search_code, analyze_repo, report_bug]:
    agent.tools.register(tool_obj._tiny_agent_tool)

result = agent.run(
    "Find 3 high-severity XSS vulnerabilities in fintech APIs on GitHub. "
    "Analyze the top result and file a bug report."
)
print(result)
```

### Example 2: Streaming ReAct Loop

```python
"""Stream the agent's full reasoning process."""
# Agent hooks for real-time visibility
def on_thinking(iter):
    print(f"\n🤔 [Iteration {iter}] Thinking...")
def on_action(name, args):
    print(f"  ⚡ Action: {name}({args})")
def on_observation(obs):
    print(f"  👁️  Observed: {str(obs)[:100]}...")

agent = Agent(llm=llm, config=AgentConfig(max_iterations=5))
agent.on("thinking", on_thinking)
agent.on("action", on_action)
agent.on("observation", on_observation)

# Stream the full ReAct reasoning, not just the final answer
full_output = ""
for chunk in agent.run("Write a haiku about debugging", stream=True):
    print(chunk, end="", flush=True)
    full_output += chunk
```

### Example 3: Checkpoint + Resume (Crash Recovery)

```python
"""Agent that survives container restarts."""
import json
from tiny_agent import Agent, tool, AgentConfig

CHECKPOINT_FILE = "/tmp/agent_checkpoint.json"

def load_checkpoint() -> dict | None:
    try:
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def save_checkpoint(agent: Agent, step: int):
    state = {
        "step": step,
        "history": [m.to_dict() for m in agent.get_history()],
    }
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(state, f)

agent = Agent(llm=llm, config=AgentConfig(max_iterations=100))

# Resume from checkpoint if it exists
checkpoint = load_checkpoint()
if checkpoint:
    print(f"Resuming from step {checkpoint['step']}")
    for msg_dict in checkpoint["history"]:
        from tiny_agent import Message
        agent.history.append(Message.from_dict(msg_dict))
    agent._iterations = checkpoint["step"]

# Run with checkpoint every 10 steps
CHECKPOINT_INTERVAL = 10
for i in range(agent._iterations, 100):
    try:
        agent.run(f"Process batch {i}")
        if i % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(agent, i + 1)
    except Exception as e:
        print(f"Crashed at step {i}: {e}")
        save_checkpoint(agent, i)
        break
```

### Example 4: Budget Enforcement

```python
"""Agent that stops before burning through credits."""
from tiny_agent import Agent, OpenAIAdapter, AgentConfig, BudgetExceeded

def on_budget_warning(spent: float, limit: float):
    print(f"⚠️  Budget warning: ${spent:.4f} / ${limit:.2f} ({spent/limit*100:.0f}%)")

agent = Agent(
    llm=OpenAIAdapter(api_key=os.environ["OPENAI_API_KEY"], model="gpt-4o"),
    config=AgentConfig(
        budget_limit=0.10,      # Hard cap: $0.10 per run
        budget_warn_at=0.7,    # Warn at $0.07
    ),
    name="budget-agent",
)

agent.budget.on_warn = on_budget_warning

try:
    result = agent.run("Analyze this codebase for performance issues")
    print(result)
except BudgetExceeded as e:
    print(f"Stopped: {e}")
    print(f"Final spend: ${agent.budget.spent:.4f}")
```

### Example 5: Multi-Agent Pipeline

```python
"""Coordinated multi-agent workflow."""
researcher = Agent(llm=llm, name="researcher",
    config=AgentConfig(system_prompt="You research topics thoroughly.", max_iterations=5))
writer = Agent(llm=llm, name="writer",
    config=AgentConfig(system_prompt="You write clear, concise summaries.", max_iterations=5))
editor = Agent(llm=llm, name="editor",
    config=AgentConfig(system_prompt="You edit for clarity and brevity.", max_iterations=3))

coordinator = Agent(llm=llm, name="coordinator", config=AgentConfig(max_iterations=3))
coordinator.register_delegate("researcher", researcher)
coordinator.register_delegate("writer", writer)
coordinator.register_delegate("editor", editor)

result = coordinator.run(
    "Research quantum computing basics, write a 3-paragraph summary, "
    "then edit for clarity. Return the final edited version."
)
```

## 🏗️ Architecture

```
tiny-agent = ReAct loop + Observable + Budget + Idempotency

ReAct Loop:
  query → think → act → observe → think → act → observe → ... → answer
                ↑___________________________↓
```

## 🧩 LLM Adapters

| Adapter | Provider | Streaming | Tool Calls |
|---------|----------|-----------|-----------|
| `OpenAIAdapter` | OpenAI / Azure / compatible | ✅ | ✅ |
| `AnthropicAdapter` | Claude / Anthropic | ⚠️ | ✅ |
| `OllamaAdapter` | Local Ollama | ✅ | ✅ |
| `HTTPAdapter` | Any HTTP endpoint | ❌ | ❌ |
| `UserAdapter` | Interactive console | ❌ | ❌ |

All adapters use only `urllib.request` — **zero external dependencies**.

## 📊 Comparison

| Feature | Tiny Agent | LangChain | CrewAI | AutoGen |
|---------|------------|-----------|--------|---------|
| **File Count** | 1 | 1000+ | 500+ | 800+ |
| **Dependencies** | 0 | 50+ | 30+ | 40+ |
| **ReAct Loop** | ✅ Native | ✅ | ✅ | ✅ |
| **Streaming** | ✅ Full loop | ✅ | ❌ | ⚠️ |
| **Checkpoint** | ✅ Manual | ❌ | ❌ | ❌ |
| **Budget Governor** | ✅ Native | ❌ | ❌ | ❌ |
| **Idempotency** | ✅ Native | ❌ | ❌ | ❌ |

## 🔧 Event Hooks

```python
agent = Agent(llm=llm, config=AgentConfig())

agent.on("run_started",   lambda q: print(f"Started: {q}"))
agent.on("thinking",      lambda i: print(f"Thinking {i}..."))
agent.on("action",        lambda n, a: print(f"Action: {n}"))
agent.on("observation",   lambda o: print(f"Observed: {o}"))
agent.on("error",         lambda e: print(f"Error: {e}"))
agent.on("run_complete",  lambda r: print(f"Done: {r}"))
agent.on("plan_created",   lambda p: print(f"Plan: {p.steps}"))
```

## Ecosystem

Part of the **tiny-\*** ecosystem — zero-dependency Python utilities:

| Repo | Description |
|------|-------------|
| [tiny-mesh](https://github.com/hussain-alsaibai/tiny-mesh) | Service mesh proxy |
| [tiny-repl](https://github.com/hussain-alsaibai/tiny-repl) | Agent debugging REPL |
| [tiny-trace](https://github.com/hussain-alsaibai/tiny-trace) | Distributed tracing |
| [tiny-checkpoint](https://github.com/hussain-alsaibai/tiny-checkpoint) | Agent suspend/resume |
| [tiny-llm-router](https://github.com/hussain-alsaibai/tiny-llm-router) | Multi-provider LLM routing |
| [tiny-llm](https://github.com/hussain-alsaibai/tiny-llm) | Unified LLM client |
| [fast-cache](https://github.com/hussain-alsaibai/fast-cache) | LRU + TTL cache |
| [tiny-log](https://github.com/hussain-alsaibai/tiny-log) | Structured logging |
| [tiny-prompt-cache](https://github.com/hussain-alsaibai/tiny-prompt-cache) | LLM prompt cache |
| [dev-masterkit](https://github.com/hussain-alsaibai/dev-masterkit) | Curated dev tools |

---

*MIT License. https://github.com/hussain-alsaibai/tiny-agent*
