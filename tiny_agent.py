"""
Tiny Agent — Zero-Dependency Agent Framework
Like LangChain, but in one file. Zero dependencies.

MIT License | github.com/tiny-agent/tiny-agent
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import urllib.request
import uuid
from abc import ABC, abstractmethod
from collections import deque
from contextvars import copy_context
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)

__version__ = "0.3.0"
__all__ = [
    "Agent",
    "ToolRegistry",
    "tool",
    "Memory",
    "InMemoryMemory",
    "BaseLLMAdapter",
    "OpenAIAdapter",
    "AnthropicAdapter",
    "OllamaAdapter",
    "HTTPAdapter",
    "UserAdapter",
    "Message",
    "Role",
    "ReActStep",
    "AgentStatus",
    "Plan",
    "Observable",
    "AgentConfig",
    "AgentError",
    "MaxIterationsError",
    "ToolError",
    "LLMError",
    "PlanningError",
    "BudgetGovernor",
    "BudgetExceeded",
    "IdempotencyGuard",
]

# ─────────────────────────── Logging ───────────────────────────

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("tiny_agent")


# ─────────────────────────── Errors ───────────────────────────

class AgentError(Exception):
    """Base exception for Tiny Agent."""
    pass


class MaxIterationsError(AgentError):
    """Raised when max iterations reached."""
    pass


class ToolError(AgentError):
    """Raised when tool execution fails."""
    pass


class LLMError(AgentError):
    """Raised when LLM call fails."""
    pass


class PlanningError(AgentError):
    """Raised when planning fails."""
    pass


# ─────────────────────────── Data Types ───────────────────────────

class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    role: Role
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.name:
            d["name"] = self.name
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Message:
        return cls(
            role=Role(d.get("role", "user")),
            content=d.get("content", ""),
            name=d.get("name"),
            tool_calls=d.get("tool_calls"),
            tool_call_id=d.get("tool_call_id"),
            metadata=d.get("metadata", {}),
        )


class AgentStatus(Enum):
    IDLE = auto()
    THINKING = auto()
    ACTING = auto()
    OBSERVING = auto()
    PLANNING = auto()
    DELEGATING = auto()
    DONE = auto()
    ERROR = auto()


@dataclass
class ReActStep:
    iteration: int
    thought: str
    action: str
    observation: str
    status: str = "ok"
    timestamp: float = field(default_factory=time.time)


@dataclass
class Plan:
    goal: str
    steps: List[str]
    current_step: int = 0

    def next_step(self) -> Optional[str]:
        if self.current_step < len(self.steps):
            step = self.steps[self.current_step]
            self.current_step += 1
            return step
        return None

    def is_complete(self) -> bool:
        return self.current_step >= len(self.steps)


# ─────────────────────────── Observability ───────────────────────────

class Observable:
    """Mixin for observable events."""

    def __init__(self):
        self._hooks: Dict[str, List[Callable]] = {}

    def on(self, event: str, callback: Callable) -> None:
        self._hooks.setdefault(event, []).append(callback)

    def off(self, event: str, callback: Callable) -> None:
        if event in self._hooks:
            self._hooks[event] = [c for c in self._hooks[event] if c != callback]

    def emit(self, event: str, *args, **kwargs) -> None:
        for callback in self._hooks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Hook error for {event}: {e}")


# ─────────────────────────── Tool Registry ───────────────────────────

@dataclass
class Tool:
    name: str
    func: Callable[..., Any]
    description: str
    parameters: Dict[str, Any]

    def to_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def call(self, **kwargs) -> Any:
        return self.func(**kwargs)


class ToolRegistry:
    """Registry for tools."""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise ToolError(f"Tool '{name}' not found")
        return self._tools[name]

    def list_tools(self) -> List[Tool]:
        return list(self._tools.values())

    def get_schemas(self) -> List[Dict[str, Any]]:
        return [t.to_schema() for t in self._tools.values()]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)


T = TypeVar("T")


def tool(name: Optional[str] = None, description: Optional[str] = None, schema: Optional[Dict[str, Any]] = None):
    """Decorator to register a function as a tool."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        tool_name = name or func.__name__
        tool_desc = description or (func.__doc__ or "").strip()

        if schema:
            params = schema
        else:
            import inspect
            sig = inspect.signature(func)
            properties: Dict[str, Any] = {}
            required: List[str] = []
            for param_name, param in sig.parameters.items():
                param_type = "string"
                if param.annotation != inspect.Parameter.empty:
                    if param.annotation in (int, float):
                        param_type = "number"
                    elif param.annotation == bool:
                        param_type = "boolean"
                    elif param.annotation == list:
                        param_type = "array"
                    elif param.annotation == dict:
                        param_type = "object"
                properties[param_name] = {"type": param_type, "description": f"Parameter: {param_name}"}
                if param.default == inspect.Parameter.empty:
                    required.append(param_name)
            params = {
                "type": "object",
                "properties": properties,
                "required": required,
            }

        tool_obj = Tool(name=tool_name, func=func, description=tool_desc, parameters=params)
        func._tiny_agent_tool = tool_obj  # type: ignore
        return func

    return decorator


# ─────────────────────────── Memory ───────────────────────────

@runtime_checkable
class Memory(Protocol):
    """Protocol for memory backends."""

    def add(self, content: str, **metadata) -> None:
        ...

    def get(self, query: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        ...

    def clear(self) -> None:
        ...


class InMemoryMemory:
    """Simple in-memory memory backend."""

    def __init__(self, max_items: int = 1000):
        self._items: deque = deque(maxlen=max_items)

    def add(self, content: str, **metadata) -> None:
        self._items.append({
            "content": content,
            "timestamp": time.time(),
            **metadata,
        })

    def get(self, query: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        items = list(self._items)
        if query:
            query_lower = query.lower()
            items = [i for i in items if query_lower in i.get("content", "").lower()]
        return items[-limit:]

    def clear(self) -> None:
        self._items.clear()


# ─────────────────────────── LLM Adapters ───────────────────────────

class BaseLLMAdapter(ABC):
    """Base class for LLM adapters."""

    @abstractmethod
    def chat(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Message:
        """Synchronous chat completion."""
        ...

    def stream(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Generator[str, None, None]:
        """Streaming chat completion. Override if supported."""
        msg = self.chat(messages, tools, **kwargs)
        yield msg.content


class OpenAIAdapter(BaseLLMAdapter):
    """Adapter for OpenAI-compatible APIs (uses urllib, no external deps)."""

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1", model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    def _request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/{endpoint}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8")
            raise LLMError(f"OpenAI API error {e.code}: {body}")
        except Exception as e:
            raise LLMError(f"OpenAI request failed: {e}")

    def chat(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Message:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        response = self._request("chat/completions", payload)
        choice = response["choices"][0]
        msg_data = choice["message"]
        return Message(
            role=Role.ASSISTANT,
            content=msg_data.get("content", ""),
            tool_calls=msg_data.get("tool_calls"),
        )

    def stream(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Generator[str, None, None]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                for line in resp:
                    line = line.decode("utf-8").strip()
                    if line.startswith("data: "):
                        chunk = line[6:]
                        if chunk == "[DONE]":
                            break
                        try:
                            obj = json.loads(chunk)
                            delta = obj["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except (json.JSONDecodeError, KeyError):
                            pass
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8")
            raise LLMError(f"OpenAI stream error {e.code}: {body}")
        except Exception as e:
            raise LLMError(f"OpenAI stream failed: {e}")


class AnthropicAdapter(BaseLLMAdapter):
    """Adapter for Anthropic Claude API."""

    def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.anthropic.com/v1"

    def chat(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Message:
        # Filter system message
        system_msg = ""
        chat_messages = []
        for m in messages:
            if m.role == Role.SYSTEM:
                system_msg = m.content
            else:
                chat_messages.append({"role": m.role.value, "content": m.content})

        payload: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": chat_messages,
        }
        if system_msg:
            payload["system"] = system_msg
        if tools:
            payload["tools"] = [{"name": t["function"]["name"], "description": t["function"]["description"], "input_schema": t["function"]["parameters"]} for t in tools]

        url = f"{self.base_url}/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                content = "".join([b.get("text", "") for b in result.get("content", []) if b.get("type") == "text"])
                tool_use = next((b for b in result.get("content", []) if b.get("type") == "tool_use"), None)
                tool_calls = None
                if tool_use:
                    tool_calls = [{"id": tool_use["id"], "type": "function", "function": {"name": tool_use["name"], "arguments": json.dumps(tool_use["input"])}}]
                return Message(role=Role.ASSISTANT, content=content, tool_calls=tool_calls)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8")
            raise LLMError(f"Anthropic API error {e.code}: {body}")
        except Exception as e:
            raise LLMError(f"Anthropic request failed: {e}")

    def stream(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Generator[str, None, None]:
        # Claude streaming is more complex; fallback to non-streaming for simplicity
        msg = self.chat(messages, tools, **kwargs)
        yield msg.content


class OllamaAdapter(BaseLLMAdapter):
    """Adapter for local Ollama instance."""

    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def chat(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Message:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
        headers = {"Content-Type": "application/json"}
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return Message(role=Role.ASSISTANT, content=result.get("message", {}).get("content", ""))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8")
            raise LLMError(f"Ollama error {e.code}: {body}")
        except Exception as e:
            raise LLMError(f"Ollama request failed: {e}")

    def stream(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Generator[str, None, None]:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "stream": True,
        }
        headers = {"Content-Type": "application/json"}
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                for line in resp:
                    line = line.decode("utf-8").strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        content = obj.get("message", {}).get("content", "")
                        if content:
                            yield content
                        if obj.get("done", False):
                            break
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            raise LLMError(f"Ollama stream failed: {e}")


class HTTPAdapter(BaseLLMAdapter):
    """Generic HTTP adapter for custom endpoints."""

    def __init__(self, url: str, headers: Optional[Dict[str, str]] = None, response_parser: Optional[Callable[[Dict[str, Any]], str]] = None):
        self.url = url
        self.headers = headers or {"Content-Type": "application/json"}
        self.response_parser = response_parser or (lambda r: r.get("response", ""))

    def chat(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Message:
        payload = {
            "messages": [m.to_dict() for m in messages],
            "tools": tools or [],
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.url, data=data, headers=self.headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                content = self.response_parser(result)
                return Message(role=Role.ASSISTANT, content=content)
        except Exception as e:
            raise LLMError(f"HTTP adapter failed: {e}")


class UserAdapter(BaseLLMAdapter):
    """Adapter that prompts the user directly (for testing/demo)."""

    def chat(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Message:
        print("\n--- Agent is asking ---")
        for m in messages[-2:]:
            print(f"{m.role.value}: {m.content}")
        user_input = input("Your response: ")
        return Message(role=Role.ASSISTANT, content=user_input)


# ─────────────────────────── Agent Config ───────────────────────────

@dataclass
class AgentConfig:
    max_iterations: int = 10
    max_history: int = 50
    temperature: float = 0.7
    system_prompt: str = "You are a helpful assistant. Use tools when needed. Think step by step."
    enable_planning: bool = True
    enable_delegation: bool = True
    verbose: bool = False
    budget_limit: float = 1.0       # USD hard cap per run
    budget_warn_at: float = 0.7     # fire on_warn at this ratio
    idempotency_path: str | None = None  # deduplication store path


# ─────────────────────────── Main Agent ───────────────────────────

class Agent(Observable):
    """Tiny Agent — ReAct loop with tools, memory, and planning."""

    def __init__(
        self,
        llm: BaseLLMAdapter,
        tools: Optional[ToolRegistry] = None,
        memory: Optional[Any] = None,
        config: Optional[AgentConfig] = None,
        name: str = "TinyAgent",
    ):
        super().__init__()
        self.name = name
        self.llm = llm
        self.tools = tools or ToolRegistry()
        self.memory = memory or InMemoryMemory()
        self.config = config or AgentConfig()
        self.history: List[Message] = []
        self.status = AgentStatus.IDLE
        self.steps: List[ReActStep] = []
        self.plan: Optional[Plan] = None
        self.delegates: Dict[str, Agent] = {}
        self._iterations = 0

        # System prompt
        self._system_message = Message(role=Role.SYSTEM, content=self.config.system_prompt)

        # Budget + idempotency (v0.2.0)
        self.budget = BudgetGovernor(
            limit_usd=self.config.budget_limit,
            warn_at=self.config.budget_warn_at,
        )
        self.idempotency = (
            IdempotencyGuard(self.config.idempotency_path)
            if self.config.idempotency_path else None
        )

    # ── Delegation ──

    def register_delegate(self, name: str, agent: Agent) -> None:
        self.delegates[name] = agent

    def _delegate_task(self, agent_name: str, task: str) -> str:
        if agent_name not in self.delegates:
            return f"Error: Agent '{agent_name}' not found"
        delegate = self.delegates[agent_name]
        result = delegate.run(task)
        return f"Delegate '{agent_name}' result: {result}"

    # ── Planning ──

    def _create_plan(self, goal: str) -> Plan:
        if not self.config.enable_planning:
            return Plan(goal=goal, steps=[goal])

        planning_prompt = f"""Break down this goal into small, actionable steps. 
Return ONLY a JSON array of step strings. No other text.

Goal: {goal}"""
        messages = [
            self._system_message,
            Message(role=Role.USER, content=planning_prompt),
        ]
        try:
            response = self.llm.chat(messages)
            content = response.content.strip()
            # Try to extract JSON array
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()
            steps = json.loads(content)
            if not isinstance(steps, list):
                steps = [goal]
        except Exception:
            steps = [goal]

        plan = Plan(goal=goal, steps=steps)
        self.emit("plan_created", plan)
        return plan

    # ── ReAct Loop ──

    def _build_react_prompt(self, query: str, available_tools: List[Tool]) -> str:
        tools_desc = "\n".join([
            f"- {t.name}: {t.description}"
            for t in available_tools
        ])

        prompt = f"""You are an agent that can use tools. Follow the ReAct pattern:
1. Think about what you need to do
2. Act by calling a tool or providing an answer
3. Observe the result

Available tools:
{tools_desc}

Respond in this format:
Thought: <your reasoning>
Action: <tool_name>(<JSON arguments>) OR Action: Final Answer(<your final answer>)

User query: {query}
"""
        return prompt

    def _parse_action(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse Action: tool(args) from text."""
        text = text.strip()

        # Look for Final Answer
        if "Final Answer" in text or "final answer" in text.lower():
            # Extract after the marker
            for marker in ["Final Answer(", "final answer(", "Final Answer:", "final answer:"]:
                if marker in text:
                    idx = text.index(marker) + len(marker)
                    answer = text[idx:].strip()
                    if answer.endswith(")"):
                        answer = answer[:-1]
                    return "FINAL", answer
            return "FINAL", text

        # Look for Action: tool(args)
        if "Action:" in text:
            action_part = text.split("Action:")[-1].strip()
            if "(" in action_part and ")" in action_part:
                tool_name = action_part[:action_part.index("(")].strip()
                args_str = action_part[action_part.index("(") + 1:action_part.rindex(")")]
                return tool_name, args_str
            else:
                return action_part, None

        return None, None

    def _execute_tool(self, tool_name: str, args_str: Optional[str]) -> str:
        if tool_name == "FINAL":
            return args_str or ""

        tool = self.tools.get(tool_name)
        args: Dict[str, Any] = {}
        if args_str:
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                # Try simple key=value parsing
                args = {}
                for part in args_str.split(","):
                    if "=" in part:
                        k, v = part.split("=", 1)
                        args[k.strip()] = v.strip().strip('"').strip("'")

        try:
            result = tool.call(**args)
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    def run(self, query: str, stream: bool = False) -> Union[str, Generator[str, None, None]]:
        if stream:
            return self._run_stream(query)
        return self._run_sync(query)

    def _run_sync(self, query: str) -> str:
        self.status = AgentStatus.THINKING
        self._iterations = 0
        self.steps = []
        self.emit("run_started", query)

        # Create plan
        if self.config.enable_planning:
            self.status = AgentStatus.PLANNING
            self.plan = self._create_plan(query)
        else:
            self.plan = Plan(goal=query, steps=[query])

        # Check memory
        memories = self.memory.get(query=query, limit=5)
        memory_context = ""
        if memories:
            memory_context = "\nRelevant memories:\n" + "\n".join([f"- {m['content']}" for m in memories])

        # Build messages
        messages = [self._system_message]

        # Add conversation history (recent only)
        history_to_add = self.history[-self.config.max_history:]
        messages.extend(history_to_add)

        full_query = query + memory_context

        # ReAct loop
        current_query = full_query
        final_answer = ""

        while self._iterations < self.config.max_iterations:
            self._iterations += 1

            # Check if plan has next step
            if self.plan and not self.plan.is_complete():
                current_step = self.plan.steps[self.plan.current_step - 1] if self.plan.current_step > 0 else query
            else:
                current_step = query

            # Build ReAct prompt
            react_prompt = self._build_react_prompt(current_step, self.tools.list_tools())

            if self._iterations > 1:
                # Add previous steps context
                previous = "\n".join([
                    f"Iteration {s.iteration}:\nThought: {s.thought}\nAction: {s.action}\nObservation: {s.observation}"
                    for s in self.steps[-3:]
                ])
                react_prompt += f"\n\nPrevious steps:\n{previous}\n\nContinue solving the original query: {query}"

            messages.append(Message(role=Role.USER, content=react_prompt))

            self.status = AgentStatus.THINKING
            self.emit("thinking", self._iterations)

            try:
                response = self.llm.chat(messages, tools=self.tools.get_schemas() if len(self.tools) > 0 else None)
            except LLMError as e:
                self.status = AgentStatus.ERROR
                self.emit("error", e)
                raise

            # Check budget before spending more (v0.2.0)
            if hasattr(response, "usage") and response.usage:
                model_name = getattr(self.llm, "model", None) or "unknown"
                self.budget.check(
                    model=model_name,
                    input_tokens=getattr(response.usage, "prompt_tokens", 0) or 0,
                    output_tokens=getattr(response.usage, "completion_tokens", 0) or 0,
                )

            content = response.content
            tool_calls = response.tool_calls

            # Parse thought and action
            thought = ""
            if "Thought:" in content:
                thought = content.split("Thought:")[1].split("Action:")[0].strip() if "Action:" in content else content.split("Thought:")[1].strip()
            else:
                thought = content[:200]

            action = ""
            observation = ""

            # Handle tool calls (OpenAI style)
            if tool_calls:
                self.status = AgentStatus.ACTING
                for tc in tool_calls:
                    func = tc["function"]
                    tool_name = func["name"]
                    args_str = func["arguments"]
                    action = f"{tool_name}({args_str})"
                    self.emit("action", tool_name, args_str)

                    observation = self._execute_tool(tool_name, args_str)
                    self.emit("observation", observation)

                    # Add tool result to messages
                    messages.append(Message(
                        role=Role.TOOL,
                        content=observation,
                        tool_call_id=tc.get("id"),
                    ))

                    step = ReActStep(
                        iteration=self._iterations,
                        thought=thought,
                        action=action,
                        observation=observation,
                    )
                    self.steps.append(step)

                    if observation and "Error" not in observation:
                        # Check if this was a final answer tool
                        pass

            # Check for Final Answer in content
            elif "Final Answer" in content or "final answer" in content.lower():
                tool_name, answer = self._parse_action(content)
                if tool_name == "FINAL":
                    final_answer = answer
                    step = ReActStep(
                        iteration=self._iterations,
                        thought=thought,
                        action="Final Answer",
                        observation=final_answer,
                    )
                    self.steps.append(step)
                    break

            # Try manual action parsing
            else:
                tool_name, args_str = self._parse_action(content)
                if tool_name and tool_name != "FINAL":
                    self.status = AgentStatus.ACTING
                    action = f"{tool_name}({args_str or ''})"
                    self.emit("action", tool_name, args_str)

                    observation = self._execute_tool(tool_name, args_str)
                    self.emit("observation", observation)

                    step = ReActStep(
                        iteration=self._iterations,
                        thought=thought,
                        action=action,
                        observation=observation,
                    )
                    self.steps.append(step)

                    messages.append(Message(role=Role.ASSISTANT, content=content))
                    messages.append(Message(role=Role.USER, content=f"Observation: {observation}"))

                elif tool_name == "FINAL":
                    final_answer = args_str or content
                    step = ReActStep(
                        iteration=self._iterations,
                        thought=thought,
                        action="Final Answer",
                        observation=final_answer,
                    )
                    self.steps.append(step)
                    break
                else:
                    # No clear action, assume it's the final answer
                    final_answer = content
                    step = ReActStep(
                        iteration=self._iterations,
                        thought=thought,
                        action="Respond",
                        observation=final_answer,
                    )
                    self.steps.append(step)
                    break

            # Update plan step
            if self.plan and not self.plan.is_complete():
                self.plan.next_step()

        else:
            # Max iterations reached
            self.status = AgentStatus.ERROR
            err = MaxIterationsError(f"Max iterations ({self.config.max_iterations}) reached")
            self.emit("error", err)
            raise err

        # Store in memory and history
        self.memory.add(content=final_answer, query=query, type="response")
        self.history.append(Message(role=Role.USER, content=query))
        self.history.append(Message(role=Role.ASSISTANT, content=final_answer))

        # Trim history
        if len(self.history) > self.config.max_history:
            self.history = self.history[-self.config.max_history:]

        self.status = AgentStatus.DONE
        self.emit("run_complete", final_answer)
        return final_answer

    def _run_stream(self, query: str) -> Generator[str, None, None]:
        """Streaming version - yields chunks as they arrive."""
        self.status = AgentStatus.THINKING
        self._iterations = 0

        messages = [self._system_message]
        messages.append(Message(role=Role.USER, content=query))

        # For streaming, we do a simpler pass - just stream the LLM response
        try:
            for chunk in self.llm.stream(messages):
                yield chunk
        except LLMError as e:
            self.status = AgentStatus.ERROR
            self.emit("error", e)
            raise

        self.status = AgentStatus.DONE

    def reset(self) -> None:
        """Reset agent state."""
        self.history.clear()
        self.steps.clear()
        self.plan = None
        self.status = AgentStatus.IDLE
        self._iterations = 0
        self.emit("reset")

    def get_history(self) -> List[Message]:
        return list(self.history)

    def get_steps(self) -> List[ReActStep]:
        return list(self.steps)


# ─────────────────────────── Budget & Idempotency (v0.2.0) ───────────────────────────

@dataclass
class BudgetGovernor:
    """Runtime cost + token enforcement for agents. Prevents runaway LLM spend.

    Usage:
        gov = BudgetGovernor(limit_usd=0.50, warn_at=0.7)
        gov.check("openai/gpt-4", input_tokens=200, output_tokens=80)
        # raises BudgetExceeded if limit crossed
    """
    limit_usd: float = 1.0
    warn_at: float = 0.7
    on_warn: Optional[Callable[[float, float], None]] = None
    _spent: float = 0.0
    _warned: bool = False
    _path: Optional[str] = None

    # Approximate pricing (USD per 1M tokens)
    PRICES = {
        "gpt-4o": 5.0, "gpt-4o-mini": 0.15,
        "gpt-4-turbo": 10.0, "gpt-3.5-turbo": 2.0,
        "claude-3-5-sonnet": 3.0, "claude-3-5-haiku": 0.8,
        "claude-3-opus": 15.0, "claude-3-haiku": 0.25,
        "gemini-1.5-pro": 1.25, "gemini-1.5-flash": 0.075,
        "default": 2.0,
    }

    def price_for(self, model: str) -> float:
        for k, v in self.PRICES.items():
            if k in (model or "").lower():
                return v
        return self.PRICES["default"]

    def record(self, model: str, input_tokens: int, output_tokens: int = 0) -> float:
        cost = (input_tokens + output_tokens) / 1_000_000 * self.price_for(model)
        self._spent += cost
        return cost

    def check(self, model: str, input_tokens: int = 0, output_tokens: int = 0) -> None:
        cost = self.record(model, input_tokens, output_tokens)
        # Use cumulative spend for ratio so warnings/exceeded reflect total usage.
        ratio = self._spent / self.limit_usd if self.limit_usd > 0 else 1.0
        if ratio >= 1.0:
            raise BudgetExceeded(
                f"Budget exceeded: ${self._spent:.4f} > ${self.limit_usd:.2f}"
            )
        if not self._warned and ratio >= self.warn_at and self.on_warn:
            self._warned = True
            self.on_warn(self._spent, self.limit_usd)

    @property
    def spent(self) -> float:
        return self._spent

    @property
    def remaining(self) -> float:
        return max(0.0, self.limit_usd - self._spent)


class BudgetExceeded(Exception):
    """Raised when the agent's runtime budget is exceeded."""
    pass


class IdempotencyGuard:
    """Prevents duplicate side-effects using a dedup key hash.

    Usage:
        guard = IdempotencyGuard(path="/tmp/myagent/idempotency")
        if guard.already_done("issue-123-processed"):
            return  # skip, already handled
        guard.mark_done("issue-123-processed")
        # ... do the work
    """
    def __init__(self, path: str = "/tmp/idempotency") -> None:
        self._path = path
        self._done: set = set()
        os.makedirs(path, exist_ok=True)
        self._load()

    def _key_path(self, key: str) -> str:
        h = hashlib.sha1(key.encode()).hexdigest()[:16]
        return os.path.join(self._path, h)

    def _load(self) -> None:
        for fname in os.listdir(self._path):
            self._done.add(fname)

    def already_done(self, key: str) -> bool:
        return os.path.exists(self._key_path(key))

    def mark_done(self, key: str) -> None:
        open(self._key_path(key), "w").close()
        self._done.add(os.path.basename(self._key_path(key)))
