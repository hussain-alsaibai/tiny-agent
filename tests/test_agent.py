"""Unit tests for tiny-agent core functionality."""
import pytest
from tiny_agent import (
    Agent, AgentConfig, AgentStatus, Message, Role,
    Tool, ToolRegistry, tool, InMemoryMemory,
    BudgetGovernor, BudgetExceeded, IdempotencyGuard,
    Plan, ReActStep, Observable,
)
from tiny_agent import OpenAIAdapter, HTTPAdapter


class TestMessage:
    def test_message_to_dict(self):
        msg = Message(role=Role.USER, content="hello")
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "hello"

    def test_message_from_dict(self):
        d = {"role": "assistant", "content": "hi there"}
        msg = Message.from_dict(d)
        assert msg.role == Role.ASSISTANT
        assert msg.content == "hi there"

    def test_message_roundtrip(self):
        original = Message(
            role=Role.TOOL,
            content="tool result",
            name="calculator",
            tool_call_id="call_123",
        )
        d = original.to_dict()
        restored = Message.from_dict(d)
        assert restored.role == original.role
        assert restored.content == original.content
        assert restored.name == original.name
        assert restored.tool_call_id == original.tool_call_id


class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()
        t = Tool(name="test", func=lambda: None, description="test tool", parameters={})
        registry.register(t)
        assert "test" in registry
        assert registry.get("test").name == "test"
        assert len(registry) == 1

    def test_list_tools(self):
        registry = ToolRegistry()
        registry.register(Tool("a", lambda: None, "A", {}))
        registry.register(Tool("b", lambda: None, "B", {}))
        assert len(registry.list_tools()) == 2

    def test_get_schema(self):
        registry = ToolRegistry()
        registry.register(Tool("test", lambda: None, "A test tool", {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        }))
        schemas = registry.get_schemas()
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "test"
        assert schemas[0]["function"]["parameters"]["required"] == ["x"]

    def test_get_nonexistent_raises(self):
        registry = ToolRegistry()
        with pytest.raises(Exception):
            registry.get("does_not_exist")


class TestToolDecorator:
    def test_tool_decorator_basic(self):
        @tool()
        def hello(name: str) -> str:
            """Greet someone."""
            return f"Hello, {name}!"
        
        assert hasattr(hello, "_tiny_agent_tool")
        t = hello._tiny_agent_tool
        assert t.name == "hello"
        assert "Greet" in t.description

    def test_tool_decorator_custom_name(self):
        @tool(name="custom")
        def original(): return ""
        
        assert original._tiny_agent_tool.name == "custom"

    def test_tool_decorator_schema(self):
        @tool()
        def add(a: int, b: int = 5) -> int:
            """Add two numbers."""
            return a + b
        
        schema = add._tiny_agent_tool.parameters
        assert "a" in schema["properties"]
        assert "b" in schema["properties"]
        assert "a" in schema["required"]
        assert "b" not in schema["required"]


class TestInMemoryMemory:
    def test_add_and_get(self):
        mem = InMemoryMemory()
        mem.add("visited the page", url="https://example.com")
        results = mem.get()
        assert len(results) == 1
        assert results[0]["content"] == "visited the page"

    def test_query_filter(self):
        mem = InMemoryMemory()
        mem.add("cat fact")
        mem.add("dog fact")
        mem.add("cat behavior")
        results = mem.get(query="cat")
        assert len(results) == 2

    def test_limit(self):
        mem = InMemoryMemory(max_items=3)
        for i in range(5):
            mem.add(f"item {i}")
        results = mem.get()
        assert len(results) == 3

    def test_clear(self):
        mem = InMemoryMemory()
        mem.add("test")
        mem.clear()
        assert len(mem.get()) == 0


class TestObservable:
    def test_on_off_emit(self):
        obs = Observable()
        results = []
        def handler(x): results.append(x)
        obs.on("test_event", handler)
        obs.emit("test_event", 42)
        assert results == [42]
        obs.off("test_event", handler)
        obs.emit("test_event", 99)
        assert results == [42]

    def test_multiple_handlers(self):
        obs = Observable()
        results = []
        obs.on("e", lambda x: results.append(x * 2))
        obs.on("e", lambda x: results.append(x * 3))
        obs.emit("e", 10)
        assert results == [20, 30]


class TestPlan:
    def test_next_step(self):
        plan = Plan(goal="do things", steps=["step 1", "step 2", "step 3"])
        assert plan.next_step() == "step 1"
        assert plan.next_step() == "step 2"
        assert plan.next_step() == "step 3"
        assert plan.next_step() is None

    def test_is_complete(self):
        plan = Plan(goal="test", steps=["a", "b"])
        assert not plan.is_complete()
        plan.next_step()
        assert not plan.is_complete()
        plan.next_step()
        assert plan.is_complete()


class TestBudgetGovernor:
    def test_price_lookup(self):
        gov = BudgetGovernor()
        assert gov.price_for("gpt-4o") == 5.0
        assert gov.price_for("claude-3-5-sonnet") == 3.0
        assert gov.price_for("unknown-model") == gov.PRICES["default"]

    def test_record(self):
        gov = BudgetGovernor(limit_usd=1.0)
        cost = gov.record("gpt-4o", 1000, 500)  # 1500 tokens
        # 1500/1e6 * 5.0 = 0.0075
        assert cost > 0
        assert gov.spent == cost

    def test_budget_exceeded(self):
        gov = BudgetGovernor(limit_usd=0.001)
        with pytest.raises(BudgetExceeded):
            gov.check("gpt-4o", input_tokens=1_000_000, output_tokens=0)

    def test_budget_warning(self):
        gov = BudgetGovernor(limit_usd=1.0, warn_at=0.5)
        warned = []
        gov.on_warn = lambda s, l: warned.append((s, l))
        # Force a warning by spending enough
        gov._spent = 0.4
        gov.check("gpt-4o", input_tokens=20_000, output_tokens=0)  # adds ~0.1
        assert len(warned) == 1


class TestIdempotencyGuard:
    def test_already_done(self, tmp_path):
        guard = IdempotencyGuard(path=str(tmp_path / "idempotency"))
        key = "task-123"
        assert not guard.already_done(key)
        guard.mark_done(key)
        assert guard.already_done(key)

    def test_different_keys_independent(self, tmp_path):
        guard = IdempotencyGuard(path=str(tmp_path / "idempotency2"))
        guard.mark_done("key-a")
        assert guard.already_done("key-a")
        assert not guard.already_done("key-b")


class TestHTTPAdapter:
    def test_basic_chat(self):
        adapter = HTTPAdapter(
            url="https://httpbin.org/post",
            headers={"Content-Type": "application/json"},
            response_parser=lambda r: r.get("url", ""),
        )
        messages = [Message(role=Role.USER, content="hello")]
        # Should not raise
        response = adapter.chat(messages)
        assert response.content != ""

    def test_empty_url_fails_gracefully(self):
        adapter = HTTPAdapter(url="")
        messages = [Message(role=Role.USER, content="hi")]
        with pytest.raises(Exception):
            adapter.chat(messages)
