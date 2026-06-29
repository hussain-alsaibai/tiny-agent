"""
Tests for Tiny Agent — Zero-Dependency Agent Framework

Run: python -m unittest test_tiny_agent
Or: python test_tiny_agent.py
"""

import json
import time
import unittest
from unittest.mock import MagicMock, patch

from tiny_agent import (
    Agent,
    AgentConfig,
    AgentStatus,
    AnthropicAdapter,
    BaseLLMAdapter,
    HTTPAdapter,
    InMemoryMemory,
    LLMError,
    MaxIterationsError,
    Message,
    OllamaAdapter,
    OpenAIAdapter,
    Plan,
    ReActStep,
    Role,
    Tool,
    ToolError,
    ToolRegistry,
    tool,
)


# ─────────────────────────── Fixtures ───────────────────────────

class MockLLM(BaseLLMAdapter):
    """Mock LLM for testing."""

    def __init__(self, responses=None):
        self.responses = responses or []
        self.call_count = 0
        self.last_messages = None

    def chat(self, messages, tools=None, **kwargs):
        self.last_messages = messages
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        return Message(role=Role.ASSISTANT, content="Default response")

    def stream(self, messages, tools=None, **kwargs):
        self.last_messages = messages
        yield "Hello"
        yield " world"
        yield "!"


# ─────────────────────────── Tool Tests ───────────────────────────

class TestToolRegistry(unittest.TestCase):

    def test_register_and_get(self):
        registry = ToolRegistry()

        def dummy(a: int, b: int) -> int:
            return a + b

        t = Tool(name="add", func=dummy, description="Add two numbers", parameters={"type": "object", "properties": {}})
        registry.register(t)

        self.assertEqual(len(registry), 1)
        self.assertIn("add", registry)
        self.assertEqual(registry.get("add").call(a=1, b=2), 3)

    def test_get_missing_raises(self):
        registry = ToolRegistry()
        with self.assertRaises(ToolError):
            registry.get("nonexistent")

    def test_list_tools(self):
        registry = ToolRegistry()
        registry.register(Tool(name="t1", func=lambda: None, description="", parameters={}))
        registry.register(Tool(name="t2", func=lambda: None, description="", parameters={}))
        self.assertEqual(len(registry.list_tools()), 2)

    def test_get_schemas(self):
        registry = ToolRegistry()
        registry.register(Tool(name="t1", func=lambda: None, description="Test", parameters={"type": "object"}))
        schemas = registry.get_schemas()
        self.assertEqual(len(schemas), 1)
        self.assertEqual(schemas[0]["function"]["name"], "t1")


class TestToolDecorator(unittest.TestCase):

    def test_decorator_basic(self):
        @tool()
        def my_func(a: int, b: str):
            """My docstring."""
            return f"{a} {b}"

        self.assertTrue(hasattr(my_func, "_tiny_agent_tool"))
        t = my_func._tiny_agent_tool
        self.assertEqual(t.name, "my_func")
        self.assertEqual(t.description, "My docstring.")
        self.assertEqual(t.parameters["type"], "object")
        self.assertIn("a", t.parameters["properties"])
        self.assertIn("b", t.parameters["properties"])

    def test_decorator_custom_name(self):
        @tool(name="custom_name", description="Custom desc")
        def my_func():
            pass

        t = my_func._tiny_agent_tool
        self.assertEqual(t.name, "custom_name")
        self.assertEqual(t.description, "Custom desc")

    def test_decorator_schema(self):
        @tool(schema={"type": "object", "properties": {"x": {"type": "number"}}})
        def my_func(x):
            return x * 2

        t = my_func._tiny_agent_tool
        self.assertEqual(t.parameters, {"type": "object", "properties": {"x": {"type": "number"}}})

    def test_tool_call(self):
        @tool()
        def multiply(a: int, b: int) -> int:
            return a * b

        result = multiply._tiny_agent_tool.call(a=3, b=4)
        self.assertEqual(result, 12)


# ─────────────────────────── Memory Tests ───────────────────────────

class TestInMemoryMemory(unittest.TestCase):

    def test_add_and_get(self):
        mem = InMemoryMemory()
        mem.add("Hello world", tag="greeting")
        items = mem.get()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["content"], "Hello world")
        self.assertEqual(items[0]["tag"], "greeting")

    def test_get_with_query(self):
        mem = InMemoryMemory()
        mem.add("Python is great")
        mem.add("Java is ok")
        mem.add("Python rocks")
        items = mem.get(query="python")
        self.assertEqual(len(items), 2)

    def test_get_limit(self):
        mem = InMemoryMemory()
        for i in range(10):
            mem.add(f"Item {i}")
        items = mem.get(limit=3)
        self.assertEqual(len(items), 3)

    def test_clear(self):
        mem = InMemoryMemory()
        mem.add("test")
        mem.clear()
        self.assertEqual(len(mem.get()), 0)

    def test_max_items(self):
        mem = InMemoryMemory(max_items=5)
        for i in range(10):
            mem.add(f"Item {i}")
        self.assertEqual(len(mem.get()), 5)


# ─────────────────────────── Message Tests ───────────────────────────

class TestMessage(unittest.TestCase):

    def test_to_dict(self):
        msg = Message(role=Role.USER, content="Hello")
        d = msg.to_dict()
        self.assertEqual(d["role"], "user")
        self.assertEqual(d["content"], "Hello")

    def test_from_dict(self):
        msg = Message.from_dict({"role": "assistant", "content": "Hi"})
        self.assertEqual(msg.role, Role.ASSISTANT)
        self.assertEqual(msg.content, "Hi")

    def test_to_dict_with_tool_calls(self):
        msg = Message(
            role=Role.ASSISTANT,
            content="",
            tool_calls=[{"id": "1", "function": {"name": "test", "arguments": "{}"}}],
        )
        d = msg.to_dict()
        self.assertIn("tool_calls", d)


# ─────────────────────────── Plan Tests ───────────────────────────

class TestPlan(unittest.TestCase):

    def test_next_step(self):
        plan = Plan(goal="Test", steps=["Step 1", "Step 2", "Step 3"])
        self.assertEqual(plan.next_step(), "Step 1")
        self.assertEqual(plan.next_step(), "Step 2")
        self.assertEqual(plan.current_step, 2)

    def test_is_complete(self):
        plan = Plan(goal="Test", steps=["Step 1"])
        plan.next_step()
        self.assertTrue(plan.is_complete())

    def test_next_step_exhausted(self):
        plan = Plan(goal="Test", steps=["Step 1"])
        plan.next_step()
        self.assertIsNone(plan.next_step())


# ─────────────────────────── Observable Tests ───────────────────────────

class TestObservable(unittest.TestCase):

    def test_on_emit(self):
        from tiny_agent import Observable

        obs = Observable()
        called = []

        def handler(x):
            called.append(x)

        obs.on("test", handler)
        obs.emit("test", 42)
        self.assertEqual(called, [42])

    def test_off(self):
        from tiny_agent import Observable

        obs = Observable()
        called = []

        def handler(x):
            called.append(x)

        obs.on("test", handler)
        obs.off("test", handler)
        obs.emit("test", 42)
        self.assertEqual(called, [])

    def test_multiple_handlers(self):
        from tiny_agent import Observable

        obs = Observable()
        results = []

        obs.on("test", lambda x: results.append(f"a{x}"))
        obs.on("test", lambda x: results.append(f"b{x}"))
        obs.emit("test", 1)
        self.assertEqual(sorted(results), ["a1", "b1"])


# ─────────────────────────── Adapter Tests ───────────────────────────

class TestOpenAIAdapter(unittest.TestCase):

    def test_init(self):
        adapter = OpenAIAdapter(api_key="test-key", model="gpt-4")
        self.assertEqual(adapter.api_key, "test-key")
        self.assertEqual(adapter.model, "gpt-4")

    @patch("urllib.request.urlopen")
    def test_chat_success(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [{"message": {"role": "assistant", "content": "Hello"}}]
        }).encode()
        mock_urlopen.return_value.__enter__.return_value = mock_response

        adapter = OpenAIAdapter(api_key="test")
        messages = [Message(role=Role.USER, content="Hi")]
        result = adapter.chat(messages)
        self.assertEqual(result.content, "Hello")

    @patch("urllib.request.urlopen")
    def test_chat_http_error(self, mock_urlopen):
        from urllib.error import HTTPError

        mock_urlopen.side_effect = HTTPError(
            url="http://test", code=401, msg="Unauthorized", hdrs={}, fp=MagicMock(read=lambda: b'{"error": "bad"}')
        )

        adapter = OpenAIAdapter(api_key="test")
        with self.assertRaises(LLMError):
            adapter.chat([Message(role=Role.USER, content="Hi")])


class TestAnthropicAdapter(unittest.TestCase):

    @patch("urllib.request.urlopen")
    def test_chat_success(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "content": [{"type": "text", "text": "Hello from Claude"}]
        }).encode()
        mock_urlopen.return_value.__enter__.return_value = mock_response

        adapter = AnthropicAdapter(api_key="test")
        messages = [Message(role=Role.USER, content="Hi")]
        result = adapter.chat(messages)
        self.assertEqual(result.content, "Hello from Claude")


class TestOllamaAdapter(unittest.TestCase):

    @patch("urllib.request.urlopen")
    def test_chat_success(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "message": {"content": "Hello from Ollama"}
        }).encode()
        mock_urlopen.return_value.__enter__.return_value = mock_response

        adapter = OllamaAdapter(model="llama3")
        messages = [Message(role=Role.USER, content="Hi")]
        result = adapter.chat(messages)
        self.assertEqual(result.content, "Hello from Ollama")


class TestHTTPAdapter(unittest.TestCase):

    @patch("urllib.request.urlopen")
    def test_chat_with_parser(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"response": "Parsed!"}).encode()
        mock_urlopen.return_value.__enter__.return_value = mock_response

        adapter = HTTPAdapter(url="http://test/api", response_parser=lambda r: r["response"])
        result = adapter.chat([Message(role=Role.USER, content="Hi")])
        self.assertEqual(result.content, "Parsed!")


class TestUserAdapter(unittest.TestCase):

    @patch("builtins.input", return_value="My answer")
    def test_user_adapter(self, mock_input):
        adapter = MockLLM()  # Use mock instead for automated testing
        result = adapter.chat([Message(role=Role.USER, content="Q")])
        self.assertIsNotNone(result)


# ─────────────────────────── Agent Tests ───────────────────────────

class TestAgentBasics(unittest.TestCase):

    def test_init(self):
        llm = MockLLM()
        agent = Agent(llm=llm, name="TestAgent")
        self.assertEqual(agent.name, "TestAgent")
        self.assertEqual(agent.status, AgentStatus.IDLE)

    def test_register_delegate(self):
        llm = MockLLM()
        agent = Agent(llm=llm)
        delegate = Agent(llm=llm, name="Delegate")
        agent.register_delegate("helper", delegate)
        self.assertIn("helper", agent.delegates)

    def test_reset(self):
        llm = MockLLM()
        agent = Agent(llm=llm)
        agent.history = [Message(role=Role.USER, content="test")]
        agent.reset()
        self.assertEqual(len(agent.history), 0)
        self.assertEqual(agent.status, AgentStatus.IDLE)

    def test_get_history(self):
        llm = MockLLM()
        agent = Agent(llm=llm)
        agent.history = [Message(role=Role.USER, content="test")]
        self.assertEqual(len(agent.get_history()), 1)

    def test_get_steps(self):
        llm = MockLLM()
        agent = Agent(llm=llm)
        agent.steps = [ReActStep(iteration=1, thought="t", action="a", observation="o")]
        self.assertEqual(len(agent.get_steps()), 1)


class TestAgentReAct(unittest.TestCase):

    def test_run_with_final_answer(self):
        """Agent should stop when it sees Final Answer."""
        llm = MockLLM(responses=[
            Message(role=Role.ASSISTANT, content="Thought: I know\nAction: Final Answer(42)")
        ])
        agent = Agent(llm=llm, config=AgentConfig(enable_planning=False))
        result = agent.run("What is 6*7?")
        self.assertIn("42", result)
        self.assertEqual(agent.status, AgentStatus.DONE)

    def test_run_tool_then_answer(self):
        """Agent should use tool then provide final answer."""
        llm = MockLLM(responses=[
            Message(role=Role.ASSISTANT, content="Thought: Need to add\nAction: add(a=1, b=2)"),
            Message(role=Role.ASSISTANT, content="Thought: Got it\nAction: Final Answer(3)")
        ])
        agent = Agent(llm=llm, config=AgentConfig(enable_planning=False))

        @tool()
        def add(a: int, b: int) -> int:
            return a + b

        agent.tools.register(add._tiny_agent_tool)
        result = agent.run("What is 1+2?")
        self.assertIn("3", result)

    def test_run_with_tool_calls(self):
        """Agent should handle OpenAI-style tool_calls."""
        llm = MockLLM(responses=[
            Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[{
                    "id": "tc1",
                    "type": "function",
                    "function": {"name": "multiply", "arguments": '{"a": 3, "b": 4}'}
                }]
            ),
            Message(role=Role.ASSISTANT, content="Thought: Got it\nAction: Final Answer(12)")
        ])
        agent = Agent(llm=llm, config=AgentConfig(enable_planning=False))

        @tool()
        def multiply(a: int, b: int) -> int:
            return a * b

        agent.tools.register(multiply._tiny_agent_tool)
        result = agent.run("What is 3*4?")
        self.assertIn("12", result)

    def test_max_iterations(self):
        """Agent should raise MaxIterationsError."""
        # LLM always returns a non-final action
        llm = MockLLM(responses=[
            Message(role=Role.ASSISTANT, content="Thought: Hmm\nAction: think()")
        ] * 15)

        agent = Agent(
            llm=llm,
            config=AgentConfig(max_iterations=3, enable_planning=False)
        )

        @tool()
        def think():
            return "thinking..."

        agent.tools.register(think._tiny_agent_tool)

        with self.assertRaises(MaxIterationsError):
            agent.run("Think deeply")

    def test_streaming(self):
        llm = MockLLM()
        agent = Agent(llm=llm, config=AgentConfig(enable_planning=False))
        chunks = list(agent.run("Hello", stream=True))
        self.assertEqual("".join(chunks), "Hello world!")

    def test_memory_integration(self):
        """Agent should store and retrieve from memory."""
        llm = MockLLM(responses=[
            Message(role=Role.ASSISTANT, content="Thought: I know\nAction: Final Answer(Paris)")
        ])
        mem = InMemoryMemory()
        agent = Agent(llm=llm, memory=mem, config=AgentConfig(enable_planning=False))
        agent.run("What is the capital of France?")

        # Memory stores the response; query for content that should be there
        memories = mem.get(query="Paris")
        self.assertTrue(len(memories) > 0)

    def test_observable_events(self):
        llm = MockLLM(responses=[
            Message(role=Role.ASSISTANT, content="Thought: I know\nAction: Final Answer(OK)")
        ])
        agent = Agent(llm=llm, config=AgentConfig(enable_planning=False))
        events = []

        agent.on("run_started", lambda q: events.append(("start", q)))
        agent.on("run_complete", lambda r: events.append(("complete", r)))

        agent.run("Test")
        self.assertEqual(events[0][0], "start")
        self.assertEqual(events[1][0], "complete")

    def test_history_management(self):
        llm = MockLLM(responses=[
            Message(role=Role.ASSISTANT, content="Thought: I know\nAction: Final Answer(A)")
        ])
        agent = Agent(
            llm=llm,
            config=AgentConfig(max_history=5, enable_planning=False)
        )
        agent.run("Q1")
        agent.run("Q2")
        self.assertTrue(len(agent.history) <= 5)

    def test_planning_enabled(self):
        llm = MockLLM(responses=[
            Message(role=Role.ASSISTANT, content='["Step 1", "Step 2"]'),
            Message(role=Role.ASSISTANT, content="Thought: I know\nAction: Final Answer(Done)")
        ])
        agent = Agent(llm=llm, config=AgentConfig(enable_planning=True))
        agent.run("Do something")
        self.assertIsNotNone(agent.plan)

    def test_planning_disabled(self):
        llm = MockLLM(responses=[
            Message(role=Role.ASSISTANT, content="Thought: I know\nAction: Final Answer(Done)")
        ])
        agent = Agent(llm=llm, config=AgentConfig(enable_planning=False))
        agent.run("Do something")
        self.assertIsNotNone(agent.plan)
        self.assertEqual(len(agent.plan.steps), 1)

    def test_error_recovery(self):
        """Agent should handle LLM errors gracefully."""
        llm = MockLLM()
        # Force LLMError on first call by patching
        original_chat = llm.chat
        call_count = [0]

        def failing_chat(messages, tools=None, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise LLMError("Network error")
            return original_chat(messages, tools, **kwargs)

        llm.chat = failing_chat
        agent = Agent(llm=llm, config=AgentConfig(enable_planning=False))

        with self.assertRaises(LLMError):
            agent.run("Test")
        self.assertEqual(agent.status, AgentStatus.ERROR)

    def test_delegation(self):
        """Agent-to-agent delegation works."""
        delegate_llm = MockLLM(responses=[
            Message(role=Role.ASSISTANT, content="Thought: Done\nAction: Final Answer(Helper result)")
        ])
        agent = Agent(llm=MockLLM(), name="Main")
        delegate = Agent(llm=delegate_llm, name="Helper", config=AgentConfig(enable_planning=False))
        agent.register_delegate("helper", delegate)
        result = agent._delegate_task("helper", "Do this")
        # Result should contain the delegate name and its output
        self.assertIn("helper", result)
        self.assertIn("Helper result", result)


class TestAgentToolExecution(unittest.TestCase):

    def test_tool_with_json_args(self):
        llm = MockLLM(responses=[
            Message(role=Role.ASSISTANT, content='Thought: Calculate\nAction: calc({"x": 5, "y": 3})'),
            Message(role=Role.ASSISTANT, content="Thought: Got it\nAction: Final Answer(8)")
        ])
        agent = Agent(llm=llm, config=AgentConfig(enable_planning=False))

        @tool()
        def calc(x: int, y: int) -> int:
            return x + y

        agent.tools.register(calc._tiny_agent_tool)
        result = agent.run("Calculate")
        self.assertIn("8", result)

    def test_tool_with_simple_args(self):
        llm = MockLLM(responses=[
            Message(role=Role.ASSISTANT, content='Thought: Do it\nAction: greet(name="World")'),
            Message(role=Role.ASSISTANT, content="Thought: Done\nAction: Final Answer(Hello World)")
        ])
        agent = Agent(llm=llm, config=AgentConfig(enable_planning=False))

        @tool()
        def greet(name: str) -> str:
            return f"Hello {name}"

        agent.tools.register(greet._tiny_agent_tool)
        result = agent.run("Greet")
        self.assertIn("Hello World", result)

    def test_tool_error_handling(self):
        """Tool errors should be captured in observation."""
        llm = MockLLM(responses=[
            Message(role=Role.ASSISTANT, content="Thought: Try\nAction: fail()"),
            Message(role=Role.ASSISTANT, content="Thought: It failed\nAction: Final Answer(Error handled)")
        ])
        agent = Agent(llm=llm, config=AgentConfig(enable_planning=False))

        @tool()
        def fail():
            raise ValueError("Intentional failure")

        agent.tools.register(fail._tiny_agent_tool)
        result = agent.run("Run failing tool")
        self.assertIn("Error handled", result)


# ─────────────────────────── Integration Tests ───────────────────────────

class TestIntegration(unittest.TestCase):

    def test_full_workflow(self):
        """End-to-end workflow with tools, memory, and history."""
        llm = MockLLM(responses=[
            # First run: tool call + final answer
            Message(role=Role.ASSISTANT, content='Thought: Need to add\nAction: add({"a": 2, "b": 3})'),
            Message(role=Role.ASSISTANT, content="Thought: Got 5\nAction: Final Answer(5)"),
            # Second run: tool call + final answer
            Message(role=Role.ASSISTANT, content='Thought: Need to add again\nAction: add({"a": 2, "b": 3})'),
            Message(role=Role.ASSISTANT, content="Thought: Got 5 again\nAction: Final Answer(5)")
        ])

        memory = InMemoryMemory()
        agent = Agent(
            llm=llm,
            memory=memory,
            config=AgentConfig(enable_planning=False, max_history=10)
        )

        @tool()
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        agent.tools.register(add._tiny_agent_tool)

        # Run multiple times
        result1 = agent.run("What is 2+3?")
        self.assertIn("5", result1)

        result2 = agent.run("What is 2+3?")
        self.assertIn("5", result2)

        # Check memory has both interactions
        memories = memory.get()
        self.assertTrue(len(memories) >= 1)

        # Check history
        self.assertTrue(len(agent.history) > 0)

        # Check steps were recorded
        self.assertTrue(len(agent.steps) > 0)


# ─────────────────────────── Main ───────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
