"""
Tiny Agent — Benchmark Suite

Measures:
- Iterations/sec (ReAct loop iterations)
- Tool calls/sec
- Memory ops/sec
- Agent lifecycle performance

Run: python benchmark.py
"""

import time
import statistics
from typing import Callable, List

from tiny_agent import (
    Agent,
    AgentConfig,
    BaseLLMAdapter,
    InMemoryMemory,
    Message,
    ReActStep,
    Role,
    Tool,
    ToolRegistry,
    tool,
)


# ═══════════════════════════════════════════════════════════════
# Fast Mock LLM (minimal overhead)
# ═══════════════════════════════════════════════════════════════

class FastMockLLM(BaseLLMAdapter):
    """Ultra-fast mock LLM for benchmarking."""

    def __init__(self, response_text: str = "Thought: Done\nAction: Final Answer(42)"):
        self.response_text = response_text
        self.call_count = 0

    def chat(self, messages, tools=None, **kwargs):
        self.call_count += 1
        return Message(role=Role.ASSISTANT, content=self.response_text)

    def stream(self, messages, tools=None, **kwargs):
        for word in self.response_text.split():
            yield word + " "


# ═══════════════════════════════════════════════════════════════
# Benchmark Utilities
# ═══════════════════════════════════════════════════════════════

def benchmark(name: str, fn: Callable, iterations: int = 1000, warmup: int = 100) -> dict:
    """Run a benchmark and return statistics."""
    print(f"  Warming up...", end=" ")
    for _ in range(warmup):
        try:
            fn()
        except Exception:
            pass
    print("done")

    print(f"  Running {iterations} iterations...", end=" ")
    times: List[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        try:
            fn()
        except Exception:
            pass
        end = time.perf_counter()
        times.append(end - start)
    print("done")

    total_time = sum(times)
    avg_time = statistics.mean(times)
    median_time = statistics.median(times)
    min_time = min(times)
    max_time = max(times)

    # Calculate ops/sec
    ops_per_sec = iterations / total_time if total_time > 0 else float('inf')

    return {
        "name": name,
        "iterations": iterations,
        "total_time": total_time,
        "avg_time": avg_time,
        "median_time": median_time,
        "min_time": min_time,
        "max_time": max_time,
        "ops_per_sec": ops_per_sec,
    }


def print_results(results: dict) -> None:
    """Pretty print benchmark results."""
    print(f"\n  {results['name']}:")
    print(f"    Iterations:     {results['iterations']:,}")
    print(f"    Total time:     {results['total_time']:.4f}s")
    print(f"    Avg time:       {results['avg_time']*1000:.4f}ms")
    print(f"    Median time:    {results['median_time']*1000:.4f}ms")
    print(f"    Min time:       {results['min_time']*1000:.4f}ms")
    print(f"    Max time:       {results['max_time']*1000:.4f}ms")
    print(f"    Ops/sec:        {results['ops_per_sec']:,.2f}")


# ═══════════════════════════════════════════════════════════════
# Benchmark Functions
# ═══════════════════════════════════════════════════════════════

def benchmark_react_loop():
    """Benchmark full ReAct loop iterations."""
    llm = FastMockLLM()
    agent = Agent(
        llm=llm,
        config=AgentConfig(enable_planning=False, max_iterations=5)
    )

    def run():
        agent.reset()
        agent.run("Calculate 2+2")

    return benchmark("ReAct Loop (full cycle)", run, iterations=500, warmup=50)


def benchmark_tool_calls():
    """Benchmark tool execution speed."""
    @tool()
    def fast_add(a: int, b: int) -> int:
        return a + b

    tool_obj = fast_add._tiny_agent_tool

    def run():
        tool_obj.call(a=1, b=2)

    return benchmark("Tool Calls (add)", run, iterations=100000, warmup=10000)


def benchmark_memory_ops():
    """Benchmark memory add/get operations."""
    mem = InMemoryMemory()

    def run():
        mem.add(f"Test item at {time.time()}", tag="benchmark")
        mem.get(query="test", limit=5)

    return benchmark("Memory Ops (add + get)", run, iterations=10000, warmup=1000)


def benchmark_message_creation():
    """Benchmark message creation overhead."""
    def run():
        msg = Message(role=Role.USER, content="Hello world")
        msg.to_dict()
        Message.from_dict({"role": "assistant", "content": "Hi!"})

    return benchmark("Message Creation", run, iterations=50000, warmup=5000)


def benchmark_tool_registry():
    """Benchmark tool registry operations."""
    registry = ToolRegistry()

    for i in range(100):
        registry.register(Tool(
            name=f"tool_{i}",
            func=lambda: None,
            description=f"Tool {i}",
            parameters={}
        ))

    def run():
        registry.get("tool_50")
        registry.get_schemas()

    return benchmark("Tool Registry (get + schemas)", run, iterations=10000, warmup=1000)


def benchmark_streaming():
    """Benchmark streaming response speed."""
    llm = FastMockLLM("This is a test streaming response with multiple words")
    agent = Agent(
        llm=llm,
        config=AgentConfig(enable_planning=False, max_iterations=5)
    )

    def run():
        list(agent.run("Test", stream=True))

    return benchmark("Streaming (full response)", run, iterations=500, warmup=50)


def benchmark_agent_lifecycle():
    """Benchmark agent creation and destruction."""
    llm = FastMockLLM()

    def run():
        agent = Agent(
            llm=llm,
            config=AgentConfig(enable_planning=False)
        )
        agent.run("Test")
        agent.reset()

    return benchmark("Agent Lifecycle (create + run + reset)", run, iterations=500, warmup=50)


def benchmark_planning():
    """Benchmark planning step."""
    llm = FastMockLLM('["Step 1", "Step 2", "Step 3"]')
    agent = Agent(
        llm=llm,
        config=AgentConfig(enable_planning=True)
    )

    def run():
        agent._create_plan("Solve this complex problem")

    return benchmark("Planning (task decomposition)", run, iterations=1000, warmup=100)


def benchmark_observability():
    """Benchmark event system performance."""
    llm = FastMockLLM()
    agent = Agent(llm=llm, config=AgentConfig(enable_planning=False))

    # Register multiple handlers
    for i in range(10):
        agent.on("test_event", lambda x, i=i: x + i)

    def run():
        agent.emit("test_event", 1)

    return benchmark("Observability (10 handlers)", run, iterations=50000, warmup=5000)


def benchmark_comparison():
    """Compare sync vs streaming performance."""
    llm = FastMockLLM()

    # Sync
    agent_sync = Agent(llm=llm, config=AgentConfig(enable_planning=False))

    def run_sync():
        agent_sync.run("Test")

    result_sync = benchmark("Sync Response", run_sync, iterations=500, warmup=50)

    # Stream
    agent_stream = Agent(llm=llm, config=AgentConfig(enable_planning=False))

    def run_stream():
        list(agent_stream.run("Test", stream=True))

    result_stream = benchmark("Streaming Response", run_stream, iterations=500, warmup=50)

    return result_sync, result_stream


# ═══════════════════════════════════════════════════════════════
# Main Benchmark Runner
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("🚀 Tiny Agent Benchmark Suite")
    print("=" * 60)
    print(f"Python: Measuring performance...")
    print()

    all_results = []

    # Core operations
    print("Core Operations:")
    print("-" * 40)
    all_results.append(benchmark_message_creation())
    all_results.append(benchmark_tool_calls())
    all_results.append(benchmark_memory_ops())
    all_results.append(benchmark_tool_registry())

    for r in all_results:
        print_results(r)

    # Agent operations
    print("\n" + "=" * 60)
    print("Agent Operations:")
    print("-" * 40)
    agent_results = []
    agent_results.append(benchmark_react_loop())
    agent_results.append(benchmark_agent_lifecycle())
    agent_results.append(benchmark_streaming())
    agent_results.append(benchmark_planning())

    for r in agent_results:
        print_results(r)

    all_results.extend(agent_results)

    # Observability
    print("\n" + "=" * 60)
    print("Observability:")
    print("-" * 40)
    obs_result = benchmark_observability()
    print_results(obs_result)
    all_results.append(obs_result)

    # Comparison
    print("\n" + "=" * 60)
    print("Comparison:")
    print("-" * 40)
    sync_result, stream_result = benchmark_comparison()
    print_results(sync_result)
    print_results(stream_result)
    all_results.extend([sync_result, stream_result])

    # Summary
    print("\n" + "=" * 60)
    print("📊 Summary")
    print("=" * 60)
    print(f"{'Benchmark':<35} {'Ops/sec':>15} {'Avg (ms)':>12}")
    print("-" * 60)
    for r in all_results:
        name = r['name'][:34]
        ops = f"{r['ops_per_sec']:,.2f}"
        avg = f"{r['avg_time']*1000:.4f}"
        print(f"{name:<35} {ops:>15} {avg:>12}")

    print()
    print("=" * 60)
    print("✅ Benchmark complete!")
    print("=" * 60)

    # Performance notes
    print("\n📝 Notes:")
    print("  - Results vary based on hardware and Python version")
    print("  - Tool calls/sec is limited by Python function call overhead")
    print("  - Memory ops include both add and get operations")
    print("  - ReAct loop includes full LLM call + parsing + execution")
    print("  - For production use, consider using PyPy for 2-5x speedup")


if __name__ == "__main__":
    main()
