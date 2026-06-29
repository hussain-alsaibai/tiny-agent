"""
Tiny Agent — Working Example

A calculator agent with:
- Math tools (add, multiply, divide, sqrt)
- Memory across conversations
- Multi-step reasoning

Run: python example.py
"""

from tiny_agent import Agent, InMemoryMemory, AgentConfig, ToolRegistry, tool


# ═══════════════════════════════════════════════════════════════
# Step 1: Define Tools
# ═══════════════════════════════════════════════════════════════

@tool()
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@tool()
def divide(a: float, b: float) -> float:
    """Divide two numbers. Returns error if dividing by zero."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


@tool()
def sqrt(x: float) -> float:
    """Calculate square root of a number."""
    if x < 0:
        raise ValueError("Cannot calculate square root of negative number")
    import math
    return math.sqrt(x)


@tool()
def power(base: float, exponent: float) -> float:
    """Raise base to the power of exponent."""
    return base ** exponent


# ═══════════════════════════════════════════════════════════════
# Step 2: Create a Mock LLM for Demo (no API key needed!)
# ═══════════════════════════════════════════════════════════════

from tiny_agent import BaseLLMAdapter, Message, Role


class DemoLLM(BaseLLMAdapter):
    """
    A demo LLM that simulates reasoning without calling a real API.
    This is for demonstration purposes — in production, use OpenAIAdapter,
    AnthropicAdapter, or OllamaAdapter.
    """

    def __init__(self):
        self.call_count = 0

    def chat(self, messages, tools=None, **kwargs):
        """Simulate LLM responses based on the conversation context."""
        self.call_count += 1

        # Get the last user message
        user_msgs = [m for m in messages if m.role == Role.USER]
        last_query = user_msgs[-1].content if user_msgs else ""

        # Extract query from ReAct prompt
        if "User query:" in last_query:
            query = last_query.split("User query:")[-1].strip()
        else:
            query = last_query

        # Simple pattern matching for demo
        query_lower = query.lower()

        # Check for final answer patterns
        if "final answer" in query_lower:
            return Message(role=Role.ASSISTANT, content="Thought: Done\nAction: Final Answer(Complete)")

        # Determine which tool to call based on query
        if "sqrt" in query_lower or "square root" in query_lower:
            # Extract number
            import re
            nums = re.findall(r"\d+\.?\d*", query)
            if nums:
                return Message(
                    role=Role.ASSISTANT,
                    content=f"Thought: Need to calculate square root of {nums[0]}\nAction: sqrt(x={nums[0]})"
                )

        if "power" in query_lower or "^" in query_lower:
            import re
            nums = re.findall(r"\d+\.?\d*", query)
            if len(nums) >= 2:
                return Message(
                    role=Role.ASSISTANT,
                    content=f"Thought: Need to calculate {nums[0]}^{nums[1]}\nAction: power(base={nums[0]}, exponent={nums[1]})"
                )

        if "divide" in query_lower or "/" in query_lower:
            import re
            nums = re.findall(r"\d+\.?\d*", query)
            if len(nums) >= 2:
                return Message(
                    role=Role.ASSISTANT,
                    content=f"Thought: Need to divide {nums[0]} by {nums[1]}\nAction: divide(a={nums[0]}, b={nums[1]})"
                )

        if "multiply" in query_lower or "*" in query_lower:
            import re
            nums = re.findall(r"\d+\.?\d*", query)
            if len(nums) >= 2:
                return Message(
                    role=Role.ASSISTANT,
                    content=f"Thought: Need to multiply {nums[0]} by {nums[1]}\nAction: multiply(a={nums[0]}, b={nums[1]})"
                )

        if "add" in query_lower or "+" in query_lower or "sum" in query_lower:
            import re
            nums = re.findall(r"\d+\.?\d*", query)
            if len(nums) >= 2:
                return Message(
                    role=Role.ASSISTANT,
                    content=f"Thought: Need to add {nums[0]} and {nums[1]}\nAction: add(a={nums[0]}, b={nums[1]})"
                )

        # Multi-step: Check if we have a previous result to continue
        if "more" in query_lower or "then" in query_lower or "plus" in query_lower:
            # Look for previous result in conversation
            for m in reversed(messages):
                if m.role == Role.TOOL and any(c.isdigit() for c in m.content):
                    prev_result = m.content
                    import re
                    nums = re.findall(r"\d+\.?\d*", query)
                    if nums:
                        return Message(
                            role=Role.ASSISTANT,
                            content=f"Thought: Previous result was {prev_result}. Now I need to add {nums[-1]}\nAction: add(a={prev_result}, b={nums[-1]})"
                        )

        # Multi-step word problem
        if "area" in query_lower and "rectangle" in query_lower:
            import re
            nums = re.findall(r"\d+\.?\d*", query)
            if len(nums) >= 2:
                return Message(
                    role=Role.ASSISTANT,
                    content=f"Thought: Area = width * height = {nums[0]} * {nums[1]}\nAction: multiply(a={nums[0]}, b={nums[1]})"
                )

        # Compound calculation: 2 + 3 * 4
        if "2 + 3 * 4" in query_lower or "(2 + 3) * 4" in query_lower:
            if "(2 + 3)" in query_lower:
                return Message(
                    role=Role.ASSISTANT,
                    content="Thought: First add 2 + 3, then multiply by 4\nAction: add(a=2, b=3)"
                )
            else:
                return Message(
                    role=Role.ASSISTANT,
                    content="Thought: Following order of operations, first multiply 3 * 4 = 12, then add 2\nAction: multiply(a=3, b=4)"
                )

        # Default: provide a final answer
        return Message(role=Role.ASSISTANT, content="Thought: I'll provide the best answer I can\nAction: Final Answer(I need more specific instructions)")

    def stream(self, messages, tools=None, **kwargs):
        """Simulate streaming by yielding words."""
        msg = self.chat(messages, tools, **kwargs)
        for word in msg.content.split():
            yield word + " "


# ═══════════════════════════════════════════════════════════════
# Step 3: Create the Agent
# ═══════════════════════════════════════════════════════════════

memory = InMemoryMemory()
llm = DemoLLM()

agent = Agent(
    llm=llm,
    memory=memory,
    config=AgentConfig(
        system_prompt="""You are a calculator agent. Use mathematical tools to solve problems.
Always show your reasoning step by step. If a calculation has multiple steps,
perform them one at a time.""",
        max_iterations=10,
        enable_planning=True,
        max_history=20
    ),
    name="CalculatorAgent"
)

# Register tools
agent.tools.register(add._tiny_agent_tool)
agent.tools.register(multiply._tiny_agent_tool)
agent.tools.register(divide._tiny_agent_tool)
agent.tools.register(sqrt._tiny_agent_tool)
agent.tools.register(power._tiny_agent_tool)

print("=" * 60)
print("🧮 Tiny Agent Calculator Demo")
print("=" * 60)
print()

# ═══════════════════════════════════════════════════════════════
# Example 1: Simple Addition
# ═══════════════════════════════════════════════════════════════
print("Example 1: Simple Addition")
print("-" * 40)
result = agent.run("What is 15 + 27?")
print(f"Result: {result}")
print(f"Steps taken: {len(agent.steps)}")
print()

# ═══════════════════════════════════════════════════════════════
# Example 2: Multi-step Reasoning (Order of Operations)
# ═══════════════════════════════════════════════════════════════
print("Example 2: Multi-step Reasoning")
print("-" * 40)
agent.reset()
result = agent.run("Calculate 2 + 3 * 4 following order of operations")
print(f"Result: {result}")
print(f"Steps taken: {len(agent.steps)}")
for step in agent.steps:
    print(f"  Step {step.iteration}: {step.action} -> {step.observation}")
print()

# ═══════════════════════════════════════════════════════════════
# Example 3: Memory Across Conversations
# ═══════════════════════════════════════════════════════════════
print("Example 3: Memory Across Conversations")
print("-" * 40)
agent.reset()

# First conversation
result1 = agent.run("Calculate the area of a rectangle with width 5 and height 3")
print(f"Conversation 1: {result1}")
print(f"Memory entries: {len(memory.get())}")
print()

# Second conversation - agent remembers context!
result2 = agent.run("What if the width was 8 instead?")
print(f"Conversation 2: {result2}")
print(f"Memory entries: {len(memory.get())}")
print()

# Show what's in memory
print("Memory contents:")
for item in memory.get(limit=5):
    print(f"  - {item['content'][:60]}...")
print()

# ═══════════════════════════════════════════════════════════════
# Example 4: Complex Calculation with Multiple Steps
# ═══════════════════════════════════════════════════════════════
print("Example 4: Complex Calculation")
print("-" * 40)
agent.reset()
result = agent.run("Calculate (2 + 3) * 4")
print(f"Result: {result}")
print(f"Steps taken: {len(agent.steps)}")
for step in agent.steps:
    print(f"  Step {step.iteration}: {step.action}")
print()

# ═══════════════════════════════════════════════════════════════
# Example 5: Streaming Demo
# ═══════════════════════════════════════════════════════════════
print("Example 5: Streaming Response")
print("-" * 40)
agent.reset()
print("Response: ", end="", flush=True)
for chunk in agent.run("What is 10 * 10?", stream=True):
    print(chunk, end="", flush=True)
print()
print()

# ═══════════════════════════════════════════════════════════════
# Example 6: Error Handling
# ═══════════════════════════════════════════════════════════════
print("Example 6: Error Handling (Division by Zero)")
print("-" * 40)
agent.reset()
result = agent.run("What is 10 / 0?")
print(f"Result: {result}")
print()

# ═══════════════════════════════════════════════════════════════
# Example 7: Square Root and Power
# ═══════════════════════════════════════════════════════════════
print("Example 7: Advanced Math (Square Root)")
print("-" * 40)
agent.reset()
result = agent.run("What is the square root of 144?")
print(f"Result: {result}")
print()

print("Example 8: Advanced Math (Power)")
print("-" * 40)
agent.reset()
result = agent.run("What is 2 to the power of 10?")
print(f"Result: {result}")
print()

# ═══════════════════════════════════════════════════════════════
# Show conversation history
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("📜 Conversation History Summary")
print("=" * 60)
history = agent.get_history()
print(f"Total messages in history: {len(history)}")
print()

print("=" * 60)
print("✅ Demo complete!")
print("=" * 60)
print()
print("To use with a real LLM, replace DemoLLM with:")
print("  - OpenAIAdapter(api_key='your-key')")
print("  - AnthropicAdapter(api_key='your-key')")
print("  - OllamaAdapter(model='llama3')")
print()
