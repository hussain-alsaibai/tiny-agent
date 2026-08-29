"""Integration tests for tiny-agent (require API keys)."""
import pytest
import os

@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY") == "test-key-12345",
    reason="Requires real API key"
)
class TestAgentIntegration:
    """Run only when OPENAI_API_KEY is set to a real key."""
    
    def test_simple_agent_run(self):
        from tiny_agent import Agent, OpenAIAdapter, tool, AgentConfig
        
        @tool()
        def add(a: int, b: int) -> int:
            """Add two integers."""
            return a + b
        
        agent = Agent(
            llm=OpenAIAdapter(
                api_key=os.environ["OPENAI_API_KEY"],
                model="gpt-4o-mini",
            ),
            config=AgentConfig(max_iterations=3, system_prompt="You are a calculator assistant."),
        )
        agent.tools.register(add._tiny_agent_tool)
        result = agent.run("What is 5 + 7? Use the add tool.")
        assert result
