"""Tests for LLM adapters (mocked)."""
import pytest
from tiny_agent import (
    Message, Role, OpenAIAdapter, AnthropicAdapter,
    OllamaAdapter, HTTPAdapter, UserAdapter,
)
from tiny_agent import LLMError


class MockHTTPAdapter:
    """Spy that records calls instead of making real HTTP requests."""
    def __init__(self, responses=None):
        self.responses = responses or []
        self.calls = []
    
    def _request(self, endpoint, payload):
        self.calls.append({"endpoint": endpoint, "payload": payload})
        return self.responses.pop(0) if self.responses else {
            "choices": [{"message": {"content": "mock response"}}]
        }


class TestOpenAIAdapterSchema:
    """Test adapter behavior without making real API calls."""
    
    def test_adapter_initialization(self):
        adapter = OpenAIAdapter(
            api_key="test-key",
            model="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
        )
        assert adapter.model == "gpt-4o-mini"
        assert adapter.base_url == "https://api.openai.com/v1"

    def test_anthropic_adapter_init(self):
        adapter = AnthropicAdapter(
            api_key="test-key",
            model="claude-3-5-haiku-20240307",
        )
        assert adapter.model == "claude-3-5-haiku-20240307"
        assert adapter.base_url == "https://api.anthropic.com/v1"

    def test_ollama_adapter_init(self):
        adapter = OllamaAdapter(
            model="llama3",
            base_url="http://localhost:11434",
        )
        assert adapter.model == "llama3"
        assert adapter.base_url == "http://localhost:11434"
