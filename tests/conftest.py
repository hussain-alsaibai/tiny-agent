"""Pytest configuration and shared fixtures."""
import pytest
import tempfile
import os

@pytest.fixture
def temp_dir():
    """Provide a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as d:
        yield d

@pytest.fixture
def mock_api_key(monkeypatch):
    """Mock API key for testing."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-12345")
