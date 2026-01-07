import pytest
from unittest.mock import MagicMock, patch
from app.services.llm_service import LLMService

# Mock the settings so we don't need a real API key for unit tests
@patch("app.services.llm_service.settings")
def test_count_tokens(mock_settings):
    # Setup
    mock_settings.openai_api_key.get_secret_value.return_value = "fake-key"
    mock_settings.openai_model = "gpt-4"
    mock_settings.mlflow_tracking_uri = "file:./test_mlruns"
    
    service = LLMService()
    text = "Hello world"
    
    # Execution
    count = service.count_tokens(text)
    
    # Assertion
    assert isinstance(count, int)
    assert count > 0

@patch("app.services.llm_service.OpenAI")
@patch("app.services.llm_service.settings")
def test_generate_response_success(mock_settings, mock_openai_class):
    # 1. Setup Mocks
    mock_settings.openai_api_key.get_secret_value.return_value = "fake-key"
    mock_settings.max_retries = 1
    
    # Mock the client instance and its method chain: client.chat.completions.create
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    
    # Mock the API response object
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "This is a mock response"
    mock_response.usage.completion_tokens = 10
    mock_response.usage.total_tokens = 20
    
    mock_client.chat.completions.create.return_value = mock_response
    
    # 2. Initialize Service
    service = LLMService()
    
    # 3. Execute
    result = service.generate_response("System", "User prompt")
    
    # 4. Verify
    assert result["content"] == "This is a mock response"
    assert result["usage"]["output_tokens"] == 10
    
    # Verify OpenAI was called correctly
    mock_client.chat.completions.create.assert_called_once()
