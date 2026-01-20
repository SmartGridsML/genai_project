from unittest.mock import MagicMock, patch
from backend.app.services.llm_service import get_llm_service


def _fake_settings(with_gemini=True):
    s = MagicMock()
    s.openai_api_key = MagicMock()
    s.openai_api_key.get_secret_value.return_value = "fake-openai-key"
    s.gemini_api_key = MagicMock() if with_gemini else None
    if with_gemini:
        s.gemini_api_key.get_secret_value.return_value = "fake-gemini-key"
    s.openai_model = "gpt-4o"
    s.gemini_model = "gemini-1.5-flash"
    s.timeout_seconds = 30
    s.max_retries = 1
    s.mlflow_tracking_uri = "file:./test_mlruns"
    s.experiment_name = "test-exp"
    return s


@patch("backend.app.services.llm_service.mlflow")
@patch("backend.app.services.llm_service.get_settings")
@patch("backend.app.services.llm_service.ChatOpenAI")
@patch("backend.app.services.llm_service.ChatGoogleGenerativeAI")
def test_count_tokens(mock_gemini, mock_openai, mock_get_settings, mock_mlflow):
    """Test token counting functionality."""
    mock_get_settings.return_value = _fake_settings()

    # Mock LangChain clients initialization
    mock_openai.return_value = MagicMock()
    mock_gemini.return_value = MagicMock()

    llm_service = get_llm_service()
    count = llm_service.count_tokens("Hello world")

    assert isinstance(count, int)
    assert count > 0


@patch("backend.app.services.llm_service.mlflow")
@patch("backend.app.services.llm_service.get_settings")
@patch("backend.app.services.llm_service.ChatOpenAI")
@patch("backend.app.services.llm_service.ChatGoogleGenerativeAI")
def test_generate_response_success(mock_gemini, mock_openai, mock_get_settings, mock_mlflow):
    """Test successful response generation using OpenAI."""
    mock_get_settings.return_value = _fake_settings()

    # Make mlflow.start_run act like a context manager
    mock_mlflow.start_run.return_value.__enter__.return_value = None
    mock_mlflow.start_run.return_value.__exit__.return_value = None

    # Mock OpenAI client and response
    mock_openai_client = MagicMock()
    mock_openai.return_value = mock_openai_client

    mock_response = MagicMock()
    mock_response.content = "This is a mock response from OpenAI"
    mock_openai_client.invoke.return_value = mock_response

    # Mock Gemini client (shouldn't be called in success case)
    mock_gemini_client = MagicMock()
    mock_gemini.return_value = mock_gemini_client

    llm_service = get_llm_service()
    result = llm_service.generate_response("System prompt", "User prompt")

    # Verify response structure
    assert "content" in result
    assert result["content"] == "This is a mock response from OpenAI"
    assert "usage" in result
    assert "provider" in result
    assert result["provider"] == "openai"

    # OpenAI should have been called
    mock_openai_client.invoke.assert_called_once()

    # Gemini should NOT have been called (no fallback needed)
    mock_gemini_client.invoke.assert_not_called()


@patch("backend.app.services.llm_service.mlflow")
@patch("backend.app.services.llm_service.get_settings")
@patch("backend.app.services.llm_service.ChatOpenAI")
@patch("backend.app.services.llm_service.ChatGoogleGenerativeAI")
def test_fallback_to_gemini(mock_gemini, mock_openai, mock_get_settings, mock_mlflow):
    """Test fallback to Gemini when OpenAI fails."""
    mock_get_settings.return_value = _fake_settings()

    # Make mlflow.start_run act like a context manager
    mock_mlflow.start_run.return_value.__enter__.return_value = None
    mock_mlflow.start_run.return_value.__exit__.return_value = None

    # Mock OpenAI client to fail
    mock_openai_client = MagicMock()
    mock_openai.return_value = mock_openai_client
    mock_openai_client.invoke.side_effect = Exception("OpenAI API error")

    # Mock Gemini client to succeed
    mock_gemini_client = MagicMock()
    mock_gemini.return_value = mock_gemini_client

    mock_gemini_response = MagicMock()
    mock_gemini_response.content = "This is a mock response from Gemini"
    mock_gemini_client.invoke.return_value = mock_gemini_response

    llm_service = get_llm_service()
    result = llm_service.generate_response("System prompt", "User prompt")

    # Verify response structure
    assert "content" in result
    assert result["content"] == "This is a mock response from Gemini"
    assert "provider" in result
    assert result["provider"] == "gemini"

    # Both should have been called (OpenAI failed, Gemini succeeded)
    mock_openai_client.invoke.assert_called_once()
    mock_gemini_client.invoke.assert_called_once()
