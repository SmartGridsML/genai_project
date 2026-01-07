from unittest.mock import MagicMock, patch
from app.services.llm_service import get_llm_service


def _fake_settings():
    s = MagicMock()
    s.openai_api_key = MagicMock()
    s.openai_api_key.get_secret_value.return_value = "fake-key"
    s.openai_model = "gpt-4"
    s.timeout_seconds = 30
    s.max_retries = 1
    s.mlflow_tracking_uri = "file:./test_mlruns"
    s.experiment_name = "test-exp"
    return s


@patch("app.services.llm_service.mlflow")
@patch("app.services.llm_service.get_settings")
def test_count_tokens(mock_get_settings, mock_mlflow):
    mock_get_settings.return_value = _fake_settings()

    llm_service = get_llm_service()
    count = llm_service.count_tokens("Hello world")

    assert isinstance(count, int)
    assert count > 0


@patch("app.services.llm_service.mlflow")
@patch("app.services.llm_service.OpenAI")
@patch("app.services.llm_service.get_settings")
def test_generate_response_success(mock_get_settings, mock_openai_class, mock_mlflow):
    mock_get_settings.return_value = _fake_settings()

    # Make mlflow.start_run act like a context manager
    mock_mlflow.start_run.return_value.__enter__.return_value = None
    mock_mlflow.start_run.return_value.__exit__.return_value = None

    # Mock OpenAI client and response
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "This is a mock response"
    mock_response.usage.completion_tokens = 10
    mock_response.usage.total_tokens = 20

    mock_client.chat.completions.create.return_value = mock_response

    llm_service = get_llm_service()
    result = llm_service.generate_response("System", "User prompt")

    assert result["content"] == "This is a mock response"
    assert result["usage"]["output_tokens"] == 10
    mock_client.chat.completions.create.assert_called_once()
