# How to Implement `llm_service.py` as a 10x Engineer

This guide outlines the architecture for a production-grade LLM service. It moves beyond simple API calls to building a robust, observable I/O engine.

## 1. The Architecture: Class-Based & Stateful

**Why:** Initialization is expensive. You don't want to re-authenticate or re-load tokenizers for every single request.

*   **Pattern:** Singleton or Dependency Injection.
*   **Action:** Create a `LLMService` class.
*   **Constructor (`__init__`)**:
    *   Initialize the `OpenAI` client using your Pydantic settings.
    *   Initialize the Tokenizer (e.g., `tiktoken.encoding_for_model`).
    *   Initialize your Experiment Tracker (MLflow) configuration.

## 2. Guardrails: Input Validation & Token Counting

**Why:** API failures often happen because inputs exceed context windows. Also, without tracking tokens *before* the call, you can't estimate costs or prevent budget runaways.

*   **Tool:** `tiktoken` (for OpenAI).
*   **Action:** Implement a `count_tokens(text: str) -> int` helper method.
*   **Usage:** Call this *before* sending the request. Log the input size.

## 3. Resilience: The Decorator Pattern

**Why:** Network calls are flaky. Writing nested `try/except` loops with `time.sleep` is messy and error-prone.

*   **Tool:** `tenacity` library.
*   **Action:** Decorate your main generation method with `@retry`.
*   **Configuration:**
    *   **Wait Strategy:** Exponential Backoff (`wait_exponential`). Wait 1s, then 2s, then 4s. This prevents thundering herd problems.
    *   **Stop Strategy:** Stop after `X` attempts (e.g., 3 or 5).
    *   **Filter:** Only retry on "retriable" errors (Timeouts, 500s), NOT on fatal errors (401 Unauthorized, 400 Bad Request).

## 4. Observability: The "Black Box" Problem

**Why:** When a request fails or costs spike, "logging text" isn't enough. You need structured data to query.

*   **Tool:** `mlflow` (or LangSmith/Arize).
*   **Action:** Wrap the API call in a `with mlflow.start_run():` block.
*   **What to Log:**
    *   **Parameters:** `temperature`, `model_name`, `max_tokens`.
    *   **Metrics:** `latency_seconds`, `input_tokens`, `output_tokens`, `total_cost`.
    *   **Artifacts:** The actual prompt and response text (optional, be careful with PII).

## 5. The "Rich Return" Object

**Why:** Returning a simple string (`"Hello"`) hides crucial metadata needed for debugging and billing.

*   **Action:** Return a Dictionary or Data Class, never just a string.
*   **Structure:**
    ```python
    {
        "content": "The actual LLM response",
        "usage": {
            "input_tokens": 150,
            "output_tokens": 50,
            "total_tokens": 200
        },
        "model": "gpt-4-0613",
        "latency": 1.25  # seconds
    }
    ```

## Summary Checklist

- [ ]  Class-based structure initialized with config.
- [ ]  `tiktoken` integrated for accurate counting.
- [ ]  `@retry` decorator from `tenacity` with exponential backoff.
- [ ]  `mlflow` context manager for tracking every call.
- [ ]  Returns a structured object (Content + Metadata).
