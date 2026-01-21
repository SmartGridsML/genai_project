class NullLLMService:
    """Non-None default for tests/imports. Fails only when called."""
    def generate_response(self, *args, **kwargs):
        raise RuntimeError("LLM not configured. Set OPENAI_API_KEY or GEMINI_API_KEY.")
