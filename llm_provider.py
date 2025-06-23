# llm_provider.py
class LLMProvider:
    def generate(self, prompt: str) -> str:
        return f"Mock response to prompt: {prompt[:50]}..."