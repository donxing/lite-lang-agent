import requests
import json

class RAGPipeline:
    def __init__(self, retriever: 'GraphRAGRetriever', llm_provider: str = "mock", model: str = "llama3"):
        self.retriever = retriever
        self.llm_provider = llm_provider.lower()
        self.model = model
        self.llm = self._get_llm()

    def _get_llm(self):
        """Return a callable LLM based on the provider."""
        if self.llm_provider == "ollama":
            def ollama_llm(prompt):
                try:
                    response = requests.post(
                        "http://localhost:11434/api/generate",
                        json={
                            "model": self.model,
                            "prompt": prompt,
                            "stream": False
                        },
                        timeout=30
                    )
                    response.raise_for_status()
                    return json.loads(response.text)["response"]
                except Exception as e:
                    return f"Error with Ollama LLM: {str(e)}"
            return ollama_llm

        elif self.llm_provider == "lmstudio":
            def lmstudio_llm(prompt):
                try:
                    response = requests.post(
                        "http://localhost:1234/v1/completions",
                        json={
                            "model": self.model,
                            "prompt": prompt,
                            "max_tokens": 512,
                            "temperature": 0.7
                        },
                        timeout=30
                    )
                    response.raise_for_status()
                    return json.loads(response.text)["choices"][0]["text"]
                except Exception as e:
                    return f"Error with LMStudio LLM: {str(e)}"
            return lmstudio_llm

        else:  # Default to mock
            def mock_llm(prompt):
                return "（模拟回答）这是一个基于上下文的回答。"
            return mock_llm

    def answer(self, query):
        context_chunks = self.retriever.retrieve(query)
        context = "\n\n".join([chunk["text"] for chunk in context_chunks])

        prompt = f"""Answer the question based on the following context:\n\n{context}\n\nQuestion: {query}\nAnswer:"""
        return self.llm(prompt)