import requests
from typing import List, Dict
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, retriever):
        try:
            self.retriever = retriever
            from config import CONFIG
            from knowledge_graph import KnowledgeGraph
            self.ollama_api_url = CONFIG["ollama_api_url"]
            self.lmstudio_api_url = CONFIG["lmstudio_api_url"]
            self.default_llm_model = CONFIG["default_llm_model"]
            self.default_llm_provider = CONFIG["default_llm_provider"]
            self.knowledge_graph = KnowledgeGraph()
            logger.info(f"RAGPipeline initialized with default provider: {self.default_llm_provider}")
        except ImportError as e:
            logger.error(f"Failed to initialize RAGPipeline: {e}")
            raise

    def answer(self, query: str, top_k: int = 3, llm_model: str = None, llm_provider: str = None) -> Dict:
        try:
            model = llm_model or self.default_llm_model
            provider = llm_provider or self.default_llm_provider
            logger.info(f"Using LLM: {model}, provider: {provider}")

            # Retrieve chunks
            chunks = self.retriever.retrieve(query, top_k)
            if not chunks:
                logger.warning(f"No chunks retrieved for query: {query[:50]}...")
                return {"answer": "No relevant information found.", "context": []}

            # Get knowledge graph context
            kg_chunk_ids = self.knowledge_graph.get_related_chunks(query, top_k)
            kg_context = []
            for chunk_id in kg_chunk_ids:
                if chunk_id in self.retriever.chunk_data:
                    kg_context.append(self.retriever.chunk_data[chunk_id]["text"])

            # Prepare context and prompt
            context = "\n".join([chunk["text"] for chunk in chunks])
            kg_context_str = "\n".join(kg_context) if kg_context else "No additional knowledge graph context."
            prompt = f"Context: {context}\nKnowledge Graph Context: {kg_context_str}\n\nQuestion: {query}\n\nAnswer: "

            # Call LLM API
            if provider == "ollama":
                response = requests.post(
                    f"{self.ollama_api_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.7, "max_tokens": 200}
                    }
                )
                response.raise_for_status()
                answer = json.loads(response.text)["response"].strip()
            elif provider == "lmstudio":
                response = requests.post(
                    f"{self.lmstudio_api_url}/v1/completions",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "max_tokens": 200,
                        "temperature": 0.7
                    }
                )
                response.raise_for_status()
                answer = json.loads(response.text)["choices"][0]["text"].strip()
            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")

            logger.info(f"Generated answer for query: {query[:50]}...")
            return {
                "answer": answer,
                "context": [
                    {
                        "chunk_id": chunk["chunk_id"],
                        "text": chunk["text"],
                        "document_id": chunk["document_id"],
                        "score": chunk["score"]
                    }
                    for chunk in chunks
                ]
            }
        except Exception as e:
            logger.error(f"Failed to generate answer for query {query}: {e}")
            raise