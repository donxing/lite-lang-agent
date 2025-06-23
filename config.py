import os

# Configuration for the RAG application
CONFIG = {
    "embedder_model_path": os.path.expanduser("~/.cache/modelscope/hub/models/sentence-transformers/all-MiniLM-L6-v2"),
    "embedder_model_name": "all-MiniLM-L6-v2",
    "ollama_api_url": "http://localhost:11434",
    "lmstudio_api_url": "http://localhost:1234",
    "default_llm_model": "llama3.2:3b",
    "default_llm_provider": "ollama",
    "vectors_db_path": "vectors.db",
    "workflows_db_path": "workflows.db",
    "chunk_size": 300,
    "chunk_overlap": 50
}