from pydantic import BaseModel
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContextChunk(BaseModel):
    chunk_id: str
    text: str
    document_id: str
    score: float

class QueryResponse(BaseModel):
    query: str
    answer: str
    context: List[ContextChunk]

class QueryService:
    def __init__(self):
        try:
            from vector_store import VectorStore
            from rag_infer import RAGPipeline
            self.vector_store = VectorStore()
            # Initialize RAGPipeline with a mock retriever
            class MockRetriever:
                def retrieve(self, query):
                    chunks = self.vector_store.search(query, 3)
                    return chunks
            self.rag_pipeline = RAGPipeline(retriever=MockRetriever(), llm_provider="mock")
            logger.info("VectorStore and RAGPipeline initialized successfully")
        except ImportError as e:
            logger.error(f"Failed to import modules: {e}")
            raise

    async def execute_query(self, query: str, top_k: int) -> QueryResponse:
        try:
            # Update retriever to use specified top_k
            self.rag_pipeline.retriever.retrieve = lambda q: self.vector_store.search(q, top_k)
            answer = self.rag_pipeline.answer(query)
            chunks = self.rag_pipeline.retriever.retrieve(query)
            return QueryResponse(
                query=query,
                answer=answer,
                context=[
                    ContextChunk(chunk_id=c["chunk_id"], text=c["text"], document_id=c["document_id"], score=c["score"])
                    for c in chunks
                ]
            )
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise

    async def recall(self, query: str, top_k: int) -> List[ContextChunk]:
        try:
            results = self.vector_store.search(query, top_k)
            return [
                ContextChunk(chunk_id=res["chunk_id"], text=res["text"], document_id=res["document_id"], score=res["score"])
                for res in results
            ]
        except Exception as e:
            logger.error(f"Error in recall: {e}")
            raise