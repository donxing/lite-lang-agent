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
            from graph_index import GraphRAGRetriever
            self.vector_store = VectorStore()
            self.rag_pipeline = RAGPipeline(retriever=GraphRAGRetriever())
            logger.info("QueryService initialized with VectorStore and RAGPipeline")
        except ImportError as e:
            logger.error(f"Failed to import modules: {e}")
            raise

    async def execute_query(self, query: str, top_k: int, llm_model: str = None, llm_provider: str = None) -> QueryResponse:
        try:
            logger.info(f"Executing query: {query[:50]}... with model: {llm_model or 'default'}, provider: {llm_provider or 'default'}")
            result = self.rag_pipeline.answer(query, top_k, llm_model, llm_provider)
            logger.info(f"Query executed: {query[:50]}..., retrieved {len(result['context'])} chunks")
            return QueryResponse(
                query=query,
                answer=result["answer"],
                context=[
                    ContextChunk(chunk_id=c["chunk_id"], text=c["text"], document_id=c["document_id"], score=c["score"])
                    for c in result["context"]
                ]
            )
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise

    async def recall(self, query: str, top_k: int) -> List[ContextChunk]:
        try:
            logger.info(f"Executing recall: {query[:50]}...")
            results = self.vector_store.search(query, top_k)
            if not results:
                logger.warning(f"No chunks retrieved for query: {query[:50]}...")
            else:
                logger.info(f"Recall retrieved {len(results)} chunks for query: {query[:50]}...")
            return [
                ContextChunk(chunk_id=res["chunk_id"], text=res["text"], document_id=res["document_id"], score=res["score"])
                for res in results
            ]
        except Exception as e:
            logger.error(f"Error in recall: {e}")
            raise