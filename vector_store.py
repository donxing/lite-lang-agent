from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        try:
            from graph_index import GraphRAGRetriever
            self.retriever = GraphRAGRetriever()
            logger.info("VectorStore initialized with GraphRAGRetriever")
        except ImportError as e:
            logger.error(f"Failed to initialize VectorStore: {e}")
            raise

    def add_document(self, document_id: str, file_name: str, chunks: List[str], chunk_ids: List[str]):
        try:
            self.retriever.index(document_id, file_name, chunks, chunk_ids)
            logger.info(f"Added {len(chunks)} chunks for document {document_id}")
        except Exception as e:
            logger.error(f"Failed to add document {document_id}: {e}")
            raise

    def search(self, query: str, top_k: int) -> List[Dict]:
        try:
            results = self.retriever.retrieve(query, top_k)
            if not results:
                logger.warning(f"No chunks retrieved for query: {query[:50]}...")
            return results
        except Exception as e:
            logger.error(f"Search failed for query {query}: {e}")
            raise