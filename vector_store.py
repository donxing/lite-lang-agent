import numpy as np
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        try:
            from config import CONFIG
            self.db_path = CONFIG["vectors_db_path"]
            from embedder import Embedder
            self.embedder = Embedder()
            self.index = {}  # In-memory index for quick access
            logger.info("VectorStore initialized")
        except ImportError as e:
            logger.error(f"Failed to initialize VectorStore: {e}")
            raise

    def add_document(self, document_id: str, file_name: str, chunks: List[str], chunk_ids: List[str]):
        try:
            embeddings = self.embedder.embed(chunks)
            logger.info(f"Generated {len(embeddings)} embeddings for document {document_id}")
            for chunk_id, chunk, embedding in zip(chunk_ids, chunks, embeddings):
                self.index[chunk_id] = {
                    "chunk_id": chunk_id,
                    "text": chunk,
                    "document_id": document_id,
                    "embedding": embedding,
                    "score": 1.0
                }
            logger.info(f"Indexed {len(chunks)} chunks for document {document_id} in VectorStore")
            from graph_index import GraphRAGRetriever
            retriever = GraphRAGRetriever()
            retriever.index(document_id, file_name, chunks, chunk_ids)
            logger.info(f"Called GraphRAGRetriever.index for document {document_id}")
        except Exception as e:
            logger.error(f"Failed to add document {document_id} to VectorStore: {e}", exc_info=True)
            raise

    def search(self, query: str, top_k: int) -> List[Dict]:
        try:
            query_embedding = self.embedder.embed([query])[0]
            similarities = {}
            for chunk_id, data in self.index.items():
                chunk_embedding = data["embedding"]
                similarity = np.dot(chunk_embedding, query_embedding) / (
                    np.linalg.norm(chunk_embedding) * np.linalg.norm(query_embedding)
                )
                similarities[chunk_id] = similarity
            sorted_chunks = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
            results = []
            for chunk_id, score in sorted_chunks:
                chunk = self.index[chunk_id].copy()
                chunk["score"] = float(score)
                results.append(chunk)
            logger.info(f"Search retrieved {len(results)} chunks for query: {query[:50]}...")
            return results
        except Exception as e:
            logger.error(f"Search failed for query {query}: {e}")
            raise