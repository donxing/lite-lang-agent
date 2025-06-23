import networkx as nx
import numpy as np
from typing import List, Dict
import sqlite3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphRAGRetriever:
    def __init__(self):
        try:
            from embedder import Embedder
            from config import CONFIG
            self.embedder = Embedder()
            self.graph = nx.Graph()
            self.chunk_data: Dict[str, Dict] = {}
            self.db_path = CONFIG["vectors_db_path"]
            self._load_from_db()
            logger.info("GraphRAGRetriever initialized with Embedder")
        except ImportError as e:
            logger.error(f"Failed to initialize GraphRAGRetriever: {e}")
            raise

    def _load_from_db(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT chunk_id, text, document_id, embedding FROM chunks")
                rows = cursor.fetchall()
                for chunk_id, text, document_id, embedding_blob in rows:
                    embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                    self.chunk_data[chunk_id] = {
                        "chunk_id": chunk_id,
                        "text": text,
                        "document_id": document_id,
                        "embedding": embedding,
                        "score": 1.0
                    }
                    self.graph.add_node(chunk_id, text=text, embedding=embedding, document_id=document_id)
                    # Add edges between consecutive chunks in the same document
                    doc_chunks = [row[0] for row in rows if row[2] == document_id]
                    idx = doc_chunks.index(chunk_id)
                    if idx > 0:
                        prev_chunk_id = doc_chunks[idx-1]
                        self.graph.add_edge(chunk_id, prev_chunk_id, weight=1.0)
                logger.info(f"Loaded {len(rows)} chunks from vectors.db")
        except sqlite3.Error as e:
            logger.error(f"Failed to load chunks from vectors.db: {e}")
            raise

    def index(self, document_id: str, file_name: str, chunks: List[str], chunk_ids: List[str]):
        try:
            # Embed chunks
            embeddings = self.embedder.embed(chunks)
            for idx, (text, embedding, chunk_id) in enumerate(zip(chunks, embeddings, chunk_ids)):
                self.graph.add_node(chunk_id, text=text, embedding=embedding, document_id=document_id)
                self.chunk_data[chunk_id] = {
                    "chunk_id": chunk_id,
                    "text": text,
                    "document_id": document_id,
                    "embedding": embedding,
                    "score": 1.0
                }
                # Add edges between consecutive chunks
                if idx > 0:
                    prev_chunk_id = chunk_ids[idx-1]
                    self.graph.add_edge(chunk_id, prev_chunk_id, weight=1.0)
            logger.info(f"Indexed {len(chunks)} chunks for document {document_id}")
        except Exception as e:
            logger.error(f"Failed to index document {document_id}: {e}")
            raise

    def retrieve(self, query: str, top_k: int) -> List[Dict]:
        try:
            if not self.chunk_data:
                logger.warning("No chunks indexed for retrieval")
                return []
            # Embed query
            query_embedding = self.embedder.embed([query])[0]
            # Compute similarities
            similarities = {}
            for chunk_id, data in self.chunk_data.items():
                chunk_embedding = data["embedding"]
                similarity = np.dot(chunk_embedding, query_embedding) / (
                    np.linalg.norm(chunk_embedding) * np.linalg.norm(query_embedding)
                )
                similarities[chunk_id] = similarity
            # Rank and select top_k chunks
            sorted_chunks = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
            results = []
            for chunk_id, score in sorted_chunks:
                chunk = self.chunk_data[chunk_id].copy()
                chunk["score"] = float(score)
                # Boost score for connected chunks
                neighbors = list(self.graph.neighbors(chunk_id))
                chunk["score"] *= (1 + 0.1 * len(neighbors))
                results.append(chunk)
            logger.info(f"Retrieved {len(results)} chunks for query: {query[:50]}...")
            return results
        except Exception as e:
            logger.error(f"Retrieval failed for query {query}: {e}")
            raise