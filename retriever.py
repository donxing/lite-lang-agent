import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
from storage import VectorStore
from embedder import Embedder

class GraphRAGRetriever:
    def __init__(self, store: VectorStore, embedder: Embedder):
        self.store = store
        self.embedder = embedder

    def retrieve(self, query, top_k=3, document_ids=None):
        """
        Retrieve relevant chunks for a query, optionally filtered by document IDs.
        Returns chunk IDs, texts, document IDs, and similarity scores.
        """
        query_vec = self.embedder.embed([query])[0]
        all_chunks = self.store.get_all_chunks()

        scored = []
        for chunk_id, text, emb_str in all_chunks:
            # Extract document_id from chunk_id (e.g., "doc_id_chunk_0" -> "doc_id")
            document_id = chunk_id.split("_chunk_")[0]
            if document_ids and document_id not in document_ids:
                continue
            emb = np.array(json.loads(emb_str))
            score = cosine_similarity([query_vec], [emb])[0][0]
            scored.append((score, chunk_id, text, document_id))

        scored.sort(reverse=True)
        top_chunks = scored[:top_k]

        # Expand neighbors
        expanded_ids = set([chunk_id for _, chunk_id, _, _ in top_chunks])
        for _, chunk_id, _, _ in top_chunks:
            expanded_ids.update(self.store.get_neighbors(chunk_id))

        # Return results with metadata
        results = []
        for chunk_id in expanded_ids:
            text = self.store.get_chunk_by_id(chunk_id)
            if text:
                document_id = chunk_id.split("_chunk_")[0]
                score = next((s for s, cid, _, _ in scored if cid == chunk_id), 0.0)
                results.append({
                    "chunk_id": chunk_id,
                    "text": text,
                    "document_id": document_id,
                    "score": score
                })
        return results