# vector_store.py
class VectorStore:
    def __init__(self):
        self.documents = {}
        self.chunks = []

    def add_document(self, document_id: str, file_name: str, chunks: list):
        self.documents[document_id] = {"file_name": file_name, "chunks": chunks}
        for idx, text in enumerate(chunks):
            self.chunks.append({
                "chunk_id": f"{document_id}_{idx}",
                "text": text,
                "document_id": document_id,
                "score": 1.0
            })

    def search(self, query: str, top_k: int) -> list:
        # Mock search returning all chunks
        return self.chunks[:min(top_k, len(self.chunks))]