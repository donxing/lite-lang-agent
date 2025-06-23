import sqlite3
import json
import numpy as np

class VectorStore:
    def __init__(self, db_path="vectors.db"):
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                text TEXT,
                embedding TEXT
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS graph (
                chunk_id TEXT,
                neighbor_id TEXT
            )
        """)

    def add_chunk(self, chunk_id, text, embedding):
        emb_str = json.dumps(embedding.tolist())
        self.conn.execute("INSERT OR REPLACE INTO chunks VALUES (?, ?, ?)", (chunk_id, text, emb_str))
        self.conn.commit()

    def add_edge(self, chunk_id, neighbor_id):
        self.conn.execute("INSERT INTO graph VALUES (?, ?)", (chunk_id, neighbor_id))
        self.conn.commit()

    def get_all_chunks(self):
        return self.conn.execute("SELECT chunk_id, text, embedding FROM chunks").fetchall()

    def get_neighbors(self, chunk_id):
        return [row[0] for row in self.conn.execute("SELECT neighbor_id FROM graph WHERE chunk_id=?", (chunk_id,))]

    def get_chunk_by_id(self, chunk_id):
        row = self.conn.execute("SELECT text FROM chunks WHERE chunk_id=?", (chunk_id,)).fetchone()
        return row[0] if row else None

    def get_chunks_by_document_id(self, document_id, limit=100, offset=0):
        """
        Retrieve chunks for a specific document ID with pagination.
        """
        query = "SELECT chunk_id, text FROM chunks WHERE chunk_id LIKE ? LIMIT ? OFFSET ?"
        rows = self.conn.execute(query, (f"{document_id}_%", limit, offset)).fetchall()
        return [{"chunk_id": row[0], "text": row[1]} for row in rows]