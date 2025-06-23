import sqlite3
import uuid
import os
from typing import List
from fastapi import UploadFile
from pydantic import BaseModel
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentResponse(BaseModel):
    document_id: str
    file_name: str
    status: str

class ChunkResponse(BaseModel):
    chunk_id: str
    text: str
    document_id: str

class TaskStatusResponse(BaseModel):
    task_id: str
    file_name: str
    status: str
    upload_time: str
    error_message: str | None = None

class DocumentService:
    def __init__(self):
        self.db_path = "vectors.db"
        from vector_store import VectorStore
        self.vector_store = VectorStore()

    async def upload_files(self, files: List[UploadFile]) -> List[str]:
        task_ids = []
        for file in files:
            task_id = str(uuid.uuid4())
            file_name = file.filename
            try:
                logger.info(f"Starting upload for file: {file_name}, task_id: {task_id}")
                content = await file.read()
                temp_path = f"temp_{task_id}.txt"
                with open(temp_path, "wb") as f:
                    f.write(content)
                document_id = str(uuid.uuid4())
                logger.info(f"Decoding content for document_id: {document_id}")
                from data_loader import split_text
                chunk_dicts = split_text(content.decode("utf-8"), chunk_size=300, overlap=50)
                chunks = [d["text"] for d in chunk_dicts]
                chunk_ids = [f"{document_id}_{i}" for i in range(len(chunks))]
                logger.info(f"Generated {len(chunks)} chunks for document {document_id}")
                from embedder import Embedder
                embedder = Embedder()
                embeddings = embedder.embed(chunks)
                logger.info(f"Generated {len(embeddings)} embeddings for document {document_id}")
                logger.info(f"Indexing chunks for document {document_id}")
                self.vector_store.add_document(document_id, file_name, chunks, chunk_ids)
                self._save_document(document_id, file_name, "completed")
                self._save_chunks(document_id, chunk_dicts, embeddings)
                self._save_task_status(task_id, file_name, "completed", document_id)
                task_ids.append(task_id)
                os.remove(temp_path)
                logger.info(f"Upload completed for file: {file_name}, document_id: {document_id}")
            except Exception as e:
                self._save_task_status(task_id, file_name, "failed", None, str(e))
                logger.error(f"Upload failed for {file_name}: {e}")
                raise
        logger.info(f"Upload completed: {len(task_ids)} files processed")
        return task_ids

    def _save_document(self, document_id: str, file_name: str, status: str):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO documents (document_id, file_name, status) VALUES (?, ?, ?)",
                    (document_id, file_name, status)
                )
                conn.commit()
                logger.info(f"Saved document {document_id}")
        except sqlite3.Error as e:
            logger.error(f"Failed to save document {document_id}: {e}")
            raise

    def _save_chunks(self, document_id: str, chunk_dicts: List[dict], embeddings: np.ndarray):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                for idx, (chunk_dict, embedding) in enumerate(zip(chunk_dicts, embeddings)):
                    chunk_id = f"{document_id}_{idx}"
                    cursor.execute(
                        "INSERT INTO chunks (chunk_id, text, document_id, embedding) VALUES (?, ?, ?, ?)",
                        (chunk_id, chunk_dict["text"], document_id, embedding.tobytes())
                    )
                conn.commit()
                logger.info(f"Saved {len(chunk_dicts)} chunks for document {document_id}")
        except sqlite3.Error as e:
            logger.error(f"Failed to save chunks for document {document_id}: {e}")
            raise

    def _save_task_status(self, task_id: str, file_name: str, status: str, document_id: str | None, error: str | None = None):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO tasks (task_id, file_name, status, document_id, error_message, upload_time)
                    VALUES (?, ?, ?, ?, ?, datetime('now'))
                    """,
                    (task_id, file_name, status, document_id, error)
                )
                conn.commit()
                logger.info(f"Saved task status {task_id}")
        except sqlite3.Error as e:
            logger.error(f"Failed to save task status {task_id}: {e}")
            raise

    async def list_documents(self) -> List[DocumentResponse]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT document_id, file_name, status FROM documents")
                return [DocumentResponse(document_id=row[0], file_name=row[1], status=row[2]) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logger.error(f"Failed to list documents: {e}")
            raise

    async def get_document_chunks(self, document_id: str) -> List[ChunkResponse]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT chunk_id, text, document_id FROM chunks WHERE document_id = ?", (document_id,))
                return [ChunkResponse(chunk_id=row[0], text=row[1], document_id=row[2]) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logger.error(f"Failed to get chunks for document {document_id}: {e}")
            raise

    async def get_task_status(self, task_id: str) -> TaskStatusResponse:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT task_id, file_name, status, upload_time, error_message FROM tasks WHERE task_id = ?",
                    (task_id,)
                )
                row = cursor.fetchone()
                if not row:
                    raise ValueError("Task not found")
                return TaskStatusResponse(
                    task_id=row[0], file_name=row[1], status=row[2], upload_time=row[3], error_message=row[4]
                )
        except sqlite3.Error as e:
            logger.error(f"Failed to get task status {task_id}: {e}")
            raise