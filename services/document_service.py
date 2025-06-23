import sqlite3
import uuid
import os
from typing import List
from fastapi import UploadFile
from pydantic import BaseModel
import logging

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
        if not os.path.exists(self.db_path):
            logger.info(f"Database {self.db_path} not found, creating new database")
            self._initialize_db()
        else:
            logger.info(f"Database {self.db_path} found, using existing database")
        from vector_store import VectorStore
        self.vector_store = VectorStore()

    def _initialize_db(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE documents (
                        document_id TEXT PRIMARY KEY,
                        file_name TEXT,
                        status TEXT
                    )
                """)
                cursor.execute("""
                    CREATE TABLE chunks (
                        chunk_id TEXT PRIMARY KEY,
                        text TEXT,
                        document_id TEXT,
                        FOREIGN KEY (document_id) REFERENCES documents(document_id)
                    )
                """)
                cursor.execute("""
                    CREATE TABLE tasks (
                        task_id TEXT PRIMARY KEY,
                        file_name TEXT,
                        status TEXT,
                        document_id TEXT,
                        error_message TEXT,
                        upload_time TEXT,
                        FOREIGN KEY (document_id) REFERENCES documents(document_id)
                    )
                """)
                conn.commit()
                logger.info("Database initialized successfully: documents, chunks, tasks tables created")
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    async def upload_files(self, files: List[UploadFile]) -> List[str]:
        task_ids = []
        for file in files:
            task_id = str(uuid.uuid4())
            file_name = file.filename
            content = await file.read()
            try:
                temp_path = f"temp_{task_id}.txt"
                with open(temp_path, "wb") as f:
                    f.write(content)
                document_id = str(uuid.uuid4())
                chunks = self._chunk_text(content.decode("utf-8"))
                self.vector_store.add_document(document_id, file_name, chunks)
                self._save_document(document_id, file_name, "completed")
                self._save_chunks(document_id, chunks)
                self._save_task(task_id, file_name, "completed", document_id)
                task_ids.append(task_id)
                os.remove(temp_path)
            except Exception as e:
                self._save_task_status(task_id, file_name, "failed", None, str(e))
                logger.error(f"Upload failed for {file_name}: {e}")
                raise
        return task_ids

    def _chunk_text(self, text: str) -> List[str]:
        return [text[i:i+500] for i in range(0, len(text), 500)]

    def _save_document(self, document_id: str, file_name: str, status: str):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO documents (document_id, file_name, status) VALUES (?, ?, ?)",
                    (document_id, file_name, status)
                )
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to save document {document_id}: {e}")
            raise

    def _save_chunks(self, document_id: str, chunks: List[str]):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                for idx, text in enumerate(chunks):
                    chunk_id = f"{document_id}_{idx}"
                    cursor.execute(
                        "INSERT INTO chunks (chunk_id, text, document_id) VALUES (?, ?, ?)",
                        (chunk_id, text, document_id)
                    )
                conn.commit()
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