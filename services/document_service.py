import sqlite3
import uuid
import os
import re
from typing import List, Tuple
from fastapi import UploadFile
from pydantic import BaseModel
import logging
import numpy as np
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter

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
        from knowledge_graph import KnowledgeGraph
        self.vector_store = VectorStore()
        self.knowledge_graph = KnowledgeGraph()

    def _detect_format(self, file_name: str, text: str) -> str:
        """Detect document format based on extension and content."""
        if file_name.lower().endswith('.pdf'):
            return 'pdf'
        elif re.search(r'Speaker \d+:', text, re.MULTILINE):
            return 'transcript'
        elif re.search(r'^#{1,3}\s', text, re.MULTILINE) or re.search(r'^\d+\.\s', text, re.MULTILINE):
            return 'user_guide'
        return 'text'

    def _split_text(self, text: str, format_type: str) -> List[dict]:
        """Split text into chunks based on format."""
        if format_type == 'pdf':
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", "\n", "。", "！", "？", " ", ""]
            )
        elif format_type == 'transcript':
            # Split by speaker turns
            chunks = []
            for turn in re.split(r'(Speaker \d+:)', text):
                if turn.strip() and not turn.startswith('Speaker'):
                    chunks.append({"text": turn.strip()})
                elif turn.startswith('Speaker'):
                    if chunks and not chunks[-1]["text"].startswith('Speaker'):
                        chunks[-1]["text"] = turn + chunks[-1]["text"]
                    else:
                        chunks.append({"text": turn})
            return chunks
        elif format_type == 'user_guide':
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=300,
                chunk_overlap=50,
                separators=["\n#{1,3}\s", "\n\d+\.\s", "\n\n", "\n", "。", " ", ""]
            )
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=300,
                chunk_overlap=50,
                separators=["\n\n", "\n", "。", " ", ""]
            )
        chunks = splitter.split_text(text)
        return [{"text": chunk} for chunk in chunks]

    async def upload_files(self, files: List[UploadFile]) -> List[str]:
        task_ids = []
        for file in files:
            task_id = str(uuid.uuid4())
            file_name = file.filename
            try:
                logger.info(f"Starting upload for file: {file_name}, task_id: {task_id}")
                content = await file.read()
                temp_path = f"temp_{task_id}_{file_name}"
                with open(temp_path, "wb") as f:
                    f.write(content)
                document_id = str(uuid.uuid4())
                logger.info(f"Processing document_id: {document_id}")

                # Handle file based on extension
                if file_name.lower().endswith('.pdf'):
                    text = self._extract_pdf_text(temp_path)
                else:
                    text = content.decode("utf-8", errors="replace")

                # Detect format and split text
                format_type = self._detect_format(file_name, text)
                logger.info(f"Detected format: {format_type} for file: {file_name}")
                chunk_dicts = self._split_text(text, format_type)
                chunks = [d["text"] for d in chunk_dicts]
                chunk_ids = [f"{document_id}_{i}" for i in range(len(chunks))]
                logger.info(f"Generated {len(chunks)} chunks for document {document_id}")

                from embedder import Embedder
                embedder = Embedder()
                embeddings = embedder.embed(chunks)
                logger.info(f"Generated {len(embeddings)} embeddings for document {document_id}")

                # Extract entities and relations
                for chunk_id, chunk in zip(chunk_ids, chunks):
                    entities, relations = self.knowledge_graph.extract_entities_and_relations(chunk, chunk_id, format_type)
                    self._save_entities(document_id, chunk_id, entities)
                    self._save_relations(document_id, chunk_id, relations)

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
                logger.error(f"Upload failed for {file_name}: {e}", exc_info=True)
                raise
        logger.info(f"Upload completed: {len(task_ids)} files processed")
        return task_ids

    def _extract_pdf_text(self, file_path: str) -> str:
        try:
            with pdfplumber.open(file_path) as pdf:
                text = "".join(page.extract_text() or "" for page in pdf.pages)
            if not text.strip():
                logger.warning(f"No text extracted from PDF: {file_path}")
                return ""
            logger.info(f"Extracted {len(text)} characters from PDF: {file_path}")
            return text
        except Exception as e:
            logger.error(f"Failed to extract text from PDF {file_path}: {e}")
            raise

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

    def _save_entities(self, document_id: str, chunk_id: str, entities: List[Tuple[str, str]]):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                for entity_text, entity_type in entities:
                    entity_id = str(uuid.uuid4())
                    cursor.execute(
                        "INSERT INTO entities (entity_id, entity_text, entity_type, chunk_id) VALUES (?, ?, ?, ?)",
                        (entity_id, entity_text, entity_type, chunk_id)
                    )
                conn.commit()
                logger.info(f"Saved {len(entities)} entities for chunk {chunk_id}")
        except sqlite3.Error as e:
            logger.error(f"Failed to save entities for chunk {chunk_id}: {e}")
            raise

    def _save_relations(self, document_id: str, chunk_id: str, relations: List[Tuple[str, str, str]]):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                for source, relation, target in relations:
                    relation_id = str(uuid.uuid4())
                    cursor.execute(
                        "INSERT INTO relations (relation_id, source_entity, relation, target_entity, chunk_id) VALUES (?, ?, ?, ?, ?)",
                        (relation_id, source, relation, target, chunk_id)
                    )
                conn.commit()
                logger.info(f"Saved {len(relations)} relations for chunk {chunk_id}")
        except sqlite3.Error as e:
            logger.error(f"Failed to save relations for chunk {chunk_id}: {e}")
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