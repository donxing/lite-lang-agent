import sqlite3
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_databases():
    """Initialize vectors.db and workflows.db if they don't exist."""
    # Initialize vectors.db
    vectors_db_path = "vectors.db"
    if not os.path.exists(vectors_db_path):
        try:
            with sqlite3.connect(vectors_db_path) as conn:
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
                        embedding BLOB,
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
                logger.info("vectors.db initialized successfully: documents, chunks, tasks tables created")
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize vectors.db: {e}")
            raise
    else:
        logger.info(f"vectors.db found, skipping initialization")

    # Initialize workflows.db
    workflows_db_path = "workflows.db"
    if not os.path.exists(workflows_db_path):
        try:
            with sqlite3.connect(workflows_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE workflows (
                        workflow_id TEXT PRIMARY KEY,
                        name TEXT,
                        status TEXT,
                        workflow_json TEXT,
                        metadata TEXT
                    )
                """)
                conn.commit()
                logger.info("workflows.db initialized successfully: workflows table created")
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize workflows.db: {e}")
            raise
    else:
        logger.info(f"workflows.db found, skipping initialization")