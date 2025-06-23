from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
import asyncio
from embedder import Embedder
from storage import VectorStore
from graph_index import build_linear_graph
from retriever import GraphRAGRetriever
from data_loader import split_text
import os
import uuid
import sqlite3
import json
from typing import List, Dict
from datetime import datetime
from workflow_manager import WorkflowManager
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app with Swagger metadata
app = FastAPI(
    title="Lite RAG Web Service",
    description="A lightweight Retrieval-Augmented Generation (RAG) web service with workflow orchestration.",
    version="1.1.0",
    docs_url="/docs",
    openapi_tags=[
        {"name": "File Operations", "description": "Endpoints for uploading and processing text files."},
        {"name": "Query Operations", "description": "Endpoints for querying the RAG pipeline."},
        {"name": "Task Operations", "description": "Endpoints for checking the status of file processing tasks."},
        {"name": "Document Operations", "description": "Endpoints for managing documents in the knowledge base."},
        {"name": "Workflow Operations", "description": "Endpoints for managing and executing workflows."},
        {"name": "Health", "description": "Endpoint for checking service health."}
    ]
)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_methods=["*"], allow_headers=["*"])
# Initialize components
embedder = Embedder()
store = VectorStore()
retriever = GraphRAGRetriever(store, embedder)
workflow_manager = WorkflowManager()
workflow_manager.initialize_rag(store, embedder, llm_provider=os.getenv("LLM_PROVIDER", "mock"), llm_model=os.getenv("LLM_MODEL", "llama3"))

# Configuration for LLM provider
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mock").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")

# Lazy initialization of RAGPipeline
pipeline = None
def get_pipeline():
    global pipeline
    if pipeline is None:
        from rag_infer import RAGPipeline
        pipeline = RAGPipeline(retriever, llm_provider=LLM_PROVIDER, model=LLM_MODEL)
    return pipeline

# Initialize SQLite for task tracking
def init_task_db():
    conn = sqlite3.connect("tasks.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            file_name TEXT,
            status TEXT,
            error_message TEXT,
            upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT
        )
    """)
    conn.execute("UPDATE tasks SET upload_time = CURRENT_TIMESTAMP WHERE upload_time IS NULL")
    conn.commit()
    conn.close()

init_task_db()

# Pydantic models for request/response
class UploadResponse(BaseModel):
    task_ids: List[str] = Field(..., example=["550e8400-e29b-41d4-a716-446655440000"], description="List of unique task IDs for the uploaded files")
    message: str = Field(..., example="Files uploaded and processing started.", description="Status message")

class QueryRequest(BaseModel):
    query: str = Field(..., example="文中主要讲了什么？", description="The query to be answered by the RAG pipeline")
    top_k: int = Field(3, ge=1, le=10, example=3, description="Number of top chunks to retrieve")

class QueryResponse(BaseModel):
    query: str = Field(..., example="文中主要讲了什么？", description="The input query")
    answer: str = Field(..., example="This is a summarized answer based on the context.", description="The LLM-generated answer")
    context: List[dict] = Field(..., example=[{"chunk_id": "550e8400-e29b-41d4-a716-446655440000_chunk_0", "text": "Sample text", "document_id": "550e8400-e29b-41d4-a716-446655440000", "score": 0.95}], description="Retrieved context chunks")

class TaskStatusResponse(BaseModel):
    task_id: str = Field(..., example="550e8400-e29b-41d4-a716-446655440000", description="Unique task ID")
    file_name: str = Field(..., example="document.txt", description="Name of the uploaded file")
    status: str = Field(..., example="completed", description="Status of the task (pending, completed, failed)")
    error_message: str = Field(None, example="Invalid file format", description="Error message if the task failed")
    upload_time: str = Field(..., example="2025-06-21T12:00:00", description="Upload timestamp")
    metadata: dict = Field(..., example={"tags": []}, description="Document metadata")

class DocumentSummary(BaseModel):
    document_id: str = Field(..., example="550e8400-e29b-41d4-a716-446655440000", description="Unique document ID")
    file_name: str = Field(..., example="document.txt", description="Name of the uploaded file")
    upload_time: str = Field(None, example="2025-06-21T12:00:00", description="Upload timestamp")
    status: str = Field(..., example="completed", description="Processing status")
    metadata: dict = Field(..., example={"tags": []}, description="Document metadata")

class RecallRequest(BaseModel):
    query: str = Field(..., example="文中主要讲了什么？", description="The query to retrieve relevant chunks")
    top_k: int = Field(3, ge=1, le=10, example=3, description="Number of top chunks to retrieve")
    document_ids: List[str] | None = Field(None, example=["550e8400-e29b-41d4-a716-446655440000"], description="Optional list of document IDs to filter results")

class RecallResult(BaseModel):
    chunk_id: str = Field(..., example="550e8400-e29b-41d4-a716-446655440000_chunk_0", description="Unique chunk ID")
    text: str = Field(..., example="This is a sample chunk text.", description="Text content of the chunk")
    document_id: str = Field(..., example="550e8400-e29b-41d4-a716-446655440000", description="Document ID the chunk belongs to")
    score: float = Field(..., example=0.95, description="Similarity score of the chunk to the query")

class ChunkDetail(BaseModel):
    chunk_id: str = Field(..., example="550e8400-e29b-41d4-a716-446655440000_chunk_0", description="Unique chunk ID")
    text: str = Field(..., example="This is a sample chunk text.", description="Text content of the chunk")

class WorkflowUploadRequest(BaseModel):
    name: str = Field(..., example="RAG Query Workflow", description="Name of the workflow")
    workflow_json: Dict = Field(..., example={"nodes": [], "edges": []}, description="Workflow JSON from LangGraph Studio")
    metadata: Dict = Field({}, example={"tags": ["rag"]}, description="Optional metadata for the workflow")

class WorkflowUploadResponse(BaseModel):
    workflow_id: str = Field(..., example="550e8400-e29b-41d4-a716-446655440000", description="Unique workflow ID")
    message: str = Field(..., example="Workflow uploaded successfully.", description="Status message")

class WorkflowSummary(BaseModel):
    workflow_id: str = Field(..., example="550e8400-e29b-41d4-a716-446655440000", description="Unique workflow ID")
    name: str = Field(..., example="RAG Query Workflow", description="Name of the workflow")
    status: str = Field(..., example="active", description="Status of the workflow")
    created_at: str = Field(..., example="2025-06-21T12:00:00", description="Creation timestamp")
    metadata: Dict = Field(..., example={"tags": ["rag"]}, description="Workflow metadata")

class WorkflowExecuteRequest(BaseModel):
    workflow_id: str = Field(..., example="550e8400-e29b-41d4-a716-446655440000", description="Unique workflow ID")
    input_data: Dict = Field(..., example={"query": "What is the main topic?", "config": {"top_k": 3}}, description="Input data for the workflow, including query and optional config")

class WorkflowExecuteResponse(BaseModel):
    workflow_id: str = Field(..., example="550e8400-e29b-41d4-a716-446655440000", description="Unique workflow ID")
    result: Dict = Field(..., example={"answer": "This is the response"}, description="Result of the workflow execution")

async def process_file(file_content: str, file_name: str, task_id: str, metadata: str = "{}"):
    """
    Background task to process a single uploaded file: split, embed, and store.
    """
    conn = sqlite3.connect("tasks.db")
    try:
        conn.execute("INSERT OR REPLACE INTO tasks VALUES (?, ?, ?, ?, ?, ?)", 
                     (task_id, file_name, "pending", None, datetime.now().isoformat(), metadata))
        conn.commit()

        chunks = split_text(file_content)
        embeddings = embedder.embed([c["text"] for c in chunks])
        for i, c in enumerate(chunks):
            store.add_chunk(f"{task_id}_{c['id']}", c["text"], embeddings[i])

        build_linear_graph(chunks, store)

        conn.execute("UPDATE tasks SET status = ? WHERE task_id = ?", ("completed", task_id))
        conn.commit()
    except Exception as e:
        conn.execute("UPDATE tasks SET status = ?, error_message = ? WHERE task_id = ?", 
                     ("failed", str(e), task_id))
        conn.commit()
        print(f"Error processing file {file_name} (task {task_id}): {str(e)}")
    finally:
        conn.close()

@app.post(
    "/upload",
    response_model=UploadResponse,
    tags=["File Operations"],
    summary="Upload multiple text files",
    description="Upload one or more .txt files for vectorization and storage."
)
async def upload_files(files: List[UploadFile] = File(...), background_tasks: BackgroundTasks = None):
    task_ids = []
    for file in files:
        if not file.filename.endswith(".txt"):
            raise HTTPException(status_code=400, detail=f"File {file.filename} is not a .txt file.")
        task_id = str(uuid.uuid4())
        task_ids.append(task_id)
        content = (await file.read()).decode("utf-8")
        background_tasks.add_task(process_file, content, file.filename, task_id)
    return UploadResponse(task_ids=task_ids, message="Files uploaded and processing started.")

@app.post(
    "/query",
    response_model=QueryResponse,
    tags=["Query Operations"],
    summary="Query the RAG pipeline",
    description="Submit a query to retrieve relevant chunks and generate an answer."
)
async def query(request: QueryRequest):
    pipeline = get_pipeline()
    context_chunks = retriever.retrieve(request.query, top_k=request.top_k)
    answer = pipeline.answer(request.query)
    return QueryResponse(query=request.query, answer=answer, context=context_chunks)

@app.get(
    "/tasks/{task_id}",
    response_model=TaskStatusResponse,
    tags=["Task Operations"],
    summary="Check task status",
    description="Retrieve the status of a file processing task."
)
async def get_task_status(task_id: str):
    conn = sqlite3.connect("tasks.db")
    try:
        row = conn.execute("SELECT task_id, file_name, status, error_message, upload_time, metadata FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found.")
        return TaskStatusResponse(
            task_id=row[0],
            file_name=row[1],
            status=row[2],
            error_message=row[3],
            upload_time=row[4] or datetime.now().isoformat(),
            metadata=json.loads(row[5])
        )
    finally:
        conn.close()

@app.get(
    "/documents",
    response_model=List[DocumentSummary],
    tags=["Document Operations"],
    summary="List all documents",
    description="Retrieve a list of all uploaded documents."
)
async def list_documents():
    conn = sqlite3.connect("tasks.db")
    try:
        rows = conn.execute("SELECT task_id, file_name, status, upload_time, metadata FROM tasks").fetchall()
        return [DocumentSummary(
            document_id=row[0],
            file_name=row[1],
            status=row[2],
            upload_time=row[3] or datetime.now().isoformat(),
            metadata=json.loads(row[4])
        ) for row in rows]
    finally:
        conn.close()

@app.post(
    "/recall",
    response_model=List[RecallResult],
    tags=["Query Operations"],
    summary="Retrieve relevant chunks",
    description="Retrieve document chunks relevant to a query."
)
async def recall(request: RecallRequest):
    results = retriever.retrieve(request.query, top_k=request.top_k, document_ids=request.document_ids)
    return [RecallResult(**result) for result in results]

@app.get(
    "/documents/{document_id}/chunks",
    response_model=List[ChunkDetail],
    tags=["Document Operations"],
    summary="List document chunks",
    description="Retrieve all text chunks for a specific document."
)
async def list_document_chunks(document_id: str, limit: int = 100, offset: int = 0):
    conn = sqlite3.connect("tasks.db")
    try:
        row = conn.execute("SELECT task_id FROM tasks WHERE task_id = ?", (document_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found.")
    finally:
        conn.close()

    chunks = store.get_chunks_by_document_id(document_id, limit, offset)
    if not chunks:
        raise HTTPException(status_code=404, detail=f"No chunks found for document {document_id}.")
    return [ChunkDetail(**chunk) for chunk in chunks]

@app.post(
    "/workflows",
    response_model=WorkflowUploadResponse,
    tags=["Workflow Operations"],
    summary="Upload a workflow",
    description="Upload a workflow JSON exported from LangGraph Studio."
)
async def upload_workflow(request: WorkflowUploadRequest):
    workflow_id = str(uuid.uuid4())
    success = workflow_manager.add_workflow(workflow_id, request.name, request.workflow_json, request.metadata)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to upload workflow.")
    return WorkflowUploadResponse(workflow_id=workflow_id, message="Workflow uploaded successfully.")

@app.get(
    "/workflows",
    response_model=List[WorkflowSummary],
    tags=["Workflow Operations"],
    summary="List all workflows",
    description="Retrieve a list of all uploaded workflows."
)
async def list_workflows():
    workflows = workflow_manager.list_workflows()
    return [WorkflowSummary(**workflow) for workflow in workflows]

@app.post(
    "/workflows/execute",
    response_model=WorkflowExecuteResponse,
    tags=["Workflow Operations"],
    summary="Execute a workflow",
    description="Execute a workflow with provided input data."
)
async def execute_workflow(request: WorkflowExecuteRequest):
    if "query" not in request.input_data:
        raise HTTPException(status_code=400, detail="Input data must include a 'query' field.")
    result = workflow_manager.execute_workflow(request.workflow_id, request.input_data)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return WorkflowExecuteResponse(**result)

@app.get(
    "/health",
    tags=["Health"],
    summary="Check service health",
    description="Returns the health status of the web service."
)
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)