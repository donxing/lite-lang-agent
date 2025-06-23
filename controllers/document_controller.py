from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from typing import List
from services.document_service import DocumentService
from pydantic import BaseModel

router = APIRouter()

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

def get_document_service():
    return DocumentService()

@router.post("/upload")
async def upload_files(files: List[UploadFile] = File(...), service: DocumentService = Depends(get_document_service)):
    try:
        task_ids = await service.upload_files(files)
        return {"message": "Files uploaded successfully", "task_ids": task_ids}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/documents", response_model=List[DocumentResponse])
async def list_documents(service: DocumentService = Depends(get_document_service)):
    return await service.list_documents()

@router.get("/documents/{document_id}/chunks", response_model=List[ChunkResponse])
async def get_document_chunks(document_id: str, service: DocumentService = Depends(get_document_service)):
    return await service.get_document_chunks(document_id)

@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str, service: DocumentService = Depends(get_document_service)):
    return await service.get_task_status(task_id)