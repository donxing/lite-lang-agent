from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from services.query_service import QueryService
from typing import List

router = APIRouter()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

class ContextChunk(BaseModel):
    chunk_id: str
    text: str
    document_id: str
    score: float

class QueryResponse(BaseModel):
    query: str
    answer: str
    context: List[ContextChunk]

def get_query_service():
    return QueryService()

@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest, service: QueryService = Depends(get_query_service)):
    try:
        return await service.execute_query(request.query, request.top_k)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/recall", response_model=List[ContextChunk])
async def recall_rag(request: QueryRequest, service: QueryService = Depends(get_query_service)):
    try:
        return await service.recall(request.query, request.top_k)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))