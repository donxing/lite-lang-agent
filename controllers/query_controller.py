from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import List
from services.query_service import QueryService, QueryResponse, ContextChunk

router = APIRouter()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3
    llm_model: str | None = None
    llm_provider: str | None = None

@router.post("/query")
async def query(request: QueryRequest, query_service: QueryService = Depends()):
    return await query_service.execute_query(request.query, request.top_k, request.llm_model, request.llm_provider)

@router.post("/recall")
async def recall(request: QueryRequest, query_service: QueryService = Depends()):
    return await query_service.recall(request.query, request.top_k)