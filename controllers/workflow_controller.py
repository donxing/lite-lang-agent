from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Dict, Any, List
from services.workflow_service import WorkflowService

router = APIRouter()

class WorkflowUploadRequest(BaseModel):
    name: str
    workflow_json: Dict[str, Any]
    metadata: Dict[str, Any] = {}

class WorkflowResponse(BaseModel):
    workflow_id: str
    name: str
    status: str

class ExecuteWorkflowRequest(BaseModel):
    workflow_id: str
    input_data: Dict[str, Any]

class ExecuteWorkflowResponse(BaseModel):
    workflow_id: str
    result: Dict[str, Any]

def get_workflow_service():
    return WorkflowService()

@router.post("/workflows", response_model=WorkflowResponse)
async def upload_workflow(request: WorkflowUploadRequest, service: WorkflowService = Depends(get_workflow_service)):
    try:
        return await service.upload_workflow(request.name, request.workflow_json, request.metadata)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/workflows", response_model=List[WorkflowResponse])
async def list_workflows(service: WorkflowService = Depends(get_workflow_service)):
    return await service.list_workflows()

@router.post("/workflows/execute", response_model=ExecuteWorkflowResponse)
async def execute_workflow(request: ExecuteWorkflowRequest, service: WorkflowService = Depends(get_workflow_service)):
    try:
        return await service.execute_workflow(request.workflow_id, request.input_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))