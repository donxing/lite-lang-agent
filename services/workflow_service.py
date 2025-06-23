from pydantic import BaseModel
from typing import Dict, Any, List
import sqlite3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WorkflowResponse(BaseModel):
    workflow_id: str
    name: str
    status: str

class ExecuteWorkflowResponse(BaseModel):
    workflow_id: str
    result: Dict[str, Any]

class WorkflowService:
    def __init__(self):
        self.db_path = "workflows.db"
        from workflow_manager import WorkflowManager
        self.workflow_manager = WorkflowManager()

    async def upload_workflow(self, name: str, workflow_json: Dict[str, Any], metadata: Dict[str, Any]) -> WorkflowResponse:
        try:
            workflow_id = self.workflow_manager.add_workflow(name, workflow_json, metadata)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO workflows (workflow_id, name, status, workflow_json, metadata) VALUES (?, ?, ?, ?, ?)",
                    (workflow_id, name, "active", str(workflow_json), str(metadata))
                )
                conn.commit()
            return WorkflowResponse(workflow_id=workflow_id, name=name, status="active")
        except sqlite3.Error as e:
            logger.error(f"Failed to upload workflow {name}: {e}")
            raise

    async def list_workflows(self) -> List[WorkflowResponse]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT workflow_id, name, status FROM workflows")
                return [WorkflowResponse(workflow_id=row[0], name=row[1], status=row[2]) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logger.error(f"Failed to list workflows: {e}")
            raise

    async def execute_workflow(self, workflow_id: str, input_data: Dict[str, Any]) -> ExecuteWorkflowResponse:
        try:
            result = self.workflow_manager.execute_workflow(workflow_id, input_data)
            return ExecuteWorkflowResponse(workflow_id=workflow_id, result=result)
        except Exception as e:
            logger.error(f"Failed to execute workflow {workflow_id}: {e}")
            raise