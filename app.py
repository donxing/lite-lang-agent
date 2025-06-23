from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from controllers.document_controller import router as document_router
from controllers.query_controller import router as query_router
from controllers.workflow_controller import router as workflow_router

app = FastAPI(title="Lite RAG API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(document_router, prefix="/api/v1")
app.include_router(query_router, prefix="/api/v1")
app.include_router(workflow_router, prefix="/api/v1")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}