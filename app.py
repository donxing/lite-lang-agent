from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from controllers import document_controller, query_controller, workflow_controller
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(document_controller.router, prefix="/api/v1")
app.include_router(query_controller.router, prefix="/api/v1")
app.include_router(workflow_controller.router, prefix="/api/v1")

# Initialize databases on startup
@app.on_event("startup")
async def startup_event():
    try:
        from db_init import initialize_databases
        initialize_databases()
        logger.info("Application startup: Databases initialized")
    except Exception as e:
        logger.error(f"Failed to initialize databases on startup: {e}")
        raise

@app.get("/health")
async def health_check():
    return {"status": "healthy"}