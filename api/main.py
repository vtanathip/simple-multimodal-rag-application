"""
FastAPI application for Multimodal RAG Agent
Provides REST API endpoints to interact with the LangGraph-based multimodal RAG system
"""

import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, List, Any, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from src.agent.multimodal_rag_agent import MultimodalRAGAgent, AgentResponse


# Pydantic models for request/response
class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    query: str = Field(..., description="User query to process")
    thread_id: str = Field(
        default="default", description="Thread ID for conversation tracking")


class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    answer: str = Field(..., description="Generated answer")
    sources: List[Dict[str, Any]] = Field(
        default=[], description="Source documents used")
    processing_info: Optional[Dict[str, Any]] = Field(
        None, description="Processing metadata")
    error: Optional[str] = Field(None, description="Error message if any")


class DocumentUploadResponse(BaseModel):
    """Response model for document upload"""
    success: bool = Field(..., description="Whether upload was successful")
    file_path: str = Field(..., description="Path to uploaded file")
    message: str = Field(..., description="Status message")
    processing_time: Optional[float] = Field(
        None, description="Processing time in seconds")
    error_message: Optional[str] = Field(
        None, description="Error message if failed")


class CollectionStatsResponse(BaseModel):
    """Response model for collection statistics"""
    total_documents: int = Field(..., description="Total number of documents")
    collection_name: str = Field(..., description="Name of the collection")
    additional_stats: Dict[str, Any] = Field(
        default={}, description="Additional statistics")


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    components: Dict[str, str] = Field(..., description="Component status")


# Global agent instance
agent: Optional[MultimodalRAGAgent] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global agent

    # Startup
    try:
        logging.info("Initializing MultimodalRAGAgent...")
        agent = MultimodalRAGAgent(
            config_path="config.yaml",
            model_name="llama3.2",
            ollama_base_url="http://localhost:11434"
        )
        logging.info("MultimodalRAGAgent initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize agent: {e}")
        raise

    yield

    # Shutdown
    logging.info("Shutting down application...")


# Create FastAPI app
app = FastAPI(
    title="Multimodal RAG API",
    description="REST API for LangGraph-based Multimodal RAG Agent",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Multimodal RAG API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global agent

    components = {
        "agent": "healthy" if agent else "not_initialized",
        "api": "healthy"
    }

    # Check agent components if available
    if agent:
        try:
            # Test basic functionality
            stats = agent.get_collection_stats()
            components["milvus"] = "healthy" if "error" not in stats else "error"
            # Assume healthy if agent is initialized
            components["llm"] = "healthy"
        except Exception as e:
            components["milvus"] = f"error: {str(e)}"
            components["llm"] = "unknown"

    overall_status = "healthy" if all(
        status == "healthy" for status in components.values()
    ) else "degraded"

    return HealthResponse(
        status=overall_status,
        version="1.0.0",
        components=components
    )


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a user query through the multimodal RAG agent

    Args:
        request: Query request containing the user's question and optional thread ID

    Returns:
        QueryResponse with generated answer and source information
    """
    global agent

    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        logger.info(f"Processing query: {request.query[:100]}...")

        # Process query through agent
        response = await agent.process_query(
            query=request.query,
            thread_id=request.thread_id
        )

        # Convert SearchResult objects to dictionaries
        sources = []
        for source in response.sources:
            sources.append({
                "text": source.text,
                "file_path": source.file_path,
                "distance": source.distance,
                "metadata": source.metadata
            })

        return QueryResponse(
            answer=response.answer,
            sources=sources,
            processing_info=response.processing_info,
            error=response.error
        )

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...,
                            description="PDF document to upload and process"),
    process_immediately: bool = Form(
        default=True, description="Whether to process the document immediately")
):
    """
    Upload and optionally process a document

    Args:
        file: PDF file to upload
        process_immediately: Whether to process the document immediately or just save it
        background_tasks: FastAPI background tasks for async processing

    Returns:
        DocumentUploadResponse with upload and processing status
    """
    global agent

    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    # Validate file type
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400, detail="Only PDF files are supported")

    try:
        # Create uploads directory if it doesn't exist
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)

        # Save uploaded file
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        logger.info(f"File uploaded: {file_path}")

        if process_immediately:
            # Process document immediately
            result = agent.add_document(str(file_path))

            return DocumentUploadResponse(
                success=result["success"],
                file_path=str(file_path),
                message="Document uploaded and processed successfully" if result[
                    "success"] else "Document uploaded but processing failed",
                processing_time=result.get("processing_time"),
                error_message=result.get("error_message")
            )
        else:
            # Add processing as background task
            background_tasks.add_task(
                process_document_background, str(file_path))

            return DocumentUploadResponse(
                success=True,
                file_path=str(file_path),
                message="Document uploaded successfully. Processing started in background.",
                processing_time=None,
                error_message=None
            )

    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error uploading document: {str(e)}")


async def process_document_background(file_path: str):
    """Background task to process uploaded documents"""
    global agent

    if agent:
        try:
            logger.info(f"Processing document in background: {file_path}")
            result = agent.add_document(file_path)
            logger.info(
                f"Background processing completed for {file_path}: {result['success']}")
        except Exception as e:
            logger.error(f"Error in background document processing: {e}")


@app.post("/process-document")
async def process_existing_document(file_path: str = Form(..., description="Path to existing document to process")):
    """
    Process an existing document by file path

    Args:
        file_path: Path to the document to process

    Returns:
        DocumentUploadResponse with processing status
    """
    global agent

    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    # Validate file exists
    if not Path(file_path).exists():
        raise HTTPException(
            status_code=404, detail=f"File not found: {file_path}")

    try:
        logger.info(f"Processing existing document: {file_path}")
        result = agent.add_document(file_path)

        return DocumentUploadResponse(
            success=result["success"],
            file_path=file_path,
            message="Document processed successfully" if result[
                "success"] else "Document processing failed",
            processing_time=result.get("processing_time"),
            error_message=result.get("error_message")
        )

    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error processing document: {str(e)}")


@app.get("/collection/stats", response_model=Dict[str, Any])
async def get_collection_stats():
    """
    Get statistics about the vector collection

    Returns:
        Collection statistics including document count and other metadata
    """
    global agent

    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        stats = agent.get_collection_stats()

        if "error" in stats:
            raise HTTPException(status_code=500, detail=stats["error"])

        return stats

    except Exception as e:
        logger.error(f"Error getting collection stats: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error getting collection stats: {str(e)}")


@app.delete("/collection/clear")
async def clear_collection():
    """
    Clear all documents from the vector collection
    WARNING: This will delete all stored documents and embeddings

    Returns:
        Status of the clear operation
    """
    global agent

    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        # This would need to be implemented in MilvusManager
        # For now, return a message indicating the operation would need to be implemented
        return {
            "message": "Clear collection operation would need to be implemented in MilvusManager",
            "status": "not_implemented"
        }

    except Exception as e:
        logger.error(f"Error clearing collection: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error clearing collection: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
