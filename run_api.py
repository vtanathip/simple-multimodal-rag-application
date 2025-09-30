"""
Startup script for the Multimodal RAG API
"""

import uvicorn
from api.config import settings


def main():
    """Run the FastAPI application"""
    uvicorn.run(
        "api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level
    )


if __name__ == "__main__":
    main()
