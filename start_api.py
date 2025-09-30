#!/usr/bin/env python3
"""
Simple startup script for the Multimodal RAG API
Run this script to start the API server
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if required services are running"""
    checks = {
        "Ollama": "http://localhost:11434",
        "Milvus": "localhost:19530"
    }

    logger.info("Checking dependencies...")

    try:
        import httpx

        for service, url in checks.items():
            try:
                if service == "Ollama":
                    response = httpx.get(f"{url}/api/tags", timeout=5)
                    if response.status_code == 200:
                        logger.info(f"✓ {service} is running")
                    else:
                        logger.warning(f"⚠ {service} may not be fully ready")
                else:
                    logger.info(
                        f"? {service} check skipped (would need specific client)")
            except Exception as e:
                logger.warning(f"⚠ Could not reach {service}: {e}")

    except ImportError:
        logger.warning("httpx not available for dependency checks")


def main():
    """Main function to start the API"""
    logger.info("Starting Multimodal RAG API...")

    # Check dependencies
    check_dependencies()

    # Check if config file exists
    config_file = project_root / "config.yaml"
    if not config_file.exists():
        logger.warning(f"Config file not found: {config_file}")
        logger.info("Please ensure config.yaml exists in the project root")

    # Create uploads directory
    uploads_dir = project_root / "uploads"
    uploads_dir.mkdir(exist_ok=True)
    logger.info(f"Upload directory ready: {uploads_dir}")

    # Start the API server
    try:
        logger.info("Starting FastAPI server on http://localhost:8000")
        logger.info(
            "API documentation will be available at http://localhost:8000/docs")

        # Run using uvicorn
        os.chdir(project_root)
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "api.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ])

    except KeyboardInterrupt:
        logger.info("API server stopped")
    except Exception as e:
        logger.error(f"Error starting API server: {e}")
        logger.info("Try installing dependencies: pip install -e .")
        sys.exit(1)


if __name__ == "__main__":
    main()
