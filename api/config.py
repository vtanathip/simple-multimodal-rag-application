"""
Configuration settings for FastAPI application
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings


class APISettings(BaseSettings):
    """API configuration settings"""

    # API Settings
    app_name: str = "Multimodal RAG API"
    version: str = "1.0.0"
    description: str = "REST API for LangGraph-based Multimodal RAG Agent"

    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    log_level: str = "info"

    # Agent Settings
    config_path: str = "config.yaml"
    model_name: str = "llama3.2"
    ollama_base_url: str = "http://localhost:11434"

    # Upload Settings
    upload_dir: str = "uploads"
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_extensions: list = [".pdf"]

    # CORS Settings
    cors_origins: list = ["*"]
    cors_allow_credentials: bool = True
    cors_allow_methods: list = ["*"]
    cors_allow_headers: list = ["*"]

    # Background Tasks
    enable_background_processing: bool = True

    class Config:
        env_file = ".env"
        env_prefix = "RAG_API_"


settings = APISettings()
