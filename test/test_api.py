"""
Tests for the FastAPI Multimodal RAG API
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock

from api.main import app
from src.agent.multimodal_rag_agent import AgentResponse


class TestRAGAPI:
    """Test cases for RAG API endpoints"""

    def setup_method(self):
        """Setup test client"""
        self.client = TestClient(app)

    def test_root_endpoint(self):
        """Test root endpoint"""
        response = self.client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["message"] == "Multimodal RAG API"

    @patch('api.main.agent', None)
    def test_health_check_no_agent(self):
        """Test health check when agent is not initialized"""
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
        assert "components" in data

    @patch('api.main.agent', None)
    def test_query_no_agent(self):
        """Test query endpoint when agent is not initialized"""
        response = self.client.post("/query", json={
            "query": "What is machine learning?",
            "thread_id": "test"
        })
        assert response.status_code == 503
        assert "Agent not initialized" in response.json()["detail"]

    @patch('api.main.agent')
    def test_query_with_agent(self, mock_agent):
        """Test query endpoint with mocked agent"""
        # Mock agent response
        mock_response = AgentResponse(
            answer="Machine learning is a subset of AI.",
            sources=[],
            processing_info={"status": "success"}
        )

        mock_agent.process_query = AsyncMock(return_value=mock_response)

        response = self.client.post("/query", json={
            "query": "What is machine learning?",
            "thread_id": "test"
        })

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert data["answer"] == "Machine learning is a subset of AI."
        assert "sources" in data
        assert "processing_info" in data

    @patch('api.main.agent', None)
    def test_collection_stats_no_agent(self):
        """Test collection stats when agent is not initialized"""
        response = self.client.get("/collection/stats")
        assert response.status_code == 503
        assert "Agent not initialized" in response.json()["detail"]

    @patch('api.main.agent')
    def test_collection_stats_with_agent(self, mock_agent):
        """Test collection stats with mocked agent"""
        mock_stats = {
            "total_documents": 10,
            "collection_name": "test_collection"
        }

        mock_agent.get_collection_stats.return_value = mock_stats

        response = self.client.get("/collection/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["total_documents"] == 10
        assert data["collection_name"] == "test_collection"

    def test_invalid_query_request(self):
        """Test query with invalid request data"""
        response = self.client.post("/query", json={
            "invalid_field": "test"
        })
        assert response.status_code == 422  # Validation error

    @patch('api.main.agent', None)
    def test_upload_no_agent(self):
        """Test document upload when agent is not initialized"""
        response = self.client.post(
            "/upload",
            files={"file": ("test.pdf", b"fake pdf content",
                            "application/pdf")},
            data={"process_immediately": "true"}
        )
        assert response.status_code == 503
        assert "Agent not initialized" in response.json()["detail"]

    @patch('api.main.agent')
    def test_upload_invalid_file_type(self, mock_agent):
        """Test upload with invalid file type"""
        response = self.client.post(
            "/upload",
            files={"file": ("test.txt", b"text content", "text/plain")},
            data={"process_immediately": "true"}
        )
        assert response.status_code == 400
        assert "Only PDF files are supported" in response.json()["detail"]

    @patch('api.main.agent')
    def test_upload_with_agent(self, mock_agent):
        """Test successful document upload with agent"""
        mock_result = {
            "success": True,
            "processing_time": 1.5,
            "error_message": None
        }

        mock_agent.add_document.return_value = mock_result

        response = self.client.post(
            "/upload",
            files={"file": ("test.pdf", b"fake pdf content",
                            "application/pdf")},
            data={"process_immediately": "true"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "test.pdf" in data["file_path"]
        assert "uploads" in data["file_path"]
        assert "successfully" in data["message"]


@pytest.mark.asyncio
class TestAsyncEndpoints:
    """Test async functionality"""

    @patch('api.main.agent')
    async def test_query_async(self, mock_agent):
        """Test async query processing"""
        mock_response = AgentResponse(
            answer="Test answer",
            sources=[],
            processing_info={"status": "success"}
        )

        mock_agent.process_query = AsyncMock(return_value=mock_response)

        # Direct async call to the endpoint function
        from api.main import process_query, QueryRequest

        request = QueryRequest(query="test query", thread_id="test")
        response = await process_query(request)

        assert response.answer == "Test answer"
        assert response.sources == []
        assert response.processing_info == {"status": "success"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
