"""
Tests for MultimodalRAGAgent
Tests the LangGraph agent functionality including document processing and querying
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List

from src.agent import MultimodalRAGAgent, AgentResponse
from src.MilvusManager import SearchResult


class TestMultimodalRAGAgent:
    """Test suite for MultimodalRAGAgent"""

    @pytest.fixture
    def temp_config_file(self):
        """Create temporary config file for testing"""
        config_content = """
# Test Configuration
model:
  text_generation: "llama3.2"
  embeddings: "text-embedding-3-small"

document:
  image_resolution_scale: 2
  max_tokens: 512
  doc_dir: "test_documents"
  supported_file_types: [".pdf"]

database:
  uri: "http://localhost:19530"
  name: "test_rag_db"
  collection_name: "test_collection"
  namespace: "TestNamespace"

retrieval:
  k: 2
  weights: [0.6, 0.4]
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            temp_path = f.name

        yield temp_path

        # Cleanup
        Path(temp_path).unlink(missing_ok=True)

    @pytest.fixture
    def mock_ollama_response(self):
        """Mock Ollama response"""
        mock_response = Mock()
        mock_response.content = "This is a test response from Ollama"
        return mock_response

    @pytest.fixture
    def mock_search_results(self):
        """Mock search results from Milvus"""
        return [
            SearchResult(
                id="doc1_chunk1",
                text="This is the first chunk of text from document 1.",
                score=0.95,
                metadata={"type": "text"},
                file_path="test_doc1.pdf",
                page_number=1
            ),
            SearchResult(
                id="doc1_chunk2",
                text="This is the second chunk with additional context.",
                score=0.87,
                metadata={"type": "text"},
                file_path="test_doc1.pdf",
                page_number=2
            )
        ]

    @pytest.fixture
    @patch('src.agent.multimodal_rag_agent.ChatOllama')
    @patch('src.agent.multimodal_rag_agent.DoclingPDFProcessor')
    @patch('src.agent.multimodal_rag_agent.MilvusManager')
    def agent(self, mock_milvus, mock_pdf_processor, mock_ollama, temp_config_file):
        """Create agent instance with mocked dependencies"""
        # Mock the classes
        mock_ollama_instance = Mock()
        mock_ollama.return_value = mock_ollama_instance

        mock_pdf_instance = Mock()
        mock_pdf_processor.return_value = mock_pdf_instance

        mock_milvus_instance = Mock()
        mock_milvus.return_value = mock_milvus_instance

        # Create agent
        agent = MultimodalRAGAgent(config_path=temp_config_file)

        return agent

    def test_agent_initialization(self, agent):
        """Test agent initialization"""
        assert agent is not None
        assert hasattr(agent, 'llm')
        assert hasattr(agent, 'pdf_processor')
        assert hasattr(agent, 'milvus_manager')
        assert hasattr(agent, 'graph')
        assert hasattr(agent, 'app')

    @pytest.mark.asyncio
    async def test_analyze_query_document_processing(self, agent, mock_ollama_response):
        """Test query analysis for document processing"""
        # Mock LLM response
        mock_ollama_response.content = "PROCESS_DOCUMENT: /path/to/document.pdf"
        agent.llm.ainvoke = AsyncMock(return_value=mock_ollama_response)

        # Test state
        state = {
            "user_query": "Please process document.pdf",
            "messages": [],
            "search_results": [],
            "context": "",
            "answer": "",
            "file_path": None,
            "processing_status": "initializing"
        }

        # Call method
        result_state = await agent._analyze_query(state)

        # Assertions
        assert result_state["processing_status"] == "needs_processing"
        assert result_state["file_path"] == "/path/to/document.pdf"
        agent.llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_query_search_only(self, agent, mock_ollama_response):
        """Test query analysis for search only"""
        # Mock LLM response
        mock_ollama_response.content = "SEARCH_KNOWLEDGE"
        agent.llm.ainvoke = AsyncMock(return_value=mock_ollama_response)

        # Test state
        state = {
            "user_query": "What is in the documents?",
            "messages": [],
            "search_results": [],
            "context": "",
            "answer": "",
            "file_path": None,
            "processing_status": "initializing"
        }

        # Call method
        result_state = await agent._analyze_query(state)

        # Assertions
        assert result_state["processing_status"] == "search_only"
        assert result_state.get("file_path") is None

    def test_route_query_process_document(self, agent):
        """Test query routing for document processing"""
        state = {"processing_status": "needs_processing"}
        route = agent._route_query(state)
        assert route == "process_doc"

    def test_route_query_search(self, agent):
        """Test query routing for search"""
        state = {"processing_status": "search_only"}
        route = agent._route_query(state)
        assert route == "search"

    @pytest.mark.asyncio
    async def test_process_document_success(self, agent):
        """Test successful document processing"""
        # Mock successful processing result
        mock_result = Mock()
        mock_result.success = True
        mock_result.error_message = None
        agent.pdf_processor.process_single_pdf.return_value = mock_result

        # Test state
        state = {
            "file_path": "/path/to/test.pdf",
            "processing_status": "needs_processing"
        }

        # Call method
        result_state = await agent._process_document(state)

        # Assertions
        assert result_state["processing_status"] == "processed"
        agent.pdf_processor.process_single_pdf.assert_called_once_with(
            "/path/to/test.pdf")

    @pytest.mark.asyncio
    async def test_process_document_failure(self, agent):
        """Test failed document processing"""
        # Mock failed processing result
        mock_result = Mock()
        mock_result.success = False
        mock_result.error_message = "File not found"
        agent.pdf_processor.process_single_pdf.return_value = mock_result

        # Test state
        state = {
            "file_path": "/path/to/nonexistent.pdf",
            "processing_status": "needs_processing"
        }

        # Call method
        result_state = await agent._process_document(state)

        # Assertions
        assert result_state["processing_status"] == "processing_failed"

    @pytest.mark.asyncio
    async def test_search_knowledge(self, agent, mock_search_results):
        """Test knowledge search functionality"""
        # Mock search results
        agent.milvus_manager.search.return_value = mock_search_results

        # Test state
        state = {
            "user_query": "What is the main topic?",
            "search_results": [],
            "context": ""
        }

        # Call method
        result_state = await agent._search_knowledge(state)

        # Assertions
        assert len(result_state["search_results"]) == 2
        assert "test_doc1.pdf" in result_state["context"]
        assert "first chunk of text" in result_state["context"]
        agent.milvus_manager.search.assert_called_once_with(
            query="What is the main topic?",
            limit=5
        )

    @pytest.mark.asyncio
    async def test_generate_response(self, agent, mock_ollama_response):
        """Test response generation"""
        # Mock LLM response
        mock_ollama_response.content = "Based on the context, the main topic is document processing."
        agent.llm.ainvoke = AsyncMock(return_value=mock_ollama_response)

        # Test state
        state = {
            "user_query": "What is the main topic?",
            "context": "Source: test.pdf\nContent: This document is about AI\n---",
            "answer": ""
        }

        # Call method
        result_state = await agent._generate_response(state)

        # Assertions
        assert "document processing" in result_state["answer"]
        agent.llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_query_full_pipeline(self, agent, mock_search_results, mock_ollama_response):
        """Test full query processing pipeline"""
        # Setup mocks for full pipeline
        # 1. Analysis response
        analysis_response = Mock()
        analysis_response.content = "SEARCH_KNOWLEDGE"

        # 2. Final response
        final_response = Mock()
        final_response.content = "Here is the answer based on the documents."

        agent.llm.ainvoke = AsyncMock(
            side_effect=[analysis_response, final_response])
        agent.milvus_manager.search.return_value = mock_search_results

        # Process query
        response = await agent.process_query("What is the main topic?")

        # Assertions
        assert isinstance(response, AgentResponse)
        assert response.answer == "Here is the answer based on the documents."
        assert len(response.sources) == 2
        assert response.processing_info is not None
        assert response.error is None

    def test_process_query_sync(self, agent):
        """Test synchronous query processing wrapper"""
        # Mock the async method
        with patch.object(agent, 'process_query') as mock_async:
            mock_response = AgentResponse(
                answer="Sync test response",
                sources=[],
                processing_info={}
            )
            mock_async.return_value = mock_response

            # Call sync method
            response = agent.process_query_sync("Test query")

            # Assertions
            assert response.answer == "Sync test response"

    def test_add_document_success(self, agent):
        """Test successful document addition"""
        # Mock successful processing
        mock_result = Mock()
        mock_result.success = True
        mock_result.file_path = "/path/to/test.pdf"
        mock_result.error_message = None
        mock_result.processing_time = 5.2
        agent.pdf_processor.process_single_pdf.return_value = mock_result

        # Call method
        result = agent.add_document("/path/to/test.pdf")

        # Assertions
        assert result["success"] is True
        assert result["file_path"] == "/path/to/test.pdf"
        assert result["processing_time"] == 5.2
        assert result["error_message"] is None

    def test_add_document_failure(self, agent):
        """Test failed document addition"""
        # Mock failed processing
        mock_result = Mock()
        mock_result.success = False
        mock_result.file_path = "/path/to/test.pdf"
        mock_result.error_message = "Processing failed"
        mock_result.processing_time = None
        agent.pdf_processor.process_single_pdf.return_value = mock_result

        # Call method
        result = agent.add_document("/path/to/test.pdf")

        # Assertions
        assert result["success"] is False
        assert result["error_message"] == "Processing failed"

    def test_get_collection_stats(self, agent):
        """Test collection statistics retrieval"""
        # Mock stats
        mock_stats = {
            "total_documents": 10,
            "total_chunks": 150,
            "collection_name": "test_collection"
        }
        agent.milvus_manager.get_collection_stats.return_value = mock_stats

        # Call method
        stats = agent.get_collection_stats()

        # Assertions
        assert stats["total_documents"] == 10
        assert stats["total_chunks"] == 150
        assert stats["collection_name"] == "test_collection"

    def test_get_collection_stats_error(self, agent):
        """Test collection statistics error handling"""
        # Mock error
        agent.milvus_manager.get_collection_stats.side_effect = Exception(
            "Connection failed")

        # Call method
        stats = agent.get_collection_stats()

        # Assertions
        assert "error" in stats
        assert "Connection failed" in stats["error"]


class TestAgentResponse:
    """Test AgentResponse data class"""

    def test_agent_response_creation(self):
        """Test AgentResponse creation"""
        sources = [
            SearchResult(
                id="test1",
                text="Test content",
                score=0.9,
                file_path="test.pdf"
            )
        ]

        response = AgentResponse(
            answer="Test answer",
            sources=sources,
            processing_info={"status": "success"}
        )

        assert response.answer == "Test answer"
        assert len(response.sources) == 1
        assert response.processing_info is not None
        assert response.processing_info["status"] == "success"
        assert response.error is None

    def test_agent_response_with_error(self):
        """Test AgentResponse with error"""
        response = AgentResponse(
            answer="Error occurred",
            sources=[],
            error="Connection timeout"
        )

        assert response.answer == "Error occurred"
        assert len(response.sources) == 0
        assert response.error == "Connection timeout"


@pytest.mark.integration
class TestAgentIntegration:
    """Integration tests for the agent (requires running services)"""

    @pytest.mark.skipif(
        True,  # Skip by default, enable for integration testing
        reason="Integration test requires Ollama and Milvus services"
    )
    def test_real_agent_initialization(self):
        """Test agent initialization with real services"""
        try:
            agent = MultimodalRAGAgent()
            assert agent is not None
            print("✅ Agent initialized successfully with real services")
        except Exception as e:
            pytest.skip(f"Services not available: {e}")

    @pytest.mark.skipif(
        True,  # Skip by default, enable for integration testing
        reason="Integration test requires Ollama and Milvus services"
    )
    def test_real_query_processing(self):
        """Test real query processing"""
        try:
            agent = MultimodalRAGAgent()
            response = agent.process_query_sync("Hello, how are you?")
            assert isinstance(response, AgentResponse)
            assert response.answer is not None
            print(f"✅ Real query processed: {response.answer[:100]}...")
        except Exception as e:
            pytest.skip(f"Services not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
