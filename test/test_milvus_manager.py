#!/usr/bin/env python3
"""
Test suite for the MilvusManager class using pytest

This test suite covers:
- MilvusManager initialization and configuration
- Collection management operations
- Document insertion and retrieval
- Search functionality
- Error handling and edge cases

Run with: pytest test/test_milvus_manager.py -v
"""

import pytest
import tempfile
import os
import yaml
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from src.MilvusManager import MilvusManager, VectorDocument, SearchResult


class TestMilvusManager:
    """Test class for MilvusManager"""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing"""
        return {
            "database": {
                "uri": "http://localhost:19530",
                "name": "test_db",
                "collection_name": "test_collection",
                "namespace": "test_namespace"
            }
        }

    @pytest.fixture
    def config_file(self, mock_config):
        """Create a temporary config file for testing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(mock_config, f)
            return f.name

    @pytest.fixture
    def sample_documents(self):
        """Create sample VectorDocument objects for testing"""
        return [
            VectorDocument(
                id="doc1",
                text="This is the first test document about artificial intelligence.",
                vector=[0.1, 0.2, 0.3] * 256,  # 768 dimensions
                metadata={"category": "AI", "source": "test"},
                file_path="/path/to/doc1.pdf",
                page_number=1,
                chunk_index=0
            ),
            VectorDocument(
                id="doc2",
                text="This is the second test document about machine learning.",
                vector=[0.2, 0.3, 0.4] * 256,  # 768 dimensions
                metadata={"category": "ML", "source": "test"},
                file_path="/path/to/doc2.pdf",
                page_number=1,
                chunk_index=1
            )
        ]

    def test_manager_initialization_with_config_file(self, config_file):
        """Test that the manager can be initialized with a config file"""
        with patch('src.MilvusManager.MilvusClient') as mock_client, \
                patch('src.MilvusManager.model.DefaultEmbeddingFunction') as mock_embedding:

            # Setup mocks
            mock_embedding.return_value.dim = 768
            mock_client.return_value = Mock()

            manager = MilvusManager(config_path=config_file)

            assert manager is not None
            assert hasattr(manager, 'config')
            assert hasattr(manager, 'client')
            assert hasattr(manager, 'embedding_fn')
            assert manager.collection_name == "test_collection"

        # Cleanup
        os.unlink(config_file)

    def test_manager_initialization_without_config(self):
        """Test that the manager can be initialized without a config file"""
        with patch('src.MilvusManager.MilvusClient') as mock_client, \
                patch('src.MilvusManager.model.DefaultEmbeddingFunction') as mock_embedding:

            # Setup mocks
            mock_embedding.return_value.dim = 768
            mock_client.return_value = Mock()

            manager = MilvusManager(config_path="nonexistent.yaml")

            assert manager is not None
            assert manager.config is not None
            assert isinstance(manager.config, dict)

    def test_default_config_fallback(self):
        """Test that default config is used when config file doesn't exist"""
        with patch('src.MilvusManager.MilvusClient') as mock_client, \
                patch('src.MilvusManager.model.DefaultEmbeddingFunction') as mock_embedding:

            # Setup mocks
            mock_embedding.return_value.dim = 768
            mock_client.return_value = Mock()

            manager = MilvusManager(config_path="nonexistent_config.yaml")

            # Check default config values
            assert manager.config["database"]["uri"] == "http://localhost:19530"
            assert manager.config["database"]["collection_name"] == "rag_collection"

    @patch('src.MilvusManager.MilvusClient')
    @patch('src.MilvusManager.model.DefaultEmbeddingFunction')
    def test_create_collection_success(self, mock_embedding, mock_client):
        """Test successful collection creation"""
        # Setup mocks
        mock_embedding.return_value.dim = 768
        mock_client_instance = Mock()
        mock_client_instance.has_collection.return_value = False
        mock_client_instance.create_collection.return_value = None
        mock_client.return_value = mock_client_instance

        manager = MilvusManager()
        result = manager.create_collection("test_collection")

        assert result is True
        mock_client_instance.create_collection.assert_called_once()

    @patch('src.MilvusManager.MilvusClient')
    @patch('src.MilvusManager.model.DefaultEmbeddingFunction')
    def test_create_collection_already_exists(self, mock_embedding, mock_client):
        """Test collection creation when collection already exists"""
        # Setup mocks
        mock_embedding.return_value.dim = 768
        mock_client_instance = Mock()
        mock_client_instance.has_collection.return_value = True
        mock_client.return_value = mock_client_instance

        manager = MilvusManager()
        result = manager.create_collection("existing_collection")

        assert result is True
        mock_client_instance.create_collection.assert_not_called()

    @patch('src.MilvusManager.MilvusClient')
    @patch('src.MilvusManager.model.DefaultEmbeddingFunction')
    def test_insert_documents_success(self, mock_embedding, mock_client, sample_documents):
        """Test successful document insertion"""
        # Setup mocks
        mock_embedding.return_value.dim = 768
        mock_client_instance = Mock()
        mock_client_instance.has_collection.return_value = True
        mock_client_instance.insert.return_value = {"insert_count": 2}
        mock_client.return_value = mock_client_instance

        manager = MilvusManager()
        result = manager.insert_documents(sample_documents)

        assert result is True
        mock_client_instance.insert.assert_called_once()

    @patch('src.MilvusManager.MilvusClient')
    @patch('src.MilvusManager.model.DefaultEmbeddingFunction')
    def test_insert_text_documents_success(self, mock_embedding, mock_client):
        """Test successful text document insertion with automatic embedding"""
        # Setup mocks
        mock_embedding_instance = Mock()
        mock_embedding_instance.dim = 768
        mock_embedding_instance.encode_documents.return_value = [
            Mock(tolist=lambda: [0.1, 0.2, 0.3] * 256),
            Mock(tolist=lambda: [0.2, 0.3, 0.4] * 256)
        ]
        mock_embedding.return_value = mock_embedding_instance

        mock_client_instance = Mock()
        mock_client_instance.has_collection.return_value = True
        mock_client_instance.insert.return_value = {"insert_count": 2}
        mock_client.return_value = mock_client_instance

        manager = MilvusManager()
        texts = ["Test document 1", "Test document 2"]
        result = manager.insert_text_documents(texts)

        assert result is True
        mock_embedding_instance.encode_documents.assert_called_once_with(texts)
        mock_client_instance.insert.assert_called_once()

    @patch('src.MilvusManager.MilvusClient')
    @patch('src.MilvusManager.model.DefaultEmbeddingFunction')
    def test_search_success(self, mock_embedding, mock_client):
        """Test successful document search"""
        # Setup mocks
        mock_embedding_instance = Mock()
        mock_embedding_instance.dim = 768
        mock_embedding_instance.encode_queries.return_value = [
            [0.1, 0.2, 0.3] * 256]
        mock_embedding.return_value = mock_embedding_instance

        mock_search_result = Mock()
        mock_search_result.get.side_effect = lambda key, default=None: {
            "id": "doc1",
            "text": "Test result",
            "distance": 0.85,
            "file_path": "/path/to/doc.pdf",
            "page_number": 1
        }.get(key, default)

        mock_client_instance = Mock()
        mock_client_instance.search.return_value = [[mock_search_result]]
        mock_client.return_value = mock_client_instance

        manager = MilvusManager()
        results = manager.search("test query", limit=5)

        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].text == "Test result"
        assert results[0].score == 0.85

    @patch('src.MilvusManager.MilvusClient')
    @patch('src.MilvusManager.model.DefaultEmbeddingFunction')
    def test_delete_collection_success(self, mock_embedding, mock_client):
        """Test successful collection deletion"""
        # Setup mocks
        mock_embedding.return_value.dim = 768
        mock_client_instance = Mock()
        mock_client_instance.has_collection.return_value = True
        mock_client_instance.drop_collection.return_value = None
        mock_client.return_value = mock_client_instance

        manager = MilvusManager()
        result = manager.delete_collection("test_collection")

        assert result is True
        mock_client_instance.drop_collection.assert_called_once_with(
            collection_name="test_collection")

    @patch('src.MilvusManager.MilvusClient')
    @patch('src.MilvusManager.model.DefaultEmbeddingFunction')
    def test_delete_collection_not_exists(self, mock_embedding, mock_client):
        """Test collection deletion when collection doesn't exist"""
        # Setup mocks
        mock_embedding.return_value.dim = 768
        mock_client_instance = Mock()
        mock_client_instance.has_collection.return_value = False
        mock_client.return_value = mock_client_instance

        manager = MilvusManager()
        result = manager.delete_collection("nonexistent_collection")

        assert result is False
        mock_client_instance.drop_collection.assert_not_called()

    @patch('src.MilvusManager.MilvusClient')
    @patch('src.MilvusManager.model.DefaultEmbeddingFunction')
    def test_get_collection_stats(self, mock_embedding, mock_client):
        """Test getting collection statistics"""
        # Setup mocks
        mock_embedding.return_value.dim = 768
        mock_client_instance = Mock()
        mock_client_instance.has_collection.return_value = True
        mock_client_instance.describe_collection.return_value = {
            "collection_name": "test_collection",
            "dimension": 768,
            "metric_type": "COSINE"
        }
        mock_client.return_value = mock_client_instance

        manager = MilvusManager()
        stats = manager.get_collection_stats("test_collection")

        assert "collection_name" in stats
        assert "exists" in stats
        assert "stats" in stats
        assert stats["exists"] is True

    @patch('src.MilvusManager.MilvusClient')
    @patch('src.MilvusManager.model.DefaultEmbeddingFunction')
    def test_get_collection_stats_not_exists(self, mock_embedding, mock_client):
        """Test getting statistics for non-existent collection"""
        # Setup mocks
        mock_embedding.return_value.dim = 768
        mock_client_instance = Mock()
        mock_client_instance.has_collection.return_value = False
        mock_client.return_value = mock_client_instance

        manager = MilvusManager()
        stats = manager.get_collection_stats("nonexistent_collection")

        assert "error" in stats
        assert stats["error"] == "Collection does not exist"

    def test_vector_document_dataclass(self):
        """Test VectorDocument dataclass functionality"""
        doc = VectorDocument(
            id="test_id",
            text="Test text",
            vector=[0.1, 0.2, 0.3],
            metadata={"key": "value"},
            file_path="/test/path.pdf",
            page_number=1,
            chunk_index=0
        )

        assert doc.id == "test_id"
        assert doc.text == "Test text"
        assert doc.vector == [0.1, 0.2, 0.3]
        assert doc.metadata == {"key": "value"}
        assert doc.file_path == "/test/path.pdf"
        assert doc.page_number == 1
        assert doc.chunk_index == 0

    def test_search_result_dataclass(self):
        """Test SearchResult dataclass functionality"""
        result = SearchResult(
            id="result_id",
            text="Result text",
            score=0.95,
            metadata={"source": "test"},
            file_path="/result/path.pdf",
            page_number=2
        )

        assert result.id == "result_id"
        assert result.text == "Result text"
        assert result.score == 0.95
        assert result.metadata == {"source": "test"}
        assert result.file_path == "/result/path.pdf"
        assert result.page_number == 2

    @patch('src.MilvusManager.MilvusClient')
    def test_client_connection_error(self, mock_client):
        """Test error handling when client connection fails"""
        mock_client.side_effect = Exception("Connection failed")

        with pytest.raises(Exception, match="Connection failed"):
            MilvusManager()

    @patch('src.MilvusManager.MilvusClient')
    @patch('src.MilvusManager.model.DefaultEmbeddingFunction')
    def test_embedding_function_error(self, mock_embedding, mock_client):
        """Test error handling when embedding function setup fails"""
        mock_client.return_value = Mock()
        mock_embedding.side_effect = Exception("Embedding setup failed")

        with pytest.raises(Exception, match="Embedding setup failed"):
            MilvusManager()

    @patch('src.MilvusManager.MilvusClient')
    @patch('src.MilvusManager.model.DefaultEmbeddingFunction')
    def test_close_connection(self, mock_embedding, mock_client):
        """Test closing the Milvus client connection"""
        # Setup mocks
        mock_embedding.return_value.dim = 768
        mock_client_instance = Mock()
        mock_client_instance.close = Mock()
        mock_client.return_value = mock_client_instance

        manager = MilvusManager()
        manager.close()

        mock_client_instance.close.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
