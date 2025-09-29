#!/usr/bin/env python3
"""
Test suite for the DoclingPDFProcessor with MilvusManager integration using pytest

This test suite covers:
- Processor initialization with database integration
- Database operations during PDF processing
- Text chunking functionality
- Search capabilities
- Error handling for database operations

Run with: pytest test/test_processor_with_database.py -v
"""

import pytest
import tempfile
import os
import yaml
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.DoclingPDFProcessor import DoclingPDFProcessor, ProcessingResult
from src.MilvusManager import VectorDocument, SearchResult


class TestDoclingPDFProcessorWithDatabase:
    """Test class for DoclingPDFProcessor with database integration"""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing"""
        return {
            "document": {
                "image_resolution_scale": 2,
                "max_tokens": 512,
                "doc_dir": "documents",
                "supported_file_types": [".pdf"]
            },
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
    def mock_milvus_manager(self):
        """Create a mock MilvusManager for testing"""
        mock_manager = Mock()
        mock_manager.create_collection.return_value = True
        mock_manager.insert_text_documents.return_value = True
        mock_manager.search.return_value = [
            SearchResult(
                id="test_id",
                text="Test result",
                score=0.95,
                file_path="/test/path.pdf",
                page_number=1
            )
        ]
        mock_manager.get_collection_stats.return_value = {
            "collection_name": "test_collection",
            "exists": True
        }
        mock_manager.close = Mock()
        return mock_manager

    def test_processor_initialization_with_database(self, config_file):
        """Test that the processor can be initialized with database enabled"""
        with patch('src.DoclingPDFProcessor.DocumentConverter'), \
                patch('src.DoclingPDFProcessor.MilvusManager') as mock_milvus_class:

            mock_milvus_class.return_value = Mock()
            mock_milvus_class.return_value.create_collection.return_value = True

            processor = DoclingPDFProcessor(
                config_path=config_file, use_database=True)

            assert processor is not None
            assert processor.use_database is True
            assert processor.milvus_manager is not None

        # Cleanup
        os.unlink(config_file)

    def test_processor_initialization_without_database(self, config_file):
        """Test that the processor can be initialized without database"""
        with patch('src.DoclingPDFProcessor.DocumentConverter'):
            processor = DoclingPDFProcessor(
                config_path=config_file, use_database=False)

            assert processor is not None
            assert processor.use_database is False
            assert processor.milvus_manager is None

        # Cleanup
        os.unlink(config_file)

    def test_processor_database_setup_failure(self, config_file):
        """Test processor handles database setup failure gracefully"""
        with patch('src.DoclingPDFProcessor.DocumentConverter'), \
                patch('src.DoclingPDFProcessor.MilvusManager') as mock_milvus_class:

            mock_milvus_class.side_effect = Exception(
                "Database connection failed")

            processor = DoclingPDFProcessor(
                config_path=config_file, use_database=True)

            assert processor is not None
            assert processor.use_database is True
            assert processor.milvus_manager is None  # Should be None due to error

        # Cleanup
        os.unlink(config_file)

    def test_text_chunking_functionality(self):
        """Test text chunking with different scenarios"""
        with patch('src.DoclingPDFProcessor.DocumentConverter'):
            processor = DoclingPDFProcessor(use_database=False)

            # Test normal text chunking
            text = "This is a test document. " * 100  # Create long text
            chunks = processor._chunk_text(text, max_tokens=50)

            assert len(chunks) > 1
            assert all(isinstance(chunk, str) for chunk in chunks)
            assert all(len(chunk.strip()) > 0 for chunk in chunks)

    def test_text_chunking_short_text(self):
        """Test text chunking with short text"""
        with patch('src.DoclingPDFProcessor.DocumentConverter'):
            processor = DoclingPDFProcessor(use_database=False)

            short_text = "This is a short test."
            chunks = processor._chunk_text(short_text, max_tokens=100)

            assert len(chunks) == 1
            assert chunks[0] == short_text

    def test_text_chunking_empty_text(self):
        """Test text chunking with empty text"""
        with patch('src.DoclingPDFProcessor.DocumentConverter'):
            processor = DoclingPDFProcessor(use_database=False)

            empty_text = ""
            chunks = processor._chunk_text(empty_text)

            assert len(chunks) == 0

    @patch('src.DoclingPDFProcessor.DocumentConverter')
    @patch('src.DoclingPDFProcessor.MilvusManager')
    def test_save_to_database_success(self, mock_milvus_class, mock_converter):
        """Test successful saving to database"""
        mock_milvus_manager = Mock()
        mock_milvus_manager.insert_text_documents.return_value = True
        mock_milvus_class.return_value = mock_milvus_manager
        mock_milvus_class.return_value.create_collection.return_value = True

        processor = DoclingPDFProcessor(use_database=True)

        # Mock document with pages
        mock_document = Mock()
        mock_document.pages = [Mock(), Mock()]  # 2 pages

        result = processor._save_to_database(
            "/test/file.pdf",
            "This is test content for the database.",
            mock_document
        )

        assert result is True
        mock_milvus_manager.insert_text_documents.assert_called_once()

    @patch('src.DoclingPDFProcessor.DocumentConverter')
    @patch('src.DoclingPDFProcessor.MilvusManager')
    def test_save_to_database_failure(self, mock_milvus_class, mock_converter):
        """Test handling of database save failure"""
        mock_milvus_manager = Mock()
        mock_milvus_manager.insert_text_documents.return_value = False
        mock_milvus_class.return_value = mock_milvus_manager
        mock_milvus_class.return_value.create_collection.return_value = True

        processor = DoclingPDFProcessor(use_database=True)

        mock_document = Mock()
        mock_document.pages = [Mock()]

        result = processor._save_to_database(
            "/test/file.pdf",
            "Test content",
            mock_document
        )

        assert result is False

    @patch('src.DoclingPDFProcessor.DocumentConverter')
    @patch('src.DoclingPDFProcessor.MilvusManager')
    def test_save_to_database_no_content(self, mock_milvus_class, mock_converter):
        """Test save to database with no content"""
        mock_milvus_class.return_value.create_collection.return_value = True

        processor = DoclingPDFProcessor(use_database=True)

        result = processor._save_to_database(
            "/test/file.pdf",
            "",  # Empty content
            Mock()
        )

        assert result is False

    @patch('src.DoclingPDFProcessor.DocumentConverter')
    @patch('src.DoclingPDFProcessor.MilvusManager')
    def test_search_documents_success(self, mock_milvus_class, mock_converter):
        """Test successful document search"""
        mock_milvus_manager = Mock()
        mock_search_result = SearchResult(
            id="test_id",
            text="Test result",
            score=0.95,
            file_path="/test/path.pdf",
            page_number=1
        )
        mock_milvus_manager.search.return_value = [mock_search_result]
        mock_milvus_class.return_value = mock_milvus_manager
        mock_milvus_class.return_value.create_collection.return_value = True

        processor = DoclingPDFProcessor(use_database=True)
        results = processor.search_documents("test query", limit=5)

        assert len(results) == 1
        assert results[0]["text"] == "Test result"
        assert results[0]["score"] == 0.95
        assert results[0]["file_path"] == "/test/path.pdf"

    def test_search_documents_no_database(self):
        """Test search when database is not initialized"""
        with patch('src.DoclingPDFProcessor.DocumentConverter'):
            processor = DoclingPDFProcessor(use_database=False)
            results = processor.search_documents("test query")

            assert results == []

    @patch('src.DoclingPDFProcessor.DocumentConverter')
    @patch('src.DoclingPDFProcessor.MilvusManager')
    def test_get_database_stats_success(self, mock_milvus_class, mock_converter):
        """Test getting database statistics"""
        mock_milvus_manager = Mock()
        mock_milvus_manager.get_collection_stats.return_value = {
            "collection_name": "test_collection",
            "exists": True
        }
        mock_milvus_class.return_value = mock_milvus_manager
        mock_milvus_class.return_value.create_collection.return_value = True

        processor = DoclingPDFProcessor(use_database=True)
        stats = processor.get_database_stats()

        assert "collection_name" in stats
        assert stats["exists"] is True

    def test_get_database_stats_no_database(self):
        """Test getting database stats when database not initialized"""
        with patch('src.DoclingPDFProcessor.DocumentConverter'):
            processor = DoclingPDFProcessor(use_database=False)
            stats = processor.get_database_stats()

            assert "error" in stats
            assert stats["error"] == "Database not initialized"

    @patch('src.DoclingPDFProcessor.DocumentConverter')
    @patch('src.DoclingPDFProcessor.MilvusManager')
    def test_close_database(self, mock_milvus_class, mock_converter):
        """Test closing database connection"""
        mock_milvus_manager = Mock()
        mock_milvus_class.return_value = mock_milvus_manager
        mock_milvus_class.return_value.create_collection.return_value = True

        processor = DoclingPDFProcessor(use_database=True)
        processor.close_database()

        mock_milvus_manager.close.assert_called_once()

    def test_close_database_no_database(self):
        """Test closing database when not initialized"""
        with patch('src.DoclingPDFProcessor.DocumentConverter'):
            processor = DoclingPDFProcessor(use_database=False)
            # Should not raise an exception
            processor.close_database()

    @patch('src.DoclingPDFProcessor.DocumentConverter')
    @patch('src.DoclingPDFProcessor.MilvusManager')
    @patch('os.path.exists')
    def test_process_pdf_with_database_integration(self, mock_exists, mock_milvus_class, mock_converter):
        """Test PDF processing with database integration"""
        # Setup mocks
        mock_exists.return_value = True

        mock_document = Mock()
        mock_document.pages = [Mock(), Mock()]
        mock_document.export_to_markdown.return_value = "# Test Document\nThis is test content."

        mock_result = Mock()
        mock_result.document = mock_document

        mock_converter_instance = Mock()
        mock_converter_instance.convert.return_value = mock_result
        mock_converter.return_value = mock_converter_instance

        mock_milvus_manager = Mock()
        mock_milvus_manager.insert_text_documents.return_value = True
        mock_milvus_class.return_value = mock_milvus_manager
        mock_milvus_class.return_value.create_collection.return_value = True

        processor = DoclingPDFProcessor(use_database=True)
        result = processor.process_single_pdf("/test/document.pdf")

        # Verify processing was successful
        assert result.success is True
        assert result.document is not None
        assert result.markdown_content is not None

        # Verify database insertion was attempted
        mock_milvus_manager.insert_text_documents.assert_called_once()

    @patch('src.DoclingPDFProcessor.DocumentConverter')
    @patch('src.DoclingPDFProcessor.MilvusManager')
    @patch('os.path.exists')
    def test_process_pdf_database_save_failure(self, mock_exists, mock_milvus_class, mock_converter):
        """Test PDF processing when database save fails"""
        # Setup mocks
        mock_exists.return_value = True

        mock_document = Mock()
        mock_document.pages = [Mock()]
        mock_document.export_to_markdown.return_value = "Test content"

        mock_result = Mock()
        mock_result.document = mock_document

        mock_converter_instance = Mock()
        mock_converter_instance.convert.return_value = mock_result
        mock_converter.return_value = mock_converter_instance

        mock_milvus_manager = Mock()
        mock_milvus_manager.insert_text_documents.side_effect = Exception(
            "DB Error")
        mock_milvus_class.return_value = mock_milvus_manager
        mock_milvus_class.return_value.create_collection.return_value = True

        processor = DoclingPDFProcessor(use_database=True)
        result = processor.process_single_pdf("/test/document.pdf")

        # Processing should still succeed even if database save fails
        assert result.success is True
        assert result.document is not None

    def test_chunking_with_paragraphs(self):
        """Test text chunking respects paragraph boundaries"""
        with patch('src.DoclingPDFProcessor.DocumentConverter'):
            processor = DoclingPDFProcessor(use_database=False)

            text = "First paragraph with some content.\n\nSecond paragraph with different content.\n\nThird paragraph here."
            # Small chunks to force splitting
            chunks = processor._chunk_text(text, max_tokens=20)

            assert len(chunks) > 1
            # Verify no chunk is empty
            assert all(chunk.strip() for chunk in chunks)

    def test_chunking_very_long_paragraph(self):
        """Test text chunking handles very long paragraphs"""
        with patch('src.DoclingPDFProcessor.DocumentConverter'):
            processor = DoclingPDFProcessor(use_database=False)

            # Create a very long paragraph
            long_paragraph = "This is a very long paragraph. " * 100
            chunks = processor._chunk_text(long_paragraph, max_tokens=50)

            assert len(chunks) > 1
            assert all(chunk.strip() for chunk in chunks)

    @patch('src.DoclingPDFProcessor.DocumentConverter')
    @patch('src.DoclingPDFProcessor.MilvusManager')
    def test_processor_handles_milvus_exception(self, mock_milvus_class, mock_converter):
        """Test processor handles MilvusManager exceptions gracefully"""
        from pymilvus.exceptions import MilvusException

        mock_milvus_manager = Mock()
        mock_milvus_manager.insert_text_documents.side_effect = MilvusException(
            1, "Milvus error")
        mock_milvus_class.return_value = mock_milvus_manager
        mock_milvus_class.return_value.create_collection.return_value = True

        processor = DoclingPDFProcessor(use_database=True)

        mock_document = Mock()
        mock_document.pages = [Mock()]

        result = processor._save_to_database(
            "/test/file.pdf",
            "Test content",
            mock_document
        )

        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
