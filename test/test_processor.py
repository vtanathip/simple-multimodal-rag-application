#!/usr/bin/env python3
"""
Test suite for the DoclingPDFProcessor class using pytest

This test suite covers:
- Processor initialization and configuration
- Error handling for invalid inputs
- Directory and file processing
- Processing result handling
- Configuration loading and fallbacks
- Database integration features

Run with: pytest test/test_processor.py -v
"""

import pytest
from unittest.mock import Mock, patch
from docling.document_converter import DocumentConverter
from src.DoclingPDFProcessor import DoclingPDFProcessor


class TestDoclingPDFProcessor:
    """Test class for DoclingPDFProcessor"""

    def test_processor_initialization(self):
        """Test that the processor can be initialized"""
        with patch('src.DoclingPDFProcessor.MilvusManager'):
            processor = DoclingPDFProcessor()
            assert processor is not None
            assert hasattr(processor, 'config')
            assert hasattr(processor, 'converter')
            assert hasattr(processor, 'logger')
            assert hasattr(processor, 'use_database')
            assert hasattr(processor, 'milvus_manager')

    def test_config_loading(self):
        """Test configuration loading"""
        with patch('src.DoclingPDFProcessor.MilvusManager'):
            processor = DoclingPDFProcessor()
            assert processor.config is not None
            assert isinstance(processor.config, dict)

            # Test that document config exists
            doc_config = processor.config.get("document", {})
            assert doc_config is not None
            assert isinstance(doc_config, dict)

    def test_converter_setup(self):
        """Test converter setup"""
        with patch('src.DoclingPDFProcessor.MilvusManager'):
            processor = DoclingPDFProcessor()
            assert processor.converter is not None
            assert isinstance(processor.converter, DocumentConverter)

    def test_default_config_fallback(self):
        """Test that default config is used when config file doesn't exist"""
        with patch('src.DoclingPDFProcessor.MilvusManager'):
            processor = DoclingPDFProcessor(
                config_path="nonexistent_config.yaml")
        assert processor.config is not None
        assert "document" in processor.config

        doc_config = processor.config["document"]
        assert "doc_dir" in doc_config
        assert "supported_file_types" in doc_config
        assert doc_config["supported_file_types"] == [".pdf"]

    def test_processor_attributes(self):
        """Test that processor has all required attributes"""
        processor = DoclingPDFProcessor()

        # Check required attributes exist
        assert hasattr(processor, 'config')
        assert hasattr(processor, 'converter')
        assert hasattr(processor, 'logger')

        # Check methods exist
        assert hasattr(processor, 'process_single_pdf')
        assert hasattr(processor, 'process_directory')
        assert hasattr(processor, 'save_results')
        assert hasattr(processor, 'get_document_info')

    def test_config_values(self):
        """Test that configuration values are properly loaded"""
        processor = DoclingPDFProcessor()
        doc_config = processor.config.get("document", {})

        # Test expected config values
        assert "doc_dir" in doc_config
        assert "supported_file_types" in doc_config
        assert "image_resolution_scale" in doc_config

        # Test that supported file types is a list
        assert isinstance(doc_config["supported_file_types"], list)

        # Test that image resolution scale is a number
        assert isinstance(doc_config["image_resolution_scale"], (int, float))

    def test_logger_setup(self):
        """Test that logger is properly configured"""
        processor = DoclingPDFProcessor()
        logger = processor.logger

        assert logger is not None
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'error')
        assert hasattr(logger, 'warning')


@pytest.fixture
def processor():
    """Fixture to provide a DoclingPDFProcessor instance for tests"""
    return DoclingPDFProcessor()


@pytest.fixture
def temp_directory(tmp_path):
    """Fixture to provide a temporary directory for testing"""
    return tmp_path


def test_processor_with_fixture(processor):
    """Test using pytest fixture"""
    assert processor is not None
    assert hasattr(processor, 'config')
    assert hasattr(processor, 'converter')


def test_process_single_pdf_nonexistent_file(processor):
    """Test processing a non-existent PDF file"""
    result = processor.process_single_pdf("nonexistent_file.pdf")

    assert result is not None
    assert result.success is False
    assert result.error_message is not None
    assert "File not found" in result.error_message
    assert result.file_path == "nonexistent_file.pdf"


def test_process_single_pdf_non_pdf_file(processor, temp_directory):
    """Test processing a non-PDF file"""
    # Create a temporary text file
    text_file = temp_directory / "test.txt"
    text_file.write_text("This is not a PDF file")

    result = processor.process_single_pdf(str(text_file))

    assert result is not None
    assert result.success is False
    assert result.error_message is not None
    assert "File is not a PDF" in result.error_message


def test_process_directory_nonexistent(processor):
    """Test processing a non-existent directory"""
    results = processor.process_directory("nonexistent_directory")

    assert results == []


def test_process_directory_empty(processor, temp_directory):
    """Test processing an empty directory"""
    results = processor.process_directory(str(temp_directory))

    assert results == []


def test_get_document_info_unsuccessful_result(processor):
    """Test getting document info from an unsuccessful result"""
    from src.DoclingPDFProcessor import ProcessingResult

    failed_result = ProcessingResult(
        success=False,
        file_path="test.pdf",
        error_message="Test error"
    )

    info = processor.get_document_info(failed_result)
    assert info == {}


def test_save_results_empty_list(processor, temp_directory):
    """Test saving empty results list"""
    # This should not raise an exception
    processor.save_results([], str(temp_directory))

    # Check that no files were created
    files = list(temp_directory.glob("*.md"))
    assert len(files) == 0


class TestProcessingResult:
    """Test the ProcessingResult dataclass"""

    def test_processing_result_creation(self):
        """Test creating a ProcessingResult instance"""
        from src.DoclingPDFProcessor import ProcessingResult

        result = ProcessingResult(
            success=True,
            file_path="test.pdf",
            markdown_content="# Test content",
            processing_time=1.5
        )

        assert result.success is True
        assert result.file_path == "test.pdf"
        assert result.markdown_content == "# Test content"
        assert result.processing_time == 1.5
        assert result.error_message is None
        assert result.document is None

    def test_processing_result_failed(self):
        """Test creating a failed ProcessingResult"""
        from src.DoclingPDFProcessor import ProcessingResult

        result = ProcessingResult(
            success=False,
            file_path="test.pdf",
            error_message="Processing failed"
        )

        assert result.success is False
        assert result.file_path == "test.pdf"
        assert result.error_message == "Processing failed"
        assert result.markdown_content is None
        assert result.document is None
        assert result.processing_time is None

    def test_processor_initialization_with_database_enabled(self):
        """Test processor initialization with database enabled"""
        with patch('src.DoclingPDFProcessor.MilvusManager') as mock_milvus:
            mock_milvus.return_value.create_collection.return_value = True

            processor = DoclingPDFProcessor(use_database=True)

            assert processor.use_database is True
            assert processor.milvus_manager is not None
            mock_milvus.assert_called_once()

    def test_processor_initialization_with_database_disabled(self):
        """Test processor initialization with database disabled"""
        processor = DoclingPDFProcessor(use_database=False)

        assert processor.use_database is False
        assert processor.milvus_manager is None

    def test_database_methods_without_database(self):
        """Test database methods when database is not initialized"""
        processor = DoclingPDFProcessor(use_database=False)

        # Test search without database
        results = processor.search_documents("test query")
        assert results == []

        # Test stats without database
        stats = processor.get_database_stats()
        assert "error" in stats

        # Test close without database (should not raise exception)
        processor.close_database()

    @patch('src.DoclingPDFProcessor.MilvusManager')
    def test_search_documents_with_database(self, mock_milvus_class):
        """Test search functionality with database"""
        from src.MilvusManager import SearchResult

        mock_manager = Mock()
        mock_manager.create_collection.return_value = True
        mock_manager.search.return_value = [
            SearchResult(
                id="test_id",
                text="Test result",
                score=0.95,
                file_path="/test/path.pdf",
                page_number=1
            )
        ]
        mock_milvus_class.return_value = mock_manager

        processor = DoclingPDFProcessor(use_database=True)
        results = processor.search_documents("test query", limit=5)

        assert len(results) == 1
        assert results[0]["text"] == "Test result"
        assert results[0]["score"] == 0.95

    @patch('src.DoclingPDFProcessor.MilvusManager')
    def test_get_database_stats_with_database(self, mock_milvus_class):
        """Test getting database stats with database initialized"""
        mock_manager = Mock()
        mock_manager.create_collection.return_value = True
        mock_manager.get_collection_stats.return_value = {
            "collection_name": "test_collection",
            "exists": True
        }
        mock_milvus_class.return_value = mock_manager

        processor = DoclingPDFProcessor(use_database=True)
        stats = processor.get_database_stats()

        assert "collection_name" in stats
        assert stats["exists"] is True
