import src.index as index_mod
import types
import sys
import os
import logging
import tempfile
import pytest
from src.index import RAGMultiModal


class DummyConverter:
    pass


class DummyChunker:
    pass


class DummyDoc:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


class DummyLoader:
    def __init__(self, file_path, converter, chunker, export_type):
        self.file_path = file_path
        self.converter = converter
        self.chunker = chunker
        self.export_type = export_type

    def load(self):
        # Simulate a doc with expected metadata structure
        return [
            DummyDoc(
                page_content="Test content",
                metadata={
                    "source": self.file_path,
                    "dl_meta": {
                        "doc_items": [
                            {"prov": [{"page_no": 1}]}
                        ]
                    }
                }
            )
        ]


# Patch DoclingLoader and ExportType in src.index for testing
index_mod.DoclingLoader = DummyLoader
index_mod.ExportType = type("ExportType", (), {"DOC_CHUNKS": 1})


def test_logger_initialization(tmp_path):
    log_file = tmp_path / "test.log"
    rag = RAGMultiModal(name="test_logger",
                        level=logging.DEBUG, log_file=str(log_file))
    logger = rag.get_logger()
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    # Check log file exists and contains messages
    assert log_file.exists()
    content = log_file.read_text()
    assert "debug message" in content
    assert "info message" in content
    assert "warning message" in content
    assert "error message" in content


def test_set_level_changes_log_level(tmp_path):
    log_file = tmp_path / "test2.log"
    rag = RAGMultiModal(name="test_set_level",
                        level=logging.INFO, log_file=str(log_file))
    logger = rag.get_logger()
    rag.set_level(logging.ERROR)
    for handler in logger.handlers:
        assert handler.level == logging.ERROR


def test_process_file_returns_expected_docs():
    rag = RAGMultiModal(name="test_process_file")
    docs = rag.process_file(
        file_path="dummy.txt",
        converter=DummyConverter(),
        chunker=DummyChunker(),
        namespace="testns"
    )
    assert isinstance(docs, list)
    assert len(docs) == 1
    doc = docs[0]
    assert hasattr(doc, "page_content")
    assert doc.page_content == "Test content"
    assert doc.metadata["source"] == "dummy.txt"
    assert doc.metadata["page_no"] == 1
    assert doc.metadata["namespace"] == "testns"


def test_get_logger_returns_logger():
    rag = RAGMultiModal(name="test_get_logger")
    logger = rag.get_logger()
    assert isinstance(logger, logging.Logger)
    logger.info("Logger works!")
