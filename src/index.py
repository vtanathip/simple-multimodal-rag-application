import os
import logging
from typing import Optional
import logging
from typing import List, Optional, Union
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from langchain_core.documents import Document
from langchain_docling.loader import DoclingLoader
from langchain_docling.loader import ExportType


class RAGMultiModal:
    """
    A class to configure and manage a logger instance.

    This class handles the setup of logging, including setting the level,
    configuring handlers (console and file), and formatting messages.
    """

    def __init__(self, name: str, level: int = logging.INFO, log_file: Optional[str] = None):
        """
        Initializes and configures the logger.

        Args:
            name (str): The name for the logger (e.g., __name__).
            level (int): The initial logging level (e.g., logging.DEBUG).
            log_file (Optional[str]): The path to the log file. If None, logs only to console.
        """
        self.name = name
        self.level = level
        self.log_file = log_file
        self.logger = logging.getLogger(name)

        self._configure_logger()
        self.logger.info(
            f"Logger '{self.name}' initialized with level: {logging.getLevelName(self.level)}")

    def _configure_logger(self):
        """
        Private method to set up or re-configure logger handlers and formatters.
        """
        # 1. Create a formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # 2. Configure the logger level
        self.logger.setLevel(self.level)

        # 3. Remove existing handlers to prevent duplicate logs
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # 4. Create and add console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # 5. Create and add file handler if a path is provided
        if self.log_file:
            try:
                # Create directory for log file if it doesn't exist
                log_dir = os.path.dirname(self.log_file)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir)

                # Create file handler
                file_handler = logging.FileHandler(self.log_file)
                file_handler.setLevel(self.level)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
            except Exception as e:
                self.logger.error(
                    f"Failed to set up file logging to {self.log_file}: {e}", exc_info=True)

    def set_level(self, level: int):
        """
        Changes the logging level for all handlers associated with this logger.

        Args:
            level (int): The new logging level.
        """
        self.level = level
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)
        self.logger.info(
            f"Log level for '{self.name}' changed to: {logging.getLevelName(self.level)}")

    def get_logger(self) -> logging.Logger:
        """
        Returns the configured logger instance.

        Returns:
            logging.Logger: The configured logger object.
        """
        return self.logger

    def process_file(self, file_path, converter, chunker, namespace):
        """
        Process a single file and prepare documents for indexing.

        Args:
            file_path: Path to the file to process
            converter: Document converter to use
            chunker: Document chunker to use
            namespace: Namespace to use for the documents

        Returns:
            List[Document]: List of processed documents
        """
        file_path = Path(file_path) if isinstance(
            file_path, str) else file_path

        # Create document loader
        loader = DoclingLoader(
            file_path=str(file_path),
            converter=converter,
            chunker=chunker,
            export_type=ExportType.DOC_CHUNKS
        )

        # Load and process documents
        docs = loader.load()

        # Prepare documents for indexing
        processed_docs = []
        for doc in docs:
            metadata = doc.metadata
            _metadata = dict()
            _metadata["source"] = str(metadata["source"])
            _metadata["page_no"] = metadata["dl_meta"]['doc_items'][0]['prov'][0]['page_no']
            _metadata["namespace"] = namespace

            processed_doc = Document(
                page_content=doc.page_content,
                metadata=_metadata
            )
            processed_docs.append(processed_doc)

        return processed_docs


# --- Example of how to use the class ---
if __name__ == "__main__":
    # 1. Create an instance of the logger for the main application
    # This will log DEBUG messages and above to both the console and a file.
    app_logger_setup = RAGMultiModal(
        name='my_awesome_app',
        level=logging.DEBUG,
        log_file='logs/application.log'
    )
    log = app_logger_setup.get_logger()

    log.debug("Starting application setup...")
    log.info("Application is running.")
    log.warning("Configuration file is missing a setting, using default.")

    # 2. Change the log level at runtime to be less verbose
    print("\n--- Changing log level to WARNING ---\n")
    app_logger_setup.set_level(logging.WARNING)

    log.debug("This debug message will NOT be shown.")
    log.info("This info message will NOT be shown.")
    log.warning("This is a new warning that WILL be shown.")
    log.error("An error occurred.")
