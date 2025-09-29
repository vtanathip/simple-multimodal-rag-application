"""
PDF Processing Pipeline using Docling Library
Main class for converting PDF files with multimodal support
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import yaml
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions


@dataclass
class ProcessingResult:
    """Data class to hold processing results"""
    success: bool
    file_path: str
    document: Optional[Any] = None
    markdown_content: Optional[str] = None
    error_message: Optional[str] = None
    processing_time: Optional[float] = None


class DoclingPDFProcessor:
    """
    Main class for processing PDF files using Docling library
    Supports batch processing and configurable pipeline options
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the PDF processor with configuration

        Args:
            config_path: Path to the configuration YAML file
        """
        self.logger = self._setup_logging()
        self.config = self._load_config(config_path)
        self.converter = self._setup_converter()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                self.logger.info(f"Configuration loaded from {config_path}")
                return config
        except FileNotFoundError:
            self.logger.warning(
                f"Config file {config_path} not found, using defaults")
            return self._get_default_config()
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration"""
        return {
            "document": {
                "image_resolution_scale": 2,
                "max_tokens": 512,
                "doc_dir": "documents",
                "supported_file_types": [".pdf"]
            }
        }

    def _setup_converter(self) -> DocumentConverter:
        """Setup document converter with basic configuration"""
        try:
            # Use basic converter - the advanced options can be configured later
            # This avoids typing issues with format_options
            converter = DocumentConverter()
            self.logger.info("Document converter initialized successfully")
            return converter

        except Exception as e:
            self.logger.error(f"Error setting up converter: {e}")
            # Fallback to basic converter
            return DocumentConverter()

    def process_single_pdf(self, file_path: str) -> ProcessingResult:
        """
        Process a single PDF file

        Args:
            file_path: Path to the PDF file

        Returns:
            ProcessingResult with processing details
        """
        import time
        start_time = time.time()

        try:
            # Validate file exists and is PDF
            if not os.path.exists(file_path):
                return ProcessingResult(
                    success=False,
                    file_path=file_path,
                    error_message=f"File not found: {file_path}"
                )

            if not file_path.lower().endswith('.pdf'):
                return ProcessingResult(
                    success=False,
                    file_path=file_path,
                    error_message=f"File is not a PDF: {file_path}"
                )

            self.logger.info(f"Processing PDF: {file_path}")

            # Convert document
            result = self.converter.convert(file_path)
            document = result.document

            # Export to markdown
            markdown_content = document.export_to_markdown()

            processing_time = time.time() - start_time

            self.logger.info(
                f"Successfully processed {file_path} in {processing_time:.2f}s")

            return ProcessingResult(
                success=True,
                file_path=file_path,
                document=document,
                markdown_content=markdown_content,
                processing_time=processing_time
            )

        except Exception as e:
            processing_time = time.time() - start_time
            error_message = f"Error processing {file_path}: {str(e)}"
            self.logger.error(error_message)

            return ProcessingResult(
                success=False,
                file_path=file_path,
                error_message=error_message,
                processing_time=processing_time
            )

    def process_directory(self, directory_path: str) -> List[ProcessingResult]:
        """
        Process all PDF files in a directory

        Args:
            directory_path: Path to directory containing PDF files

        Returns:
            List of ProcessingResult objects
        """
        results = []

        try:
            directory = Path(directory_path)
            if not directory.exists():
                self.logger.error(f"Directory not found: {directory_path}")
                return results

            # Find all PDF files
            pdf_files = list(directory.glob("*.pdf"))

            if not pdf_files:
                self.logger.warning(f"No PDF files found in {directory_path}")
                return results

            self.logger.info(f"Found {len(pdf_files)} PDF files to process")

            # Process each PDF file
            for pdf_file in pdf_files:
                result = self.process_single_pdf(str(pdf_file))
                results.append(result)

            # Log summary
            successful = sum(1 for r in results if r.success)
            self.logger.info(
                f"Processing complete: {successful}/{len(results)} files successful")

        except Exception as e:
            self.logger.error(
                f"Error processing directory {directory_path}: {e}")

        return results

    def save_results(self, results: List[ProcessingResult], output_dir: str = "output") -> None:
        """
        Save processing results to files

        Args:
            results: List of processing results
            output_dir: Directory to save output files
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)

            for result in results:
                if result.success and result.markdown_content:
                    # Create output filename
                    input_name = Path(result.file_path).stem
                    output_file = output_path / f"{input_name}.md"

                    # Save markdown content
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(result.markdown_content)

                    self.logger.info(f"Saved markdown to: {output_file}")

        except Exception as e:
            self.logger.error(f"Error saving results: {e}")

    def get_document_info(self, result: ProcessingResult) -> Dict[str, Any]:
        """
        Extract information from processed document

        Args:
            result: Processing result containing document

        Returns:
            Dictionary with document information
        """
        if not result.success or not result.document:
            return {}

        try:
            doc = result.document

            info = {
                "file_path": result.file_path,
                "processing_time": result.processing_time,
                "page_count": len(doc.pages) if hasattr(doc, 'pages') else 0,
                "has_tables": bool(getattr(doc, 'tables', [])),
                "has_images": bool(getattr(doc, 'pictures', [])),
                "text_length": len(result.markdown_content) if result.markdown_content else 0
            }

            return info

        except Exception as e:
            self.logger.error(f"Error extracting document info: {e}")
            return {}


def main():
    """
    Main function to run the PDF processing pipeline
    """
    # Initialize processor
    processor = DoclingPDFProcessor()

    # Get document directory from config
    doc_dir = processor.config.get("document", {}).get("doc_dir", "documents")

    print(f"Starting PDF processing pipeline...")
    print(f"Processing directory: {doc_dir}")

    # Process all PDFs in the directory
    results = processor.process_directory(doc_dir)

    if not results:
        print("No PDF files found or processed.")
        return

    # Save results
    processor.save_results(results)

    # Display summary
    print("\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)

    successful_results = [r for r in results if r.success]
    failed_results = [r for r in results if not r.success]

    print(f"Total files processed: {len(results)}")
    print(f"Successful: {len(successful_results)}")
    print(f"Failed: {len(failed_results)}")

    if successful_results:
        print("\nSuccessful conversions:")
        for result in successful_results:
            info = processor.get_document_info(result)
            print(f"  • {Path(result.file_path).name}")
            print(f"    - Pages: {info.get('page_count', 'N/A')}")
            print(f"    - Processing time: {result.processing_time:.2f}s")
            print(f"    - Text length: {info.get('text_length', 0)} chars")

    if failed_results:
        print("\nFailed conversions:")
        for result in failed_results:
            print(f"  • {Path(result.file_path).name}: {result.error_message}")

    print("\nMarkdown files saved to 'output' directory")


if __name__ == "__main__":
    main()
