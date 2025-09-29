# Simple Multimodal RAG Application

## DoclingPDFProcessor - PDF Processing Pipeline

A comprehensive Python class for processing PDF files using the Docling library, designed for multimodal RAG applications.

## Installation

The project uses UV for dependency management. Install dependencies:

```bash
uv sync
```

Or with pip:

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from src.index import DoclingPDFProcessor

# Initialize processor
processor = DoclingPDFProcessor()

# Process a single PDF
result = processor.process_single_pdf("path/to/document.pdf")

if result.success:
    print(f"Successfully processed in {result.processing_time:.2f}s")
    print(f"Output length: {len(result.markdown_content)} characters")
    
    # Save markdown output
    with open("output.md", "w") as f:
        f.write(result.markdown_content)
else:
    print(f"Processing failed: {result.error_message}")
```

### Batch Processing

```python
# Process all PDFs in a directory
results = processor.process_directory("documents/")

# Save all results automatically
processor.save_results(results, output_dir="output/")

# Get summary
successful = [r for r in results if r.success]
print(f"Processed {len(successful)}/{len(results)} files successfully")
```

### Running the Main Pipeline

```bash
# Using the configured Python environment
.venv/Scripts/python.exe src/index.py
```

Or run the demo:

```bash
.venv/Scripts/python.exe demo.py
```

## Configuration

The processor uses `config.yaml` for configuration:

```yaml
document:
  image_resolution_scale: 2  # Scale factor for image processing
  max_tokens: 512           # Maximum tokens for chunking
  doc_dir: "documents"      # Input directory for PDFs
  supported_file_types: [".pdf"]
  picture_description:
    prompt_picture_description: "Describe this image in sentences in a single paragraph."
```

## Class Documentation

### `DoclingPDFProcessor`

Main class for processing PDF files using the Docling library.

#### Constructor

```python
DoclingPDFProcessor(config_path: str = "config.yaml")
```

- `config_path`: Path to YAML configuration file

#### Methods

##### `process_single_pdf(file_path: str) -> ProcessingResult`

Process a single PDF file.

**Parameters:**
- `file_path`: Path to the PDF file

**Returns:**
- `ProcessingResult` object with processing details

##### `process_directory(directory_path: str) -> List[ProcessingResult]`

Process all PDF files in a directory.

**Parameters:**
- `directory_path`: Path to directory containing PDFs

**Returns:**
- List of `ProcessingResult` objects

##### `save_results(results: List[ProcessingResult], output_dir: str = "output") -> None`

Save processing results to markdown files.

**Parameters:**
- `results`: List of processing results
- `output_dir`: Output directory for markdown files

##### `get_document_info(result: ProcessingResult) -> Dict[str, Any]`

Extract metadata from processed document.

**Parameters:**
- `result`: Processing result containing document

**Returns:**
- Dictionary with document information

### `ProcessingResult`

Data class containing processing results.

**Attributes:**
- `success: bool` - Whether processing succeeded
- `file_path: str` - Path to processed file
- `document: Optional[Any]` - Docling document object
- `markdown_content: Optional[str]` - Converted markdown content
- `error_message: Optional[str]` - Error message if failed
- `processing_time: Optional[float]` - Processing time in seconds

## Example Output

When processing completes successfully, you'll see output like:

```
Starting PDF processing pipeline...
Processing directory: documents
2025-09-29 09:42:39,569 - INFO - Document converter initialized successfully
2025-09-29 09:42:39,570 - INFO - Processing PDF: documents/sample.pdf
2025-09-29 09:42:41,234 - INFO - Successfully processed documents/sample.pdf in 1.66s

==================================================
PROCESSING SUMMARY
==================================================
Total files processed: 1
Successful: 1
Failed: 0

Successful conversions:
  • sample.pdf
    - Pages: 5
    - Processing time: 1.66s
    - Text length: 2847 chars

Markdown files saved to 'output' directory
```

## Advanced Usage

### Custom Configuration

Create your own configuration file:

```python
processor = DoclingPDFProcessor("my_config.yaml")
```

### Error Handling

```python
results = processor.process_directory("documents/")

for result in results:
    if result.success:
        print(f"✅ {result.file_path}: {result.processing_time:.2f}s")
    else:
        print(f"❌ {result.file_path}: {result.error_message}")
```

### Document Analysis

```python
for result in results:
    if result.success:
        info = processor.get_document_info(result)
        print(f"File: {info['file_path']}")
        print(f"Pages: {info['page_count']}")
        print(f"Has tables: {info['has_tables']}")
        print(f"Has images: {info['has_images']}")
```

## Directory Structure

```
├── src/
│   └── DoclingPDFProcessor.py          # Main DoclingPDFProcessor class
├── test/
│   └── test_processor.py               # PyTest class
├── documents/            # Input PDF files
├── output/              # Generated markdown files
├── config.yaml          # Configuration file
├── main.py             # Demo script
└── README.md           # This file
```

## Dependencies

- `docling>=2.53.0` - PDF processing library
- `docling-core>=2.48.1` - Core docling functionality
- `pyyaml>=6.0.2` - YAML configuration parsing
- `transformers>=4.56.1` - For NLP processing

## Testing

Run the test suite:

```bash
.venv/Scripts/python.exe ./test/test_processor.py
```

## Contributing

1. Ensure all tests pass
2. Add tests for new features
3. Update documentation
4. Follow Python best practices
