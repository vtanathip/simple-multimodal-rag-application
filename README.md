# Simple Multimodal RAG Application

A comprehensive multimodal RAG (Retrieval-Augmented Generation) application that processes PDF documents, extracts text and images, stores them in a vector database, and enables intelligent document search and retrieval.

## 🚀 Features

### 📄 PDF Processing with Docling

- **Multi-format support**: PDF processing with advanced text and image extraction
- **Batch processing**: Process multiple PDFs in directories
- **Configurable pipeline**: Customizable processing options via YAML configuration
- **Error handling**: Robust error handling with detailed logging
- **Metadata extraction**: Extract document information including page counts, tables, and images

### 🔍 Vector Database Integration

- **Milvus integration**: Store and retrieve documents using vector similarity
- **Hybrid search**: Support for dense and sparse vector search
- **Collection management**: Automated collection creation and management
- **Document chunking**: Intelligent document segmentation for better retrieval
- **Metadata storage**: Store document metadata alongside vectors

### 🧪 Interactive Notebooks

- **Docling demonstrations**: Multiple Jupyter notebooks showcasing different features
- **OCR capabilities**: Image processing and text extraction examples
- **Hybrid chunking**: Advanced document segmentation techniques
- **Picture annotation**: Automatic image description generation
- **Serialization**: Document export and import examples

### ⚙️ Configuration Management

- **YAML-based config**: Easy configuration through YAML files
- **Environment flexibility**: Support for different deployment environments
- **Model configuration**: Configurable text generation and embedding models
- **Database settings**: Customizable vector database connections

## 📦 Installation

The project uses UV for dependency management. Install dependencies:

```bash
uv sync
```

Or with pip:

```bash
pip install docling>=2.53.0 docling-core>=2.48.1 pymilvus[milvus-lite,model]>=2.6.2 pyyaml>=6.0.2 transformers>=4.56.1 pytest>=8.4.2
```

## 🏃 Quick Start

### 1. Setup Milvus Database (Optional)

Start Milvus database for vector storage:

```bash
# Download and start Milvus
Invoke-WebRequest https://raw.githubusercontent.com/milvus-io/milvus/refs/heads/master/scripts/standalone_embed.bat -OutFile standalone.bat
standalone.bat start
```

Access Milvus Web UI: <http://localhost:9091/webui/>

### 2. Basic PDF Processing

```python
from src.DoclingPDFProcessor import DoclingPDFProcessor

# Initialize processor (with database integration)
processor = DoclingPDFProcessor(use_database=True)

# Process a single PDF
result = processor.process_single_pdf("documents/sample.pdf")

if result.success:
    print(f"✅ Successfully processed in {result.processing_time:.2f}s")
    print(f"📄 Output length: {len(result.markdown_content)} characters")
    
    # Access extracted document info
    info = processor.get_document_info(result)
    print(f"📊 Pages: {info['page_count']}")
    print(f"🖼️ Has images: {info['has_images']}")
    print(f"📋 Has tables: {info['has_tables']}")
else:
    print(f"❌ Processing failed: {result.error_message}")
```

### 3. Batch Processing with Vector Storage

```python
# Process all PDFs in a directory and store in vector database
results = processor.process_directory("documents/")

# Save markdown outputs
processor.save_results(results, output_dir="output/")

# Get processing summary
successful = [r for r in results if r.success]
failed = [r for r in results if not r.success]

print(f"✅ Successfully processed: {len(successful)} files")
print(f"❌ Failed to process: {len(failed)} files")

# If database is enabled, documents are automatically stored as vectors
if processor.use_database and processor.milvus_manager:
    print("🔍 Documents stored in vector database for similarity search")
```

### 4. Vector Search and Retrieval

```python
from src.MilvusManager import MilvusManager

# Initialize database manager
db_manager = MilvusManager()

# Search for similar documents
query = "artificial intelligence research"
search_results = db_manager.search_documents(query, k=3)

for result in search_results:
    print(f"📄 {result.file_path} (Score: {result.score:.3f})")
    print(f"📝 {result.text[:200]}...")
    print(f"📍 Page: {result.page_number}\n")
```

### 5. Running the Main Application

```bash
# Run the complete pipeline
python sample.py

# Run the milvus integration
python milvus_integration.py
```

The application will:

- 📂 Scan the `documents/` directory for PDFs
- 🔄 Process each PDF with advanced text and image extraction
- 💾 Save markdown outputs to `output/` directory
- 🔍 Store document vectors in Milvus database (if enabled)
- 📊 Display processing summary

## ⚙️ Configuration

### Complete Configuration Example

The application uses `config.yaml` for comprehensive configuration:

```yaml
# Model Configuration
model:
  text_generation: "llama3.2"  # Ollama for text generation
  embeddings: "text-embedding-3-small"  # OpenAI model for embeddings
  tokenizer: "sentence-transformers/all-MiniLM-L6-v2"  # HuggingFace tokenizer

# Document Processing Configuration
document:
  image_resolution_scale: 2  # Scale factor for image processing
  max_tokens: 512           # Maximum tokens for chunking
  doc_dir: "documents"      # Input directory for PDFs
  supported_file_types: [".pdf"]
  picture_description:
    prompt_picture_description: "Describe this image in sentences in a single paragraph."

# Vector Database Configuration
database:
  uri: "http://localhost:19530"  # Milvus database URI
  name: "rag_multimodal"         # Database name
  collection_name: "rag_collection"  # Collection name
  namespace: "CaseDoneDemo"      # Namespace for data partitioning

# Retrieval Configuration
retrieval:
  k: 2  # Number of documents to retrieve
  weights: [0.6, 0.4]  # Weights for hybrid search (dense, sparse)
```

### Configuration Options

#### Document Processing

- **`image_resolution_scale`**: Controls image quality during processing
- **`max_tokens`**: Maximum tokens per document chunk for better retrieval
- **`doc_dir`**: Source directory for PDF files
- **`supported_file_types`**: List of supported file extensions
- **`picture_description`**: AI prompt for automatic image description

#### Database Settings

- **`uri`**: Milvus database connection string
- **`collection_name`**: Vector collection name for document storage
- **`namespace`**: Data partitioning namespace

#### Model Configuration

- **`text_generation`**: Model for text generation (supports Ollama models)
- **`embeddings`**: Embedding model for vector creation
- **`tokenizer`**: Tokenization model for text processing

## 💡 Feature Examples

### Example 1: Basic PDF Processing

```python
from src.DoclingPDFProcessor import DoclingPDFProcessor

# Initialize without database
processor = DoclingPDFProcessor(use_database=False)

# Process a research paper
result = processor.process_single_pdf("documents/research_paper.pdf")

if result.success:
    # Extract metadata
    info = processor.get_document_info(result)
    print(f"Document: {info['file_path']}")
    print(f"Pages: {info['page_count']}")
    print(f"Tables detected: {info['table_count']}")
    print(f"Images detected: {info['image_count']}")
    
    # Save as markdown
    with open("output/research_paper.md", "w", encoding="utf-8") as f:
        f.write(result.markdown_content)
```

### Example 2: Multimodal RAG with Vector Search

```python
from src.DoclingPDFProcessor import DoclingPDFProcessor
from src.MilvusManager import MilvusManager

# Process documents with database integration
processor = DoclingPDFProcessor(use_database=True)
db_manager = processor.milvus_manager

# Process multiple documents
results = processor.process_directory("documents/")

# Search for specific information
query = "machine learning algorithms"
search_results = db_manager.search_documents(query, k=5)

for result in search_results:
    print(f"📄 File: {result.file_path}")
    print(f"📍 Page: {result.page_number}")
    print(f"🎯 Relevance Score: {result.score:.3f}")
    print(f"📝 Content Preview: {result.text[:150]}...\n")
```

### Example 3: Custom Configuration

```python
# Create custom configuration
import yaml

custom_config = {
    "document": {
        "image_resolution_scale": 3,  # Higher quality images
        "max_tokens": 1024,           # Larger chunks
        "doc_dir": "research_papers",
        "picture_description": {
            "prompt_picture_description": "Analyze this scientific figure and describe its key findings."
        }
    },
    "database": {
        "collection_name": "research_collection",
        "namespace": "academic_papers"
    }
}

# Save custom config
with open("custom_config.yaml", "w") as f:
    yaml.dump(custom_config, f)

# Use custom configuration
processor = DoclingPDFProcessor("custom_config.yaml")
```

### Example 4: Batch Processing with Error Handling

```python
import os
from pathlib import Path

processor = DoclingPDFProcessor()

# Get all PDF files recursively
pdf_files = []
for root, dirs, files in os.walk("large_document_collection"):
    for file in files:
        if file.lower().endswith('.pdf'):
            pdf_files.append(os.path.join(root, file))

print(f"Found {len(pdf_files)} PDF files to process")

# Process with progress tracking
successful_count = 0
failed_files = []

for i, pdf_file in enumerate(pdf_files):
    print(f"Processing {i+1}/{len(pdf_files)}: {Path(pdf_file).name}")
    
    result = processor.process_single_pdf(pdf_file)
    
    if result.success:
        successful_count += 1
        print(f"✅ Success ({result.processing_time:.2f}s)")
    else:
        failed_files.append((pdf_file, result.error_message))
        print(f"❌ Failed: {result.error_message}")

print(f"\n📊 Processing Summary:")
print(f"✅ Successful: {successful_count}")
print(f"❌ Failed: {len(failed_files)}")

if failed_files:
    print("\nFailed files:")
    for file_path, error in failed_files:
        print(f"  {Path(file_path).name}: {error}")
```

### Example 5: Vector Database Operations

```python
from src.MilvusManager import MilvusManager, VectorDocument

db_manager = MilvusManager()

# Create a collection for specific domain
collection_name = "legal_documents"
if not db_manager.collection_exists(collection_name):
    db_manager.create_collection(collection_name)

# Insert custom documents
documents = [
    VectorDocument(
        id="doc_1",
        text="Contract law governs agreements between parties...",
        vector=db_manager.get_embeddings("Contract law governs agreements between parties..."),
        metadata={"document_type": "legal", "category": "contract_law"},
        file_path="legal_docs/contract_basics.pdf",
        page_number=1
    )
]

# Insert documents
db_manager.insert_documents(documents, collection_name)

# Perform semantic search
results = db_manager.search_documents(
    "What are the requirements for a valid contract?",
    collection_name=collection_name,
    k=3
)

for result in results:
    print(f"📄 {result.file_path}")
    print(f"📝 {result.text}")
    print(f"🏷️ Category: {result.metadata.get('category', 'Unknown')}")
    print(f"🎯 Score: {result.score:.3f}\n")
```

## 📚 API Documentation

### `DoclingPDFProcessor`

Main class for processing PDF files using the Docling library with optional vector database integration.

#### Constructor

```python
DoclingPDFProcessor(config_path: str = "config.yaml", use_database: bool = True)
```

**Parameters:**

- `config_path`: Path to YAML configuration file
- `use_database`: Whether to enable Milvus database integration

#### Methods

##### `process_single_pdf(file_path: str) -> ProcessingResult`

Process a single PDF file with text and image extraction.

**Parameters:**

- `file_path`: Path to the PDF file

**Returns:**

- `ProcessingResult` object with processing details

##### `process_directory(directory_path: str) -> List[ProcessingResult]`

Process all PDF files in a directory with batch processing.

**Parameters:**

- `directory_path`: Path to directory containing PDFs

**Returns:**

- List of `ProcessingResult` objects

##### `save_results(results: List[ProcessingResult], output_dir: str = "output") -> None`

Save processing results to markdown files with automatic naming.

**Parameters:**

- `results`: List of processing results
- `output_dir`: Output directory for markdown files

##### `get_document_info(result: ProcessingResult) -> Dict[str, Any]`

Extract comprehensive metadata from processed document.

**Parameters:**

- `result`: Processing result containing document

**Returns:**

- Dictionary with document information including page count, tables, images

### `MilvusManager`

Vector database manager for document storage and retrieval.

#### Key Methods

##### `create_collection(collection_name: str, dimension: int = 768) -> bool`

Create a new vector collection.

##### `insert_documents(documents: List[VectorDocument], collection_name: str = None) -> bool`

Insert documents into the vector database.

##### `search_documents(query: str, k: int = 5, collection_name: str = None) -> List[SearchResult]`

Perform semantic search for similar documents.

### `ProcessingResult`

Data class containing processing results and metadata.

**Attributes:**

- `success: bool` - Whether processing succeeded
- `file_path: str` - Path to processed file
- `document: Optional[Any]` - Docling document object
- `markdown_content: Optional[str]` - Converted markdown content
- `error_message: Optional[str]` - Error message if failed
- `processing_time: Optional[float]` - Processing time in seconds

### `VectorDocument`

Data class for vector database documents.

**Attributes:**

- `id: str` - Unique document identifier
- `text: str` - Document text content
- `vector: List[float]` - Document embedding vector
- `metadata: Optional[Dict]` - Additional metadata
- `file_path: Optional[str]` - Source file path
- `page_number: Optional[int]` - Page number in source document

## 🧪 Testing

The project includes comprehensive test suites for all major components.

### Running Tests

```bash
# Run all tests
pytest

# Run specific test files
pytest test/test_processor.py -v
pytest test/test_milvus_manager.py -v
pytest test/test_processor_with_database.py -v

# Run tests with coverage
pytest --cov=src test/ --cov-report=html
```

### Test Coverage

#### `test_processor.py` - PDF Processing Tests

- ✅ **Processor Initialization**: Test configuration loading and setup
- ✅ **Single PDF Processing**: Validate PDF conversion to markdown
- ✅ **Batch Processing**: Test directory processing capabilities
- ✅ **Error Handling**: Verify graceful failure handling
- ✅ **Configuration Loading**: Test YAML config parsing
- ✅ **Document Info Extraction**: Validate metadata extraction

```python
# Example test usage
def test_single_pdf_processing():
    processor = DoclingPDFProcessor(use_database=False)
    result = processor.process_single_pdf("test_documents/sample.pdf")
    assert result.success == True
    assert result.markdown_content is not None
    assert result.processing_time > 0
```

#### `test_milvus_manager.py` - Vector Database Tests

- ✅ **Database Connection**: Test Milvus client initialization
- ✅ **Collection Management**: Create, list, and delete collections
- ✅ **Document Insertion**: Insert vectors with metadata
- ✅ **Similarity Search**: Test semantic search functionality
- ✅ **Error Handling**: Database connection and operation errors

```python
# Example database test
def test_document_search():
    manager = MilvusManager()
    documents = [VectorDocument(id="test_1", text="AI research", vector=[...])]
    manager.insert_documents(documents)
    
    results = manager.search_documents("artificial intelligence", k=1)
    assert len(results) > 0
    assert results[0].score > 0.5
```

#### `test_processor_with_database.py` - Integration Tests

- ✅ **End-to-End Processing**: PDF processing with vector storage
- ✅ **Database Integration**: Verify automatic vector insertion
- ✅ **Search Integration**: Test processing → storage → retrieval workflow

### Test Configuration

The tests use mock configurations and temporary files to avoid external dependencies:

```python
@pytest.fixture
def mock_config():
    return {
        "document": {
            "doc_dir": "test_documents",
            "max_tokens": 256
        },
        "database": {
            "uri": "http://localhost:19530",
            "collection_name": "test_collection"
        }
    }
```

### Interactive Testing with Notebooks

Explore features interactively using Jupyter notebooks:

- **`docling-basic.ipynb`**: Basic PDF processing examples
- **`docling-ocr.ipynb`**: OCR and image processing demonstrations
- **`docling-hybrid-chunking.ipynb`**: Advanced document segmentation
- **`docling-picture-annotate.ipynb`**: Automatic image description
- **`docling-serialization.ipynb`**: Document export/import examples
- **`milvus.ipynb`**: Vector database operations and examples

### Example Output

When processing completes successfully, you'll see output like:

```text
🚀 Simple Multimodal RAG Application
Using DoclingPDFProcessor for PDF processing
==================================================
Processing PDFs from: documents
2025-09-29 09:42:39,569 - INFO - Document converter initialized successfully
2025-09-29 09:42:39,570 - INFO - Processing PDF: documents/research_paper.pdf
2025-09-29 09:42:41,234 - INFO - Successfully processed documents/research_paper.pdf in 1.66s

✅ Successfully processed: 1 files
❌ Failed to process: 0 files

Processed files:
  • research_paper.pdf (1.66s)

📄 Markdown files saved to 'output' directory
```

## 📁 Project Structure

```text
simple-multimodal-rag-application/
├── 📂 src/                              # Source code
│   ├── __init__.py                      # Package initialization
│   ├── DoclingPDFProcessor.py           # Main PDF processing class
│   └── MilvusManager.py                 # Vector database manager
├── 📂 test/                             # Test suites
│   ├── test_processor.py                # PDF processor tests
│   ├── test_milvus_manager.py           # Database manager tests
│   └── test_processor_with_database.py  # Integration tests
├── 📂 notebooks/                        # Jupyter notebooks for examples
│   ├── docling-basic.ipynb              # Basic processing examples
│   ├── docling-ocr.ipynb                # OCR demonstrations
│   ├── docling-hybrid-chunking.ipynb    # Advanced chunking
│   ├── docling-picture-annotate.ipynb   # Image description
│   ├── docling-serialization.ipynb      # Export/import examples
│   └── milvus.ipynb                     # Vector database examples
├── 📂 documents/                        # Input PDF files
├── 📂 output/                           # Generated markdown files
├── 📂 volumes/milvus/                   # Milvus database storage
├── 📂 db/win32/                         # Database utilities
│   └── standalone.bat                   # Milvus startup script
├── 📂 docs/                             # Documentation
│   └── MILVUS_INTEGRATION.md            # Database integration guide
├── 📄 config.yaml                       # Main configuration file
├── 📄 sample.py                         # Application entry point
├── 📄 milvus_integration.py             # Application entry point
├── 📄 pyproject.toml                    # Project dependencies
├── 📄 uv.lock                           # Dependency lock file
└── 📄 README.md                         # This documentation
```

## 📦 Dependencies

### Core Dependencies

- **`docling>=2.53.0`** - Advanced PDF processing library with multimodal support
- **`docling-core>=2.48.1`** - Core docling functionality and data models
- **`pymilvus[milvus-lite,model]>=2.6.2`** - Vector database client with lite version
- **`pyyaml>=6.0.2`** - YAML configuration file parsing
- **`transformers>=4.56.1`** - Hugging Face transformers for NLP models

### Optional Dependencies

- **`langchain>=0.3.27`** - Framework for building LLM applications
- **`langchain-docling>=1.1.0`** - Docling integration for LangChain
- **`ipykernel>=6.30.1`** - Jupyter notebook kernel support
- **`ipywidgets>=8.1.7`** - Interactive widgets for notebooks
- **`pytest>=8.4.2`** - Testing framework

### Development Tools

- **UV Package Manager** - Fast Python package management
- **Pytest** - Comprehensive testing framework
- **Jupyter Notebooks** - Interactive development and examples

## 🛠️ Advanced Usage

### Custom Pipeline Configuration

```python
from src.DoclingPDFProcessor import DoclingPDFProcessor

# Custom configuration for scientific papers
scientific_config = {
    "document": {
        "image_resolution_scale": 4,  # Higher resolution for figures
        "max_tokens": 2048,           # Larger chunks for complex content
        "picture_description": {
            "prompt_picture_description": "Analyze this scientific figure, chart, or diagram. Describe the data, methodology, and key findings presented."
        }
    }
}

# Save and use custom config
import yaml
with open("scientific_config.yaml", "w") as f:
    yaml.dump(scientific_config, f)

processor = DoclingPDFProcessor("scientific_config.yaml")
```

### Production Deployment

```python
import logging
from pathlib import Path

# Setup production logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_processing.log'),
        logging.StreamHandler()
    ]
)

class ProductionRAGPipeline:
    def __init__(self):
        self.processor = DoclingPDFProcessor()
        self.processed_files = set()
        
    def monitor_directory(self, watch_dir: str):
        """Monitor directory for new PDFs and process them automatically"""
        watch_path = Path(watch_dir)
        
        for pdf_file in watch_path.glob("*.pdf"):
            if str(pdf_file) not in self.processed_files:
                logging.info(f"Processing new file: {pdf_file}")
                result = self.processor.process_single_pdf(str(pdf_file))
                
                if result.success:
                    self.processed_files.add(str(pdf_file))
                    logging.info(f"Successfully processed: {pdf_file}")
                else:
                    logging.error(f"Failed to process {pdf_file}: {result.error_message}")

# Usage
pipeline = ProductionRAGPipeline()
pipeline.monitor_directory("incoming_documents/")
```

### Integration with LangChain

```python
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Milvus
from langchain.embeddings import HuggingFaceEmbeddings

# Process documents with DoclingPDFProcessor
processor = DoclingPDFProcessor()
results = processor.process_directory("documents/")

# Convert to LangChain documents
from langchain.schema import Document

documents = []
for result in results:
    if result.success:
        doc = Document(
            page_content=result.markdown_content,
            metadata={
                "source": result.file_path,
                "processing_time": result.processing_time
            }
        )
        documents.append(doc)

# Setup text splitter and embeddings
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Split documents and create vector store
split_docs = text_splitter.split_documents(documents)
vectorstore = Milvus.from_documents(
    split_docs,
    embeddings,
    connection_args={"host": "localhost", "port": "19530"}
)

# Query the vectorstore
query = "What are the main findings about machine learning?"
relevant_docs = vectorstore.similarity_search(query, k=3)
```

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### Development Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/vtanathip/simple-multimodal-rag-application.git
   cd simple-multimodal-rag-application
   ```

2. **Install dependencies**

   ```bash
   uv sync  # or pip install -e .
   ```

3. **Setup Milvus (optional)**

   ```bash
   ./db/win32/standalone.bat start
   ```

### Testing Guidelines

- **Write tests** for all new features
- **Run full test suite** before submitting PRs
- **Maintain test coverage** above 80%
- **Test both success and error cases**

```bash
# Run tests with coverage
pytest --cov=src test/ --cov-report=html --cov-report=term
```

### Code Style

- Follow **PEP 8** Python style guidelines
- Use **type hints** for function parameters and returns
- Add **docstrings** for all public methods
- Use **meaningful variable names**

### Pull Request Process

1. **Create feature branch** from `main`
2. **Implement changes** with tests
3. **Update documentation** if needed
4. **Ensure all tests pass**
5. **Submit pull request** with clear description

### Issue Reporting

When reporting issues, please include:

- **Python version** and operating system
- **Error messages** and stack traces
- **Steps to reproduce** the issue
- **Sample files** if applicable (without sensitive data)

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[Docling](https://github.com/DS4SD/docling)** - Advanced PDF processing capabilities
- **[Milvus](https://milvus.io/)** - High-performance vector database
- **[LangChain](https://langchain.com/)** - Framework for LLM applications
- **Community contributors** who help improve this project
