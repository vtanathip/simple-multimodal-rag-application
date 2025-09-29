# Milvus Database Integration

This document describes the Milvus vector database integration added to the simple-multimodal-rag-application.

## Overview

The application now includes two main classes for handling document processing and vector database operations:

1. **MilvusManager**: Manages all Milvus database operations
2. **DoclingPDFProcessor**: Enhanced with automatic database storage capabilities

## Features

### MilvusManager
- ✅ Automatic collection creation and management
- ✅ Document insertion with text embedding
- ✅ Semantic search capabilities
- ✅ Collection statistics and monitoring
- ✅ Error handling and graceful fallbacks

### Enhanced DoclingPDFProcessor
- ✅ Automatic text chunking and embedding
- ✅ Seamless database integration during PDF processing
- ✅ Search functionality for processed documents
- ✅ Optional database usage (can be disabled)

## Configuration

All database settings are configured in `config.yaml`:

```yaml
# Vector Database Configuration
database:
  uri: "http://localhost:19530"  # Milvus database URI
  name: "rag_multimodal"  # Database name
  collection_name: "rag_collection"  # Collection name for processed documents
  namespace: "CaseDoneDemo"  # Namespace for partitioning data
```

## Quick Start

### 1. Start Milvus

Download and start Milvus using the provided script:

```bash
# Download the script (Windows)
Invoke-WebRequest https://raw.githubusercontent.com/milvus-io/milvus/refs/heads/master/scripts/standalone_embed.bat -OutFile standalone.bat

# Start Milvus
standalone.bat start

# Web UI available at: http://localhost:9091/webui/
```

### 2. Basic Usage

```python
from src.DoclingPDFProcessor import DoclingPDFProcessor
from src.MilvusManager import MilvusManager

# Initialize processor with database integration
processor = DoclingPDFProcessor(use_database=True)

# Process PDF files (automatically saves to database)
results = processor.process_directory("documents")

# Search for similar content
search_results = processor.search_documents("artificial intelligence", limit=5)

# Get database statistics
stats = processor.get_database_stats()

# Close database connection
processor.close_database()
```

### 3. Direct Database Operations

```python
from src.MilvusManager import MilvusManager

# Initialize manager
db_manager = MilvusManager()

# Create collection
db_manager.create_collection("my_collection")

# Insert text documents (with automatic embedding)
texts = ["Document 1 content", "Document 2 content"]
metadata = [{"source": "doc1.pdf"}, {"source": "doc2.pdf"}]
db_manager.insert_text_documents(texts, metadata_list=metadata)

# Search for similar documents
results = db_manager.search("search query", limit=10)

# Clean up
db_manager.close()
```

## API Reference

### MilvusManager

#### Methods

- `create_collection(collection_name, drop_existing=False)`: Create or verify collection
- `insert_documents(documents)`: Insert VectorDocument objects
- `insert_text_documents(texts, metadata_list, file_paths)`: Insert texts with automatic embedding
- `search(query, limit=10)`: Semantic search using text query
- `delete_collection(collection_name)`: Delete a collection
- `get_collection_stats(collection_name)`: Get collection information
- `close()`: Close database connection

#### Data Classes

```python
@dataclass
class VectorDocument:
    id: str
    text: str
    vector: List[float]
    metadata: Optional[Dict[str, Any]] = None
    file_path: Optional[str] = None
    page_number: Optional[int] = None
    chunk_index: Optional[int] = None

@dataclass
class SearchResult:
    id: str
    text: str
    score: float
    metadata: Optional[Dict[str, Any]] = None
    file_path: Optional[str] = None
    page_number: Optional[int] = None
```

### DoclingPDFProcessor (Enhanced)

#### New Methods

- `search_documents(query, limit=10)`: Search processed documents
- `get_database_stats()`: Get database collection statistics
- `close_database()`: Close database connection

#### New Parameters

- `use_database: bool = True`: Enable/disable database integration

## Testing

The implementation includes comprehensive test suites:

```bash
# Test MilvusManager
pytest test/test_milvus_manager.py -v

# Test DoclingPDFProcessor with database integration
pytest test/test_processor_with_database.py -v

# Test updated processor functionality
pytest test/test_processor.py -v

# Run all tests
pytest test/ -v
```

## Demo Script

Run the demonstration script to see the integration in action:

```bash
python demo.py
```

The demo script will:
1. Check Milvus connectivity
2. Process PDF documents from the `documents` directory
3. Demonstrate search functionality
4. Show database statistics
5. Handle cases where no PDFs are available

## Text Chunking

The system automatically chunks long documents based on:
- Maximum token count (configurable in `config.yaml`)
- Paragraph boundaries
- Sentence boundaries

Default chunking parameters:
- `max_tokens: 512` (approximate)
- Respects paragraph breaks
- Handles very long paragraphs by splitting

## Error Handling

The integration includes robust error handling:
- Graceful fallback when Milvus is unavailable
- Processing continues even if database operations fail
- Detailed logging for troubleshooting
- Optional database usage

## Performance Considerations

- Embedding generation uses the default PyMilvus embedding function
- Vector dimension: 768 (configurable)
- Similarity metric: Cosine similarity
- Batch processing for multiple documents
- Efficient text chunking algorithm

## Troubleshooting

### Common Issues

1. **Connection Error**: Ensure Milvus is running on `localhost:19530`
2. **Collection Not Found**: Collections are auto-created, check permissions
3. **Embedding Errors**: Verify PyMilvus model dependencies are installed
4. **Memory Issues**: Reduce `max_tokens` for large documents

### Logs

Check application logs for detailed error information:
- Database connection status
- Collection creation/verification
- Document insertion results
- Search operation details

## Dependencies

The integration requires these additional packages (already in `pyproject.toml`):

```toml
dependencies = [
    "pymilvus[milvus-lite,model]>=2.6.2",
    # ... other dependencies
]
```

## Web UI

Milvus provides a web interface for database management:
- URL: http://localhost:9091/webui/
- View collections, data, and statistics
- Monitor performance and operations