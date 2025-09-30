# Multimodal RAG API

A FastAPI-based REST API for interacting with a LangGraph multimodal RAG (Retrieval-Augmented Generation) agent. This API provides endpoints for document processing, querying, and collection management.

## Features

- **Document Processing**: Upload and process PDF documents using Docling
- **Vector Search**: Semantic search using Milvus vector database
- **LLM Integration**: Powered by Ollama for response generation
- **LangGraph Workflow**: Structured agent workflow for complex reasoning
- **Async Processing**: Asynchronous document processing capabilities
- **REST API**: Full REST API with automatic documentation
- **Docker Support**: Complete Docker setup with all dependencies

## Quick Start

### Prerequisites

- Python 3.12+
- Ollama running locally (or accessible via URL)
- Milvus vector database

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd simple-multimodal-rag-application
   ```

2. **Install dependencies**:
   ```bash
   pip install uv
   uv sync
   ```

3. **Start Ollama** (if not already running):
   ```bash
   ollama serve
   ollama pull llama3.2
   ```

4. **Start Milvus** (using Docker):
   ```bash
   # Start Milvus standalone
   cd db/win32
   ./standalone.bat
   ```

5. **Run the API**:
   ```bash
   python run_api.py
   ```

The API will be available at `http://localhost:8000`

### Docker Setup

For a complete setup including all dependencies:

```bash
docker-compose up -d
```

This will start:
- RAG API service
- Ollama service
- Milvus vector database
- Required supporting services (etcd, minio)

## API Endpoints

### Core Endpoints

#### Health Check
```http
GET /health
```
Returns the health status of the API and its components.

#### Process Query
```http
POST /query
```
Process a user query through the RAG agent.

**Request Body**:
```json
{
  "query": "What is machine learning?",
  "thread_id": "optional-session-id"
}
```

**Response**:
```json
{
  "answer": "Machine learning is...",
  "sources": [
    {
      "text": "Relevant document content",
      "file_path": "document.pdf",
      "distance": 0.85,
      "metadata": {}
    }
  ],
  "processing_info": {
    "processing_status": "search_only",
    "context_length": 1500
  },
  "error": null
}
```

### Document Management

#### Upload Document
```http
POST /upload
```
Upload and process a PDF document.

**Form Data**:
- `file`: PDF file to upload
- `process_immediately`: Boolean (default: true)

#### Process Existing Document
```http
POST /process-document
```
Process an existing document by file path.

**Form Data**:
- `file_path`: Path to the document

### Collection Management

#### Get Collection Statistics
```http
GET /collection/stats
```
Returns statistics about the vector collection.

#### Clear Collection
```http
DELETE /collection/clear
```
Clear all documents from the collection (implementation required).

## Usage Examples

### Python Client

```python
import asyncio
from api.client_example import RAGAPIClient

async def main():
    async with RAGAPIClient("http://localhost:8000") as client:
        # Health check
        health = await client.health_check()
        print(f"API Status: {health['status']}")
        
        # Query the system
        response = await client.query("What is deep learning?")
        print(f"Answer: {response['answer']}")
        
        # Upload a document
        result = await client.upload_document("path/to/document.pdf")
        print(f"Upload success: {result['success']}")

if __name__ == "__main__":
    asyncio.run(main())
```

### cURL Examples

**Health Check**:
```bash
curl -X GET "http://localhost:8000/health"
```

**Query**:
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is machine learning?", "thread_id": "session1"}'
```

**Upload Document**:
```bash
curl -X POST "http://localhost:8000/upload" \
     -F "file=@document.pdf" \
     -F "process_immediately=true"
```

## Configuration

### Environment Variables

The API can be configured using environment variables with the `RAG_API_` prefix:

- `RAG_API_HOST`: API host (default: 0.0.0.0)
- `RAG_API_PORT`: API port (default: 8000)
- `RAG_API_MODEL_NAME`: Ollama model name (default: llama3.2)
- `RAG_API_OLLAMA_BASE_URL`: Ollama base URL (default: http://localhost:11434)
- `RAG_API_CONFIG_PATH`: Configuration file path (default: config.yaml)

### Configuration File

Create a `.env` file in the project root:

```env
RAG_API_HOST=0.0.0.0
RAG_API_PORT=8000
RAG_API_MODEL_NAME=llama3.2
RAG_API_OLLAMA_BASE_URL=http://localhost:11434
RAG_API_UPLOAD_DIR=uploads
RAG_API_MAX_FILE_SIZE=104857600
```

## API Documentation

Once the API is running, you can access:

- **Interactive API docs**: http://localhost:8000/docs
- **ReDoc documentation**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## Architecture

The API is built using:

- **FastAPI**: Modern Python web framework
- **LangGraph**: Agent workflow orchestration
- **Ollama**: Local LLM inference
- **Milvus**: Vector database for embeddings
- **Docling**: PDF processing and extraction
- **Pydantic**: Data validation and settings

### Agent Workflow

The LangGraph agent follows this workflow:

1. **Query Analysis**: Determine if document processing is needed
2. **Document Processing** (if required): Process new documents with Docling
3. **Knowledge Search**: Perform vector search in Milvus
4. **Response Generation**: Generate answer using Ollama LLM

## Development

### Running in Development Mode

```bash
# Install development dependencies
uv sync --dev

# Run with auto-reload
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Testing

```bash
# Run tests
pytest test/

# Run specific test
pytest test/test_api.py -v
```

### Code Quality

```bash
# Format code
black api/
isort api/

# Type checking
mypy api/

# Linting
flake8 api/
```

## Troubleshooting

### Common Issues

1. **Ollama Connection Error**:
   - Ensure Ollama is running: `ollama serve`
   - Check the base URL in configuration
   - Verify the model is available: `ollama list`

2. **Milvus Connection Error**:
   - Ensure Milvus is running
   - Check connection settings in `config.yaml`
   - Verify port 19530 is accessible

3. **Document Processing Fails**:
   - Check file format (only PDF supported)
   - Verify file size limits
   - Check upload directory permissions

4. **Memory Issues**:
   - Reduce batch sizes in configuration
   - Increase container memory limits
   - Monitor resource usage

### Logs

Check application logs for detailed error information:

```bash
# View API logs
docker-compose logs rag-api

# View all service logs
docker-compose logs
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.