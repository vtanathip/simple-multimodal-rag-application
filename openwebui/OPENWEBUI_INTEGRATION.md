# Open WebUI Integration Guide

## Overview

This guide explains how to integrate your Multimodal RAG application with Open WebUI as a custom pipeline. This allows you to use your advanced document processing capabilities directly within the Open WebUI chat interface.

## Features

- 📄 **PDF Document Upload**: Add documents directly via chat commands
- 🔍 **Automatic RAG**: Context is automatically injected based on user queries
- 💬 **Chat Commands**: Built-in commands for document management
- ⚙️ **Configurable**: Adjustable settings through Open WebUI interface
- 🔄 **Real-time**: Instant document processing and retrieval

## Installation Steps

### 1. Install Open WebUI

```bash
# Option 1: Docker (Recommended)
docker run -d -p 3000:8080 --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main

# Option 2: Local Installation
pip install open-webui
```

### 2. Prepare Your Environment

```bash
# Ensure your RAG application dependencies are installed
uv sync

# Make sure Milvus is running
cd db/win32
./standalone.bat

# Make sure Ollama is running
ollama serve
```

### 3. Install the Pipeline

1. **Copy the pipeline file** to Open WebUI's pipelines directory:
   ```bash
   # Default location (adjust based on your installation)
   cp openwebui_pipeline.py ~/.open-webui/pipelines/
   ```

2. **Or upload via Open WebUI interface**:
   - Go to Open WebUI Admin Panel → Pipelines
   - Click "Add Pipeline" 
   - Upload `openwebui_pipeline.py`

### 4. Configure the Pipeline

1. **Navigate to Pipeline Settings** in Open WebUI
2. **Find "Multimodal RAG Pipeline"** in the list
3. **Configure the valves** (settings):
   - `ENABLE_RAG`: Enable/disable RAG functionality
   - `MAX_SEARCH_RESULTS`: Number of documents to retrieve
   - `OLLAMA_BASE_URL`: Your Ollama server URL
   - `MILVUS_URI`: Your Milvus database URL
   - `SIMILARITY_THRESHOLD`: Minimum similarity score for results

## Usage

### Chat Commands

Once installed, you can use these commands in Open WebUI chat:

#### Upload Documents
```
/add_document documents/my_report.pdf
```

#### Check Status
```
/stats
```

#### Get Help
```
/help
```

### Automatic RAG

Simply ask questions and the pipeline will automatically:
1. Search your document collection
2. Find relevant content
3. Enhance your query with context
4. Provide informed responses

**Example:**
```
User: What are the main ESG findings in the report?
Assistant: [Automatically searches documents and provides contextualized answer]
```

## Configuration Options

### Pipeline Valves (User Configurable)

| Setting | Description | Default |
|---------|-------------|---------|
| `ENABLE_RAG` | Enable RAG functionality | `True` |
| `MAX_SEARCH_RESULTS` | Max documents to retrieve | `5` |
| `SIMILARITY_THRESHOLD` | Min similarity score | `0.3` |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |
| `MILVUS_URI` | Milvus database URL | `http://localhost:19530` |
| `DOCUMENT_UPLOAD_ENABLED` | Enable document upload commands | `True` |

### Advanced Configuration

Edit `openwebui_config.yaml` for advanced settings:
- Document processing parameters
- Embedding model configuration
- Database collection settings

## Architecture

```
Open WebUI Chat Interface
         ↓
   Pipeline (Filter Type)
         ↓
   Multimodal RAG Agent
         ↓
┌─────────────────┬─────────────────┐
│   Docling       │    Milvus       │
│ PDF Processor   │ Vector Database │
└─────────────────┴─────────────────┘
```

### Flow Description

1. **User Input**: User types message in Open WebUI
2. **Pipeline Inlet**: Pipeline intercepts message
3. **Command Detection**: Check for special commands (`/add_document`, `/stats`)
4. **RAG Processing**: If not a command, search vector database
5. **Context Injection**: Add relevant documents to query
6. **LLM Processing**: Enhanced query goes to Ollama
7. **Pipeline Outlet**: Final response processing

## Advanced Features

### Custom Document Types

Extend the pipeline to support additional document types:

```python
# In the pipeline file
SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt", ".md"]
```

### Custom Embeddings

Configure different embedding models:

```yaml
# In openwebui_config.yaml
model:
  embeddings: "your-custom-embedding-model"
```

### Multi-Collection Support

Support multiple document collections:

```python
# Add to pipeline valves
COLLECTIONS: List[str] = ["rag_collection", "legal_docs", "technical_specs"]
```

## Troubleshooting

### Common Issues

1. **Pipeline Not Loading**
   - Check file permissions
   - Verify Python dependencies
   - Check Open WebUI logs

2. **RAG Not Working**
   - Verify Milvus is running: `http://localhost:19530`
   - Check collection exists and has data
   - Verify embeddings model is available

3. **Document Upload Fails**
   - Check file path accessibility
   - Verify supported file format
   - Check Docling dependencies

### Debug Mode

Enable detailed logging by setting:
```python
# In pipeline
logging.basicConfig(level=logging.DEBUG)
```

## Integration with Existing Workflow

### Batch Document Upload

Pre-populate your knowledge base:

```bash
# Using your existing CLI
uv run python main.py --add-document documents/report1.pdf
uv run python main.py --add-document documents/report2.pdf
```

### API Integration

The pipeline can be extended to support REST API calls for document management.

## Performance Considerations

- **Memory Usage**: Embedding models and vector databases use significant memory
- **Processing Time**: Large PDFs may take time to process
- **Concurrent Users**: Consider scaling Milvus for multiple users
- **Caching**: Implement response caching for frequently asked questions

## Security Considerations

- **File Access**: Ensure proper file system permissions
- **Network Security**: Secure Milvus and Ollama endpoints
- **User Isolation**: Consider user-specific collections for multi-tenant setups

## Future Enhancements

- **Multi-modal Support**: Image and audio document processing
- **Streaming Responses**: Real-time response generation
- **User Authentication**: Per-user document access control
- **Advanced Analytics**: Usage tracking and performance monitoring