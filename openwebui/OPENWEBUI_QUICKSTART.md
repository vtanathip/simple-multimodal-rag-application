# Open WebUI Integration - Quick Start Guide

This guide will help you deploy your Multimodal RAG system as an Open WebUI pipeline for enhanced chat interface with document processing capabilities.

## 🚀 Quick Deployment

### Option 1: Automated Deployment
```bash
# Run the deployment script
python deploy_openwebui.py
```

### Option 2: Manual Deployment

1. **Install Open WebUI** (if not already installed):
```bash
pip install open-webui
```

2. **Copy pipeline file**:
```bash
# Find your Open WebUI pipelines directory (usually ~/.open-webui/pipelines/)
# Copy openwebui_pipeline.py to that directory
cp openwebui_pipeline.py ~/.open-webui/pipelines/multimodal_rag_pipeline.py
```

3. **Start required services**:
```bash
# Start Milvus (in a separate terminal)
cd db/win32
./standalone.bat

# Start Ollama (if not running)
ollama serve

# Start Open WebUI
open-webui serve
```

4. **Access Open WebUI**:
   - Open browser to: http://localhost:8080
   - Create account if first time

## ⚙️ Configuration

### Enable the Pipeline
1. Go to **Admin Panel** → **Pipelines**
2. Find **"Multimodal RAG Pipeline"**
3. Toggle it **ON**
4. Click **Settings** to configure

### Pipeline Settings (Valves)
Configure these settings in the Open WebUI admin panel:

| Setting | Default | Description |
|---------|---------|-------------|
| `ENABLE_RAG` | `true` | Enable/disable RAG enhancement |
| `MAX_SEARCH_RESULTS` | `5` | Maximum documents to retrieve |
| `SIMILARITY_THRESHOLD` | `0.7` | Minimum similarity for relevant results |
| `MILVUS_URI` | `http://localhost:19530` | Milvus database connection |
| `MILVUS_TOKEN` | `""` | Milvus authentication token (if needed) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `DEBUG_MODE` | `false` | Enable detailed logging |

## 🎯 Usage

### Document Commands
These commands work in the Open WebUI chat interface:

#### Add Documents
```
/add_document path/to/your/document.pdf
```
- Supports: PDF, TXT, MD files
- Processes with multimodal extraction
- Stores in vector database

#### Get Statistics
```
/stats
```
- Shows document count
- Database status
- System health

#### Get Help
```
/help
```
- Lists available commands
- Usage examples

### Enhanced Chat
Once documents are uploaded, simply ask questions:

```
What are the key findings in the research paper?
Summarize the main points from the uploaded documents.
Can you extract the methodology from the PDF?
```

The pipeline will automatically:
1. Search relevant document chunks
2. Enhance your query with context
3. Provide accurate, source-based answers

## 🔧 Architecture

### Pipeline Flow
```
User Message → Pipeline Inlet → Command Detection → RAG Enhancement → LLM → Pipeline Outlet
```

### Components
- **Inlet**: Processes incoming messages, handles commands
- **RAG Enhancement**: Searches documents, injects context
- **Outlet**: Processes responses, adds source information
- **Commands**: Special functions for document management

### Integration Points
- **Open WebUI**: Chat interface and pipeline framework
- **Milvus**: Vector database for document storage/search
- **Ollama**: LLM inference for chat responses
- **DoclingPDFProcessor**: Document processing and extraction

## 🐛 Troubleshooting

### Common Issues

#### Pipeline Not Appearing
```bash
# Check if file is in correct location
ls ~/.open-webui/pipelines/

# Check Open WebUI logs
open-webui serve --log-level debug
```

#### Milvus Connection Error
```bash
# Start Milvus
cd db/win32
./standalone.bat

# Check Milvus status
curl http://localhost:19530
```

#### Ollama Connection Error
```bash
# Start Ollama
ollama serve

# Check Ollama status
curl http://localhost:11434/api/tags
```

#### Documents Not Processing
1. Check file permissions
2. Verify file format (PDF, TXT, MD)
3. Check debug logs in Open WebUI
4. Ensure Milvus is running and accessible

### Debug Mode
Enable debug mode in pipeline settings to see detailed logs:
1. Admin Panel → Pipelines → Multimodal RAG Pipeline → Settings
2. Set `DEBUG_MODE` to `true`
3. Check Open WebUI logs for detailed information

### Performance Optimization
- Increase `MAX_SEARCH_RESULTS` for more comprehensive answers
- Adjust `SIMILARITY_THRESHOLD` to filter relevance
- Use faster embedding models for better performance
- Consider GPU acceleration for Ollama

## 📊 Monitoring

### Health Checks
The pipeline includes built-in health monitoring:
- Database connectivity
- Model availability
- Processing status

### Metrics
Use `/stats` command to monitor:
- Document count
- Processing success rate
- Average response time
- Database status

## 🔄 Updates

### Updating the Pipeline
1. Replace the pipeline file in the pipelines directory
2. Restart Open WebUI
3. Pipeline updates automatically

### Version Compatibility
- Open WebUI: 0.1.0+
- Milvus: 2.6.2+
- Python: 3.8+

## 🎉 Advanced Features

### Custom Commands
The pipeline supports extensible commands. Add new commands by modifying the `handle_command` method in the pipeline.

### Multi-Model Support
Configure different models for different tasks:
- Document processing: Use multimodal models
- Chat responses: Use conversational models
- Embeddings: Use specialized embedding models

### Batch Processing
Process multiple documents:
```
/add_document /path/to/folder/*.pdf
```

## 📚 Additional Resources

- [Open WebUI Documentation](https://docs.openwebui.com/)
- [Milvus Documentation](https://milvus.io/docs)
- [Ollama Documentation](https://ollama.ai/docs)
- [Pipeline Development Guide](./docs/PIPELINE_DEVELOPMENT.md)

## 💡 Tips

1. **Start Simple**: Begin with a few documents and basic queries
2. **Monitor Performance**: Use debug mode to understand processing
3. **Optimize Settings**: Adjust thresholds based on your content
4. **Regular Maintenance**: Clean up old documents periodically
5. **Backup Data**: Export important conversations and documents

---

**Need Help?** Check the troubleshooting section or enable debug mode for detailed logging.