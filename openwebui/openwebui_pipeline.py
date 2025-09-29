"""
Open WebUI Pipeline for Multimodal RAG Application
Integrates the DoclingPDFProcessor and MilvusManager with Open WebUI
"""

import os
import asyncio
import logging
from typing import List, Union, Generator, Iterator, Dict, Any, Optional
from pydantic import BaseModel

# Pipeline required imports
try:
    from utils.pipelines.main import get_last_user_message
except ImportError:
    # Fallback for testing without Open WebUI
    def get_last_user_message(messages: List[Dict]) -> Optional[Dict]:
        """Fallback implementation for testing"""
        for message in reversed(messages):
            if message.get("role") == "user":
                return message
        return None


class Pipeline:
    """Open WebUI Pipeline for Multimodal RAG with Docling and Milvus"""

    class Valves(BaseModel):
        """Pipeline configuration valves (settings that users can modify)"""
        ENABLE_RAG: bool = True
        MAX_SEARCH_RESULTS: int = 5
        OLLAMA_BASE_URL: str = "http://localhost:11434"
        MILVUS_URI: str = "http://localhost:19530"
        CONFIG_PATH: str = "config.yaml"
        COLLECTION_NAME: str = "rag_collection"
        SIMILARITY_THRESHOLD: float = 0.3
        DOCUMENT_UPLOAD_ENABLED: bool = True

    def __init__(self):
        """Initialize the pipeline"""
        self.type = "filter"  # Can be "filter", "function", or "manifold"
        self.id = "multimodal_rag_pipeline"
        self.name = "Multimodal RAG Pipeline"
        self.description = "Advanced document processing and retrieval using Docling and Milvus"

        self.valves = self.Valves()
        self.logger = self._setup_logging()

        # Initialize components lazily
        self.agent = None
        self.milvus_manager = None

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(f"OpenWebUI.{self.id}")

    def _initialize_components(self):
        """Initialize RAG components if not already done"""
        if self.agent is None:
            try:
                # Import here to avoid issues if dependencies aren't available
                from src.agent.multimodal_rag_agent import MultimodalRAGAgent
                from src.MilvusManager import MilvusManager

                self.agent = MultimodalRAGAgent(
                    config_path=self.valves.CONFIG_PATH,
                    ollama_base_url=self.valves.OLLAMA_BASE_URL
                )

                self.milvus_manager = MilvusManager(
                    config_path=self.valves.CONFIG_PATH)

                self.logger.info(
                    "Multimodal RAG components initialized successfully")

            except Exception as e:
                self.logger.error(f"Failed to initialize RAG components: {e}")
                self.agent = None
                self.milvus_manager = None

    async def on_startup(self):
        """Called when the pipeline starts"""
        self.logger.info("Starting Multimodal RAG Pipeline...")
        self._initialize_components()

    async def on_shutdown(self):
        """Called when the pipeline shuts down"""
        self.logger.info("Shutting down Multimodal RAG Pipeline...")

    async def on_valves_updated(self):
        """Called when valve settings are updated"""
        self.logger.info(
            "Pipeline valves updated, reinitializing components...")
        self.agent = None
        self.milvus_manager = None
        self._initialize_components()

    def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """
        Process incoming requests before they go to the LLM
        This is where we can inject RAG context
        """
        if not self.valves.ENABLE_RAG:
            return body

        try:
            self._initialize_components()

            if self.agent is None:
                self.logger.warning(
                    "RAG components not available, skipping RAG processing")
                return body

            # Get the last user message
            last_message = get_last_user_message(body.get("messages", []))
            if not last_message:
                return body

            user_query = last_message.get("content", "")

            # Check for document upload commands
            if self._handle_document_commands(user_query, body):
                return body

            # Perform RAG search
            search_results = self.milvus_manager.search(
                query=user_query,
                limit=self.valves.MAX_SEARCH_RESULTS
            )

            # Filter results by similarity threshold
            relevant_results = [
                result for result in search_results
                if result.score >= self.valves.SIMILARITY_THRESHOLD
            ]

            if relevant_results:
                # Build context from search results
                context_parts = []
                context_parts.append("📚 **Retrieved Context:**")

                for i, result in enumerate(relevant_results, 1):
                    source = result.file_path or "Unknown source"
                    context_parts.append(f"\n**Source {i}** ({source}):")
                    context_parts.append(f"{result.text}")
                    context_parts.append("---")

                context = "\n".join(context_parts)

                # Modify the user message to include context
                enhanced_query = f"""Based on the following context, please answer the user's question:

{context}

**User Question:** {user_query}

Please provide a comprehensive answer based on the context. If the context doesn't contain relevant information, mention that clearly."""

                # Update the last message
                if body.get("messages"):
                    body["messages"][-1]["content"] = enhanced_query

                self.logger.info(
                    f"Enhanced query with {len(relevant_results)} relevant documents")
            else:
                self.logger.info(
                    "No relevant documents found for RAG enhancement")

        except Exception as e:
            self.logger.error(f"Error in RAG processing: {e}")
            # Continue without RAG enhancement if there's an error

        return body

    def _handle_document_commands(self, user_query: str, body: dict) -> bool:
        """
        Handle document upload and management commands
        Returns True if the query was handled as a command
        """
        if not self.valves.DOCUMENT_UPLOAD_ENABLED:
            return False

        query_lower = user_query.lower().strip()

        # Handle /add_document command
        if query_lower.startswith("/add_document ") or query_lower.startswith("/add "):
            file_path = user_query.split(
                " ", 1)[1].strip() if " " in user_query else ""

            if file_path:
                result = self.agent.add_document(file_path)

                if result["success"]:
                    response_message = f"""✅ **Document Added Successfully!**

**File:** {result['file_path']}
**Processing Time:** {result.get('processing_time', 'Unknown')} seconds

The document has been processed and added to the knowledge base. You can now ask questions about its content."""
                else:
                    response_message = f"""❌ **Failed to Add Document**

**File:** {file_path}
**Error:** {result.get('error_message', 'Unknown error')}

Please check the file path and ensure the file is accessible."""

            else:
                response_message = """📝 **Document Upload Usage**

To add a document to the knowledge base, use:
`/add_document <file_path>`

Example: `/add_document documents/my_report.pdf`"""

            # Replace the conversation with our response
            body["messages"] = [{
                "role": "assistant",
                "content": response_message
            }]
            return True

        # Handle /stats command
        elif query_lower in ["/stats", "/status", "/info"]:
            try:
                stats = self.agent.get_collection_stats()

                if "error" not in stats:
                    response_message = f"""📊 **Knowledge Base Statistics**

**Collection:** {stats.get('collection_name', 'Unknown')}
**Status:** {'✅ Active' if stats.get('exists', False) else '❌ Not Found'}
**Total Documents:** Available in collection"""
                else:
                    response_message = f"❌ **Error getting statistics:** {stats['error']}"

            except Exception as e:
                response_message = f"❌ **Error getting statistics:** {str(e)}"

            body["messages"] = [{
                "role": "assistant",
                "content": response_message
            }]
            return True

        # Handle /help command
        elif query_lower in ["/help", "/commands"]:
            response_message = """🤖 **Multimodal RAG Pipeline Help**

**Available Commands:**
- `/add_document <path>` - Add a PDF document to the knowledge base
- `/stats` - Show knowledge base statistics  
- `/help` - Show this help message

**Features:**
- 📄 **PDF Processing** - Advanced document analysis with Docling
- 🔍 **Semantic Search** - Vector-based document retrieval with Milvus
- 🧠 **Context Enhancement** - Automatic context injection for better answers
- 💬 **Natural Queries** - Just ask questions about your documents!

**Example Usage:**
1. Upload a document: `/add_document documents/report.pdf`
2. Ask questions: "What are the key findings in the report?"
3. Get detailed answers based on document content!"""

            body["messages"] = [{
                "role": "assistant",
                "content": response_message
            }]
            return True

        return False

    def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """
        Process outgoing responses after LLM generation
        Can be used for post-processing or logging
        """
        return body


# Required for Open WebUI pipeline discovery
def get_pipeline():
    """Factory function required by Open WebUI"""
    return Pipeline()
