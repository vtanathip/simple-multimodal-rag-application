"""
LangGraph Agent for Multimodal RAG Application
Main agent class that orchestrates document processing and retrieval using LangGraph
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from dataclasses import dataclass
from pathlib import Path

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from ..DoclingPDFProcessor import DoclingPDFProcessor
from ..MilvusManager import MilvusManager, SearchResult


@dataclass
class AgentResponse:
    """Data class for agent responses"""
    answer: str
    sources: List[SearchResult]
    processing_info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class AgentState(TypedDict):
    """State definition for the LangGraph agent"""
    messages: Annotated[List[BaseMessage], add_messages]
    user_query: str
    search_results: List[SearchResult]
    context: str
    answer: str
    file_path: Optional[str]
    processing_status: str


class MultimodalRAGAgent:
    """
    LangGraph-based agent for multimodal RAG operations
    Integrates Ollama LLM with document processing and vector search
    """

    def __init__(self,
                 config_path: str = "config.yaml",
                 model_name: str = "llama3.2",
                 ollama_base_url: str = "http://localhost:11434"):
        """
        Initialize the multimodal RAG agent

        Args:
            config_path: Path to configuration file
            model_name: Ollama model name to use
            ollama_base_url: Base URL for Ollama API
        """
        self.logger = self._setup_logging()
        self.config_path = config_path

        # Initialize components
        self.llm = ChatOllama(
            model=model_name,
            base_url=ollama_base_url,
            temperature=0.1
        )

        self.pdf_processor = DoclingPDFProcessor(
            config_path=config_path,
            use_database=True
        )

        self.milvus_manager = MilvusManager(config_path=config_path)

        # Initialize LangGraph
        self.memory = MemorySaver()
        self.graph = self._build_graph()
        self.app = self.graph.compile(checkpointer=self.memory)

        self.logger.info("MultimodalRAGAgent initialized successfully")

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("analyze_query", self._analyze_query)
        workflow.add_node("process_document", self._process_document)
        workflow.add_node("search_knowledge", self._search_knowledge)
        workflow.add_node("generate_response", self._generate_response)

        # Add edges
        workflow.add_edge(START, "analyze_query")
        workflow.add_conditional_edges(
            "analyze_query",
            self._route_query,
            {
                "process_doc": "process_document",
                "search": "search_knowledge"
            }
        )
        workflow.add_edge("process_document", "search_knowledge")
        workflow.add_edge("search_knowledge", "generate_response")
        workflow.add_edge("generate_response", END)

        return workflow

    async def _analyze_query(self, state: AgentState) -> AgentState:
        """Analyze the user query to determine the required actions"""
        try:
            user_query = state["user_query"]
            self.logger.info(f"Analyzing query: {user_query}")

            # Use LLM to classify the intent
            analysis_prompt = f"""
            Analyze this user query and determine if it requires:
            1. Document processing (if user mentions a file path or wants to process a document)
            2. Knowledge search (if user is asking questions about existing knowledge)
            
            Query: {user_query}
            
            Respond with either:
            - "PROCESS_DOCUMENT: <file_path>" if document processing is needed
            - "SEARCH_KNOWLEDGE" if only knowledge search is needed
            """

            response = await self.llm.ainvoke([HumanMessage(content=analysis_prompt)])
            intent = response.content.strip()

            # Extract file path if document processing is needed
            if intent.startswith("PROCESS_DOCUMENT"):
                file_path = intent.split(
                    ":", 1)[1].strip() if ":" in intent else None
                state["file_path"] = file_path
                state["processing_status"] = "needs_processing"
            else:
                state["processing_status"] = "search_only"

            self.logger.info(
                f"Query analysis result: {state['processing_status']}")
            return state

        except Exception as e:
            self.logger.error(f"Error analyzing query: {e}")
            state["processing_status"] = "error"
            return state

    def _route_query(self, state: AgentState) -> str:
        """Route the query based on analysis results"""
        if state["processing_status"] == "needs_processing":
            return "process_doc"
        else:
            return "search"

    async def _process_document(self, state: AgentState) -> AgentState:
        """Process a document using DoclingPDFProcessor"""
        try:
            file_path = state.get("file_path")
            if not file_path:
                state["processing_status"] = "error"
                return state

            self.logger.info(f"Processing document: {file_path}")

            # Process the document
            result = self.pdf_processor.process_single_pdf(file_path)

            if result.success:
                self.logger.info(
                    f"Document processed successfully: {file_path}")
                state["processing_status"] = "processed"
            else:
                self.logger.error(
                    f"Document processing failed: {result.error_message}")
                state["processing_status"] = "processing_failed"

            return state

        except Exception as e:
            self.logger.error(f"Error processing document: {e}")
            state["processing_status"] = "error"
            return state

    async def _search_knowledge(self, state: AgentState) -> AgentState:
        """Search the vector database for relevant information"""
        try:
            user_query = state["user_query"]
            self.logger.info(f"Searching knowledge for: {user_query}")

            # Perform vector search
            search_results = self.milvus_manager.search(
                query=user_query,
                limit=5
            )

            state["search_results"] = search_results

            # Build context from search results
            context_parts = []
            for result in search_results:
                context_parts.append(
                    f"Source: {result.file_path or 'Unknown'}")
                context_parts.append(f"Content: {result.text}")
                context_parts.append("---")

            state["context"] = "\n".join(context_parts)

            self.logger.info(f"Found {len(search_results)} relevant documents")
            return state

        except Exception as e:
            self.logger.error(f"Error searching knowledge: {e}")
            state["search_results"] = []
            state["context"] = ""
            return state

    async def _generate_response(self, state: AgentState) -> AgentState:
        """Generate final response using retrieved context"""
        try:
            user_query = state["user_query"]
            context = state.get("context", "")

            self.logger.info("Generating response")

            # Create prompt for response generation
            response_prompt = f"""
            Based on the following context, answer the user's question. If the context doesn't contain
            relevant information, say so clearly.

            Context:
            {context}

            User Question: {user_query}

            Provide a comprehensive and accurate answer based on the context. If you processed a new document,
            mention that it has been added to the knowledge base.
            """

            response = await self.llm.ainvoke([HumanMessage(content=response_prompt)])
            state["answer"] = response.content

            self.logger.info("Response generated successfully")
            return state

        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            state[
                "answer"] = f"Sorry, I encountered an error while generating the response: {str(e)}"
            return state

    async def process_query(self,
                            query: str,
                            thread_id: str = "default") -> AgentResponse:
        """
        Process a user query through the agent pipeline

        Args:
            query: User's query
            thread_id: Thread ID for conversation tracking

        Returns:
            AgentResponse with answer and sources
        """
        try:
            # Initialize state
            initial_state = {
                "messages": [HumanMessage(content=query)],
                "user_query": query,
                "search_results": [],
                "context": "",
                "answer": "",
                "file_path": None,
                "processing_status": "initializing"
            }

            # Run the graph
            config = {"configurable": {"thread_id": thread_id}}
            final_state = await self.app.ainvoke(initial_state, config=config)

            # Prepare response
            response = AgentResponse(
                answer=final_state.get("answer", "No answer generated"),
                sources=final_state.get("search_results", []),
                processing_info={
                    "processing_status": final_state.get("processing_status"),
                    "file_path": final_state.get("file_path"),
                    "context_length": len(final_state.get("context", ""))
                }
            )

            return response

        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return AgentResponse(
                answer="Sorry, I encountered an error processing your query.",
                sources=[],
                error=str(e)
            )

    def process_query_sync(self,
                           query: str,
                           thread_id: str = "default") -> AgentResponse:
        """
        Synchronous wrapper for process_query

        Args:
            query: User's query
            thread_id: Thread ID for conversation tracking

        Returns:
            AgentResponse with answer and sources
        """
        return asyncio.run(self.process_query(query, thread_id))

    def add_document(self, file_path: str) -> Dict[str, Any]:
        """
        Add a document to the knowledge base

        Args:
            file_path: Path to the document to process

        Returns:
            Processing result information
        """
        try:
            result = self.pdf_processor.process_single_pdf(file_path)
            return {
                "success": result.success,
                "file_path": result.file_path,
                "error_message": result.error_message,
                "processing_time": result.processing_time
            }
        except Exception as e:
            self.logger.error(f"Error adding document: {e}")
            return {
                "success": False,
                "file_path": file_path,
                "error_message": str(e)
            }

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector collection"""
        try:
            return self.milvus_manager.get_collection_stats()
        except Exception as e:
            self.logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}
