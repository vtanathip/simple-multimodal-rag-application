"""
Example client for interacting with the Multimodal RAG API
"""

import httpx
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional


class RAGAPIClient:
    """Client for interacting with the Multimodal RAG API"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the API client

        Args:
            base_url: Base URL of the RAG API
        """
        self.base_url = base_url.rstrip('/')
        self.client = httpx.AsyncClient()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    async def health_check(self) -> Dict[str, Any]:
        """Check API health status"""
        response = await self.client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    async def query(self, query: str, thread_id: str = "default") -> Dict[str, Any]:
        """
        Send a query to the RAG agent

        Args:
            query: User query
            thread_id: Thread ID for conversation tracking

        Returns:
            Query response with answer and sources
        """
        payload = {
            "query": query,
            "thread_id": thread_id
        }

        response = await self.client.post(f"{self.base_url}/query", json=payload)
        response.raise_for_status()
        return response.json()

    async def upload_document(self, file_path: str, process_immediately: bool = True) -> Dict[str, Any]:
        """
        Upload a document to the RAG system

        Args:
            file_path: Path to the PDF document
            process_immediately: Whether to process immediately

        Returns:
            Upload response with status
        """
        file_obj = Path(file_path)

        if not file_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_obj, "rb") as file:
            files = {"file": (file_obj.name, file, "application/pdf")}
            data = {"process_immediately": str(process_immediately).lower()}

            response = await self.client.post(
                f"{self.base_url}/upload",
                files=files,
                data=data
            )
            response.raise_for_status()
            return response.json()

    async def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        Process an existing document by file path

        Args:
            file_path: Path to the document to process

        Returns:
            Processing response with status
        """
        data = {"file_path": file_path}

        response = await self.client.post(f"{self.base_url}/process-document", data=data)
        response.raise_for_status()
        return response.json()

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        response = await self.client.get(f"{self.base_url}/collection/stats")
        response.raise_for_status()
        return response.json()

    async def clear_collection(self) -> Dict[str, Any]:
        """Clear the document collection"""
        response = await self.client.delete(f"{self.base_url}/collection/clear")
        response.raise_for_status()
        return response.json()


# Example usage functions
async def example_usage():
    """Example usage of the RAG API client"""

    async with RAGAPIClient() as client:
        try:
            # Health check
            print("=== Health Check ===")
            health = await client.health_check()
            print(f"API Status: {health['status']}")
            print(f"Components: {health['components']}")

            # Get collection stats
            print("\n=== Collection Stats ===")
            stats = await client.get_collection_stats()
            print(f"Collection Stats: {stats}")

            # Example query
            print("\n=== Example Query ===")
            query_response = await client.query(
                "What is machine learning?",
                thread_id="example_session"
            )
            print(f"Answer: {query_response['answer']}")
            print(f"Sources: {len(query_response['sources'])} documents found")

            # Example document upload (if you have a PDF file)
            # print("\n=== Document Upload ===")
            # upload_response = await client.upload_document("path/to/your/document.pdf")
            # print(f"Upload Status: {upload_response['success']}")
            # print(f"Message: {upload_response['message']}")

        except httpx.HTTPStatusError as e:
            print(f"HTTP Error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            print(f"Error: {e}")


def sync_example_usage():
    """Synchronous wrapper for the example usage"""
    asyncio.run(example_usage())


if __name__ == "__main__":
    sync_example_usage()
