"""
Milvus Database Manager for Multimodal RAG Application
Manages vector database operations including collection creation, data insertion, and retrieval
"""

import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import uuid

import yaml
from pymilvus import MilvusClient, model, DataType, FieldSchema, CollectionSchema
from pymilvus.exceptions import MilvusException


@dataclass
class VectorDocument:
    """Data class to hold vector document information"""
    id: str
    text: str
    vector: List[float]
    metadata: Optional[Dict[str, Any]] = None
    file_path: Optional[str] = None
    page_number: Optional[int] = None
    chunk_index: Optional[int] = None


@dataclass
class SearchResult:
    """Data class to hold search results"""
    id: str
    text: str
    score: float
    metadata: Optional[Dict[str, Any]] = None
    file_path: Optional[str] = None
    page_number: Optional[int] = None


class MilvusManager:
    """
    Manager class for Milvus vector database operations
    Handles collection management, document insertion, and similarity search
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the Milvus manager with configuration

        Args:
            config_path: Path to the configuration YAML file
        """
        self.logger = self._setup_logging()
        self.config = self._load_config(config_path)
        self.client = self._setup_client()
        self.embedding_fn = self._setup_embedding_function()
        self.collection_name = self.config.get("database", {}).get(
            "collection_name", "rag_collection")
        self.vector_dimension = 768  # Default dimension for text-embedding models

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
            "database": {
                "uri": "http://localhost:19530",
                "name": "rag_multimodal",
                "collection_name": "rag_collection",
                "namespace": "default"
            }
        }

    def _setup_client(self) -> MilvusClient:
        """Setup Milvus client connection"""
        try:
            db_config = self.config.get("database", {})
            uri = db_config.get("uri", "http://localhost:19530")

            client = MilvusClient(uri=uri)
            self.logger.info(f"Successfully connected to Milvus at {uri}")
            return client

        except Exception as e:
            self.logger.error(f"Error connecting to Milvus: {e}")
            raise

    def _setup_embedding_function(self):
        """Setup embedding function for text vectorization"""
        try:
            # Use default embedding function from pymilvus
            embedding_fn = model.DefaultEmbeddingFunction()
            self.vector_dimension = embedding_fn.dim
            self.logger.info(
                f"Embedding function initialized with dimension {self.vector_dimension}")
            return embedding_fn
        except Exception as e:
            self.logger.error(f"Error setting up embedding function: {e}")
            raise

    def create_collection(self, collection_name: Optional[str] = None, drop_existing: bool = False) -> bool:
        """
        Create a new collection in Milvus with explicit schema for string IDs

        Args:
            collection_name: Name of the collection to create
            drop_existing: Whether to drop existing collection with same name

        Returns:
            bool: True if collection created successfully
        """
        collection_name = collection_name or self.collection_name

        try:
            # Check if collection exists
            if self.client.has_collection(collection_name=collection_name):
                if drop_existing:
                    self.logger.info(
                        f"Dropping existing collection: {collection_name}")
                    self.client.drop_collection(
                        collection_name=collection_name)
                else:
                    self.logger.info(
                        f"Collection {collection_name} already exists")
                    return True

            # Define schema with string ID field
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR,
                            is_primary=True, max_length=100),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR,
                            dim=self.vector_dimension),
                FieldSchema(name="text", dtype=DataType.VARCHAR,
                            max_length=65535),
                FieldSchema(name="file_path",
                            dtype=DataType.VARCHAR, max_length=1000),
                FieldSchema(name="page_number", dtype=DataType.INT64),
                FieldSchema(name="chunk_index", dtype=DataType.INT64)
            ]

            schema = CollectionSchema(
                fields=fields,
                description="RAG collection with string IDs",
                enable_dynamic_field=True
            )

            # Create collection with explicit schema
            self.client.create_collection(
                collection_name=collection_name,
                schema=schema,
                metric_type="COSINE",  # Use cosine similarity
                consistency_level="Strong"
            )

            self.logger.info(
                f"Collection {collection_name} created successfully with string ID schema")

            # Create index on vector field
            self.create_index(collection_name)
            return True

        except MilvusException as e:
            self.logger.error(f"Milvus error creating collection: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error creating collection: {e}")
            return False

    def create_index(self, collection_name: Optional[str] = None) -> bool:
        """
        Create an index on the vector field for the collection

        Args:
            collection_name: Name of the collection to create index for

        Returns:
            bool: True if index creation successful
        """
        collection_name = collection_name or self.collection_name

        try:
            # Use the MilvusClient's prepare_index_params method for proper format
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                index_type="FLAT",  # Simple flat index for small datasets
                metric_type="COSINE"
            )

            # Create index on vector field
            self.client.create_index(
                collection_name=collection_name,
                index_params=index_params
            )

            self.logger.info(
                f"Index created successfully for collection {collection_name}")
            return True

        except MilvusException as e:
            self.logger.error(f"Milvus error creating index: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error creating index: {e}")
            return False

    def insert_documents(self, documents: List[VectorDocument], collection_name: Optional[str] = None) -> bool:
        """
        Insert documents into the collection

        Args:
            documents: List of VectorDocument objects to insert
            collection_name: Name of the collection to insert into

        Returns:
            bool: True if insertion successful
        """
        collection_name = collection_name or self.collection_name

        try:
            # Ensure collection exists
            if not self.client.has_collection(collection_name=collection_name):
                self.logger.warning(
                    f"Collection {collection_name} does not exist, creating it")
                if not self.create_collection(collection_name):
                    return False

            # Prepare data for insertion
            data = []
            for doc in documents:
                entry = {
                    "id": doc.id,
                    "vector": doc.vector,
                    "text": doc.text,
                    "file_path": doc.file_path or "",
                    "page_number": doc.page_number or 0,
                    "chunk_index": doc.chunk_index or 0
                }

                # Add metadata fields if present
                if doc.metadata:
                    for key, value in doc.metadata.items():
                        # Convert to string to ensure compatibility
                        entry[f"metadata_{key}"] = str(value)

                data.append(entry)

            # Insert data
            result = self.client.insert(
                collection_name=collection_name, data=data)

            self.logger.info(
                f"Successfully inserted {len(documents)} documents into {collection_name}")
            return True

        except MilvusException as e:
            self.logger.error(f"Milvus error inserting documents: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error inserting documents: {e}")
            return False

    def insert_text_documents(self, texts: List[str], metadata_list: Optional[List[Dict[str, Any]]] = None,
                              file_paths: Optional[List[str]] = None, collection_name: Optional[str] = None) -> bool:
        """
        Insert text documents by automatically generating embeddings

        Args:
            texts: List of text strings to insert
            metadata_list: Optional list of metadata dictionaries
            file_paths: Optional list of file paths
            collection_name: Name of the collection to insert into

        Returns:
            bool: True if insertion successful
        """
        try:
            # Generate embeddings
            vectors = self.embedding_fn.encode_documents(texts)

            # Create VectorDocument objects
            documents = []
            for i, text in enumerate(texts):
                doc_id = str(uuid.uuid4())
                metadata = metadata_list[i] if metadata_list and i < len(
                    metadata_list) else None
                file_path = file_paths[i] if file_paths and i < len(
                    file_paths) else None

                doc = VectorDocument(
                    id=doc_id,
                    text=text,
                    vector=vectors[i].tolist(),
                    metadata=metadata,
                    file_path=file_path
                )
                documents.append(doc)

            return self.insert_documents(documents, collection_name)

        except Exception as e:
            self.logger.error(f"Error inserting text documents: {e}")
            return False

    def load_collection(self, collection_name: Optional[str] = None) -> bool:
        """
        Load a collection into memory for searching

        Args:
            collection_name: Name of the collection to load

        Returns:
            bool: True if loading successful
        """
        collection_name = collection_name or self.collection_name

        try:
            # Check if collection exists first
            if not self.client.has_collection(collection_name=collection_name):
                self.logger.error(
                    f"Collection {collection_name} does not exist")
                return False

            # Load the collection
            self.client.load_collection(collection_name=collection_name)
            self.logger.info(
                f"Collection {collection_name} loaded successfully")
            return True

        except MilvusException as e:
            self.logger.error(f"Milvus error loading collection: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error loading collection: {e}")
            return False

    def search(self, query: str, limit: int = 10, collection_name: Optional[str] = None) -> List[SearchResult]:
        """
        Search for similar documents using text query

        Args:
            query: Text query to search for
            limit: Maximum number of results to return
            collection_name: Name of the collection to search in

        Returns:
            List of SearchResult objects
        """
        collection_name = collection_name or self.collection_name

        try:
            # Ensure collection is loaded
            self.load_collection(collection_name)

            # Generate query vector
            query_vector = self.embedding_fn.encode_queries([query])[0]

            # Perform search
            results = self.client.search(
                collection_name=collection_name,
                data=[query_vector],
                limit=limit,
                output_fields=["text", "file_path",
                               "page_number", "chunk_index"]
            )

            # Convert to SearchResult objects
            search_results = []
            for result in results[0]:  # results is a list of lists
                search_result = SearchResult(
                    id=str(result.get("id", "")),
                    text=result.get("text", ""),
                    score=float(result.get("distance", 0.0)),
                    file_path=result.get("file_path"),
                    page_number=result.get("page_number")
                )
                search_results.append(search_result)

            self.logger.info(
                f"Search completed, found {len(search_results)} results")
            return search_results

        except MilvusException as e:
            self.logger.error(f"Milvus error during search: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Error during search: {e}")
            return []

    def delete_collection(self, collection_name: Optional[str] = None) -> bool:
        """
        Delete a collection

        Args:
            collection_name: Name of the collection to delete

        Returns:
            bool: True if deletion successful
        """
        collection_name = collection_name or self.collection_name

        try:
            if self.client.has_collection(collection_name=collection_name):
                self.client.drop_collection(collection_name=collection_name)
                self.logger.info(
                    f"Collection {collection_name} deleted successfully")
                return True
            else:
                self.logger.warning(
                    f"Collection {collection_name} does not exist")
                return False

        except MilvusException as e:
            self.logger.error(f"Milvus error deleting collection: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error deleting collection: {e}")
            return False

    def get_collection_stats(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get collection statistics

        Args:
            collection_name: Name of the collection

        Returns:
            Dictionary with collection statistics
        """
        collection_name = collection_name or self.collection_name

        try:
            if not self.client.has_collection(collection_name=collection_name):
                return {"error": "Collection does not exist"}

            # Get collection info
            stats = self.client.describe_collection(
                collection_name=collection_name)

            return {
                "collection_name": collection_name,
                "exists": True,
                "stats": stats
            }

        except Exception as e:
            self.logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}

    def close(self):
        """Close the Milvus client connection"""
        try:
            if hasattr(self.client, 'close'):
                self.client.close()
            self.logger.info("Milvus client connection closed")
        except Exception as e:
            self.logger.error(f"Error closing Milvus client: {e}")
