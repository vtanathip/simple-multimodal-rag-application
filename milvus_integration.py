#!/usr/bin/env python3
"""
Example script demonstrating the MilvusManager and DoclingPDFProcessor integration

This script shows how to:
1. Process PDF documents with automatic database storage
2. Search for similar content in the database
3. Manage collections and view statistics

Note: Make sure Milvus is running before executing this script
"""

from src.MilvusManager import MilvusManager
from src.DoclingPDFProcessor import DoclingPDFProcessor
import os
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def main():
    """Main demonstration function"""
    print("=== Multimodal RAG with Milvus Demo ===\n")

    # Initialize processor with database integration
    print("1. Initializing DoclingPDFProcessor with Milvus integration...")
    try:
        processor = DoclingPDFProcessor(use_database=True)
        print("✅ Processor initialized successfully")

        # Check database connection
        db_stats = processor.get_database_stats()
        if "error" in db_stats:
            print(f"⚠️  Database connection issue: {db_stats['error']}")
            print("Make sure Milvus is running on localhost:19530")
            return
        else:
            print(
                f"✅ Database connected: {db_stats.get('collection_name', 'N/A')}")

    except Exception as e:
        print(f"❌ Failed to initialize processor: {e}")
        print("Make sure Milvus is running and accessible")
        return

    # Check if documents directory exists
    doc_dir = processor.config.get("document", {}).get("doc_dir", "documents")
    if not os.path.exists(doc_dir):
        print(f"\n2. Creating documents directory: {doc_dir}")
        os.makedirs(doc_dir, exist_ok=True)
        print(
            f"📁 Created directory. Please add PDF files to {doc_dir} and run again.")
        return

    # Process documents
    print(f"\n2. Processing PDF documents from: {doc_dir}")
    try:
        results = processor.process_directory(doc_dir)

        if not results:
            print(f"📄 No PDF files found in {doc_dir}")
            print("Add some PDF files and run again to see the full demo.")
            demonstrate_manual_insertion(processor)
        else:
            successful = sum(1 for r in results if r.success)
            print(f"📊 Processed {len(results)} files, {successful} successful")

            # Demonstrate search functionality
            demonstrate_search(processor)

    except Exception as e:
        print(f"❌ Error processing documents: {e}")

    # Display final statistics
    print("\n3. Database Statistics:")
    try:
        final_stats = processor.get_database_stats()
        if "error" not in final_stats:
            print(f"📈 Collection: {final_stats.get('collection_name', 'N/A')}")
            print("✅ Documents indexed and ready for vector search")
        else:
            print(f"⚠️  Stats error: {final_stats['error']}")
    except Exception as e:
        print(f"❌ Error getting stats: {e}")

    # Cleanup
    print("\n4. Cleaning up...")
    processor.close_database()
    print("✅ Database connection closed")


def demonstrate_manual_insertion(processor):
    """Demonstrate manual text insertion when no PDFs are available"""
    print("\n📝 Demonstrating manual text insertion...")

    # Sample documents to insert
    sample_texts = [
        "Artificial intelligence is transforming how we process and analyze documents.",
        "Machine learning models can extract meaningful information from PDF files.",
        "Vector databases enable semantic search across large document collections.",
        "Document processing pipelines are essential for modern RAG applications."
    ]

    try:
        # Insert sample texts
        success = processor.milvus_manager.insert_text_documents(
            texts=sample_texts,
            metadata_list=[{"source": "demo", "type": "sample"}
                           for _ in sample_texts],
            file_paths=["demo.txt" for _ in sample_texts]
        )

        if success:
            print(f"✅ Inserted {len(sample_texts)} sample documents")
            demonstrate_search(processor)
        else:
            print("❌ Failed to insert sample documents")

    except Exception as e:
        print(f"❌ Error inserting sample documents: {e}")


def demonstrate_search(processor):
    """Demonstrate search functionality"""
    print("\n🔍 Demonstrating search functionality...")

    search_queries = [
        "artificial intelligence",
        "document processing",
        "machine learning"
    ]

    for query in search_queries:
        try:
            print(f"\nSearching for: '{query}'")
            results = processor.search_documents(query, limit=3)

            if results:
                print(f"Found {len(results)} results:")
                for i, result in enumerate(results, 1):
                    text_preview = result['text'][:100] + \
                        "..." if len(result['text']) > 100 else result['text']
                    print(f"  {i}. Score: {result['score']:.3f}")
                    print(f"     Text: {text_preview}")
                    if result.get('file_path'):
                        print(f"     Source: {result['file_path']}")
            else:
                print("  No results found")

        except Exception as e:
            print(f"  ❌ Search error: {e}")


def check_milvus_connection():
    """Check if Milvus is accessible"""
    try:
        from pymilvus import MilvusClient
        client = MilvusClient(uri="http://localhost:19530")
        # Try a simple operation
        collections = client.list_collections()
        print("✅ Milvus is accessible")
        return True
    except Exception as e:
        print(f"❌ Cannot connect to Milvus: {e}")
        print("\nTo start Milvus, run:")
        print("  1. Download: curl -o standalone.bat https://raw.githubusercontent.com/milvus-io/milvus/refs/heads/master/scripts/standalone_embed.bat")
        print("  2. Start: standalone.bat start")
        print("  3. Web UI: http://localhost:9091/webui/")
        return False


if __name__ == "__main__":
    # Check Milvus connection first
    print("Checking Milvus connection...")
    if check_milvus_connection():
        main()
    else:
        print("\nPlease start Milvus and try again.")
