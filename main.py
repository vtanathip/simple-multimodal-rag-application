#!/usr/bin/env python3
"""
Main entry point for the Multimodal RAG Application
Provides CLI interface for interacting with the LangGraph agent
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

from src.agent import MultimodalRAGAgent


class MultimodalRAGCLI:
    """Command-line interface for the Multimodal RAG Agent"""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the CLI with agent"""
        try:
            self.agent = MultimodalRAGAgent(config_path=config_path)
            print("✅ Multimodal RAG Agent initialized successfully!")
            print("🔗 Make sure Ollama is running on http://localhost:11434")
            print("🗃️  Make sure Milvus is running on http://localhost:19530")
            print()
        except Exception as e:
            print(f"❌ Failed to initialize agent: {e}")
            sys.exit(1)

    def interactive_mode(self):
        """Run interactive chat mode"""
        print("🤖 Interactive Multimodal RAG Agent")
        print("Commands:")
        print("  /add <file_path> - Add a document to the knowledge base")
        print("  /stats - Show collection statistics")
        print("  /help - Show this help")
        print("  /quit - Exit the application")
        print("  Or just type your question!")
        print("-" * 50)

        while True:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                if user_input == "/quit":
                    print("👋 Goodbye!")
                    break

                elif user_input == "/help":
                    self.show_help()

                elif user_input == "/stats":
                    self.show_stats()

                elif user_input.startswith("/add "):
                    file_path = user_input[5:].strip()
                    self.add_document(file_path)

                else:
                    # Process as regular query
                    self.process_query(user_input)

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    def process_query(self, query: str, thread_id: str = "cli_session"):
        """Process a user query"""
        print("🤔 Processing your query...")

        try:
            response = self.agent.process_query_sync(query, thread_id)

            print("\n🤖 Agent:")
            print(response.answer)

            if response.sources:
                print(f"\n📚 Sources ({len(response.sources)}):")
                for i, source in enumerate(response.sources, 1):
                    print(
                        f"  {i}. {source.file_path or 'Unknown'} (Score: {source.score:.3f})")
                    if len(source.text) > 100:
                        print(f"     Preview: {source.text[:100]}...")
                    else:
                        print(f"     Content: {source.text}")

            if response.processing_info:
                status = response.processing_info.get("processing_status")
                if status in ["processed", "processing_failed"]:
                    print(f"\n📄 Processing Status: {status}")

            if response.error:
                print(f"\n⚠️ Warning: {response.error}")

        except Exception as e:
            print(f"❌ Error processing query: {e}")

    def add_document(self, file_path: str):
        """Add a document to the knowledge base"""
        if not Path(file_path).exists():
            print(f"❌ File not found: {file_path}")
            return

        print(f"📄 Adding document: {file_path}")

        try:
            result = self.agent.add_document(file_path)

            if result["success"]:
                print(f"✅ Document added successfully!")
                if result.get("processing_time"):
                    print(
                        f"   Processing time: {result['processing_time']:.2f} seconds")
            else:
                print(
                    f"❌ Failed to add document: {result.get('error_message', 'Unknown error')}")

        except Exception as e:
            print(f"❌ Error adding document: {e}")

    def show_stats(self):
        """Show collection statistics"""
        try:
            stats = self.agent.get_collection_stats()

            if "error" in stats:
                print(f"❌ Error getting stats: {stats['error']}")
                return

            print("📊 Collection Statistics:")
            for key, value in stats.items():
                print(f"   {key}: {value}")

        except Exception as e:
            print(f"❌ Error getting stats: {e}")

    def show_help(self):
        """Show help information"""
        print("\n🤖 Multimodal RAG Agent Help")
        print("Commands:")
        print("  /add <file_path>  - Add a PDF document to the knowledge base")
        print("  /stats            - Show collection statistics")
        print("  /help             - Show this help")
        print("  /quit             - Exit the application")
        print()
        print("Examples:")
        print("  /add documents/report.pdf")
        print("  What is the main topic in the uploaded documents?")
        print("  Summarize the key findings from page 5")
        print()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Multimodal RAG Application with LangGraph Agent"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "--query",
        help="Single query to process (non-interactive mode)"
    )
    parser.add_argument(
        "--add-document",
        help="Add a document to the knowledge base"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show collection statistics and exit"
    )

    args = parser.parse_args()

    # Initialize CLI
    cli = MultimodalRAGCLI(config_path=args.config)

    # Handle different modes
    if args.stats:
        cli.show_stats()
    elif args.add_document:
        cli.add_document(args.add_document)
    elif args.query:
        cli.process_query(args.query)
    else:
        # Interactive mode
        cli.interactive_mode()


if __name__ == "__main__":
    main()
