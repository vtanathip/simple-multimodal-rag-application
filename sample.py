from src.DoclingPDFProcessor import DoclingPDFProcessor
import sys
from pathlib import Path


def main():
    """
    Main entry point for the simple-multimodal-rag-application
    Uses DoclingPDFProcessor to process PDF files
    """
    print("🚀 Simple Multimodal RAG Application")
    print("Using DoclingPDFProcessor for PDF processing")
    print("=" * 50)

    # Initialize the PDF processor
    processor = DoclingPDFProcessor()

    # Get document directory from config
    doc_config = processor.config.get("document", {})
    doc_dir = doc_config.get("doc_dir", "documents")

    print(f"Processing PDFs from: {doc_dir}")

    # Process all PDFs in the directory
    results = processor.process_directory(doc_dir)

    if results:
        # Save results
        processor.save_results(results)

        # Show summary
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        print(f"\n✅ Successfully processed: {len(successful)} files")
        print(f"❌ Failed to process: {len(failed)} files")

        if successful:
            print("\nProcessed files:")
            for result in successful:
                print(
                    f"  • {Path(result.file_path).name} ({result.processing_time:.2f}s)")

        print(f"\n📄 Markdown files saved to 'output' directory")
    else:
        print(f"\n📭 No PDF files found in '{doc_dir}'")
        print("💡 To get started:")
        print(f"   1. Place PDF files in the '{doc_dir}' directory")
        print("   2. Run this script again")

    print("\n🎉 Application completed!")


if __name__ == "__main__":
    main()
