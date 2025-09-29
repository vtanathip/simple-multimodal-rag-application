#!/usr/bin/env python3
"""
Deployment script for Open WebUI Pipeline
Helps deploy the Multimodal RAG pipeline to Open WebUI
"""

import os
import shutil
import subprocess
from pathlib import Path


def find_openwebui_pipelines_dir():
    """Find the Open WebUI pipelines directory"""
    possible_paths = [
        Path.home() / ".open-webui" / "pipelines",
        Path.home() / "open-webui" / "pipelines",
        Path("/app/backend/pipelines"),  # Docker default
        Path("./pipelines"),  # Local development
    ]

    for path in possible_paths:
        if path.exists():
            return path

    return None


def install_openwebui():
    """Install Open WebUI using pip"""
    print("📦 Installing Open WebUI...")
    try:
        subprocess.run(["pip", "install", "open-webui"], check=True)
        print("✅ Open WebUI installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install Open WebUI")
        return False


def deploy_pipeline():
    """Deploy the pipeline to Open WebUI"""
    print("🚀 Deploying Multimodal RAG Pipeline to Open WebUI")
    print("=" * 50)

    # Check if pipeline file exists
    pipeline_file = Path("openwebui_pipeline.py")
    if not pipeline_file.exists():
        print("❌ Pipeline file not found. Make sure you're in the project directory.")
        return False

    # Find Open WebUI pipelines directory
    pipelines_dir = find_openwebui_pipelines_dir()

    if pipelines_dir is None:
        print("⚠️  Open WebUI pipelines directory not found.")
        print("Creating local pipelines directory...")
        pipelines_dir = Path("./pipelines")
        pipelines_dir.mkdir(exist_ok=True)
        print(f"📁 Created: {pipelines_dir}")
    else:
        print(f"📁 Found pipelines directory: {pipelines_dir}")

    # Copy pipeline file
    try:
        destination = pipelines_dir / "multimodal_rag_pipeline.py"
        shutil.copy2(pipeline_file, destination)
        print(f"✅ Pipeline copied to: {destination}")
    except Exception as e:
        print(f"❌ Failed to copy pipeline: {e}")
        return False

    # Copy configuration file if it exists
    config_file = Path("openwebui_config.yaml")
    if config_file.exists():
        try:
            config_dest = pipelines_dir / "multimodal_rag_config.yaml"
            shutil.copy2(config_file, config_dest)
            print(f"✅ Configuration copied to: {config_dest}")
        except Exception as e:
            print(f"⚠️  Failed to copy configuration: {e}")

    return True


def check_dependencies():
    """Check if required services are running"""
    print("\n🔍 Checking dependencies...")

    # Check Ollama
    try:
        result = subprocess.run(
            ["curl", "-s", "http://localhost:11434/api/tags"],
            capture_output=True,
            timeout=5
        )
        if result.returncode == 0:
            print("✅ Ollama is running on localhost:11434")
        else:
            print("⚠️  Ollama not detected on localhost:11434")
    except:
        print("⚠️  Could not check Ollama status")

    # Check Milvus
    try:
        result = subprocess.run(
            ["curl", "-s", "http://localhost:19530"],
            capture_output=True,
            timeout=5
        )
        if result.returncode == 0:
            print("✅ Milvus is running on localhost:19530")
        else:
            print("⚠️  Milvus not detected on localhost:19530")
    except:
        print("⚠️  Could not check Milvus status")


def show_next_steps():
    """Show next steps for the user"""
    print("\n🎉 Deployment completed!")
    print("\n📋 Next Steps:")
    print("1. Start Open WebUI:")
    print("   open-webui serve")
    print("   (or use Docker: docker run -d -p 3000:8080 ghcr.io/open-webui/open-webui:main)")

    print("\n2. Access Open WebUI:")
    print("   Open your browser to: http://localhost:8080")

    print("\n3. Enable the pipeline:")
    print("   - Go to Admin Panel → Pipelines")
    print("   - Find 'Multimodal RAG Pipeline'")
    print("   - Toggle it ON")

    print("\n4. Configure settings:")
    print("   - Click on the pipeline settings")
    print("   - Adjust valves as needed:")
    print("     • ENABLE_RAG: true")
    print("     • MAX_SEARCH_RESULTS: 5")
    print("     • MILVUS_URI: http://localhost:19530")
    print("     • OLLAMA_BASE_URL: http://localhost:11434")

    print("\n5. Start using:")
    print("   - Upload documents: /add_document path/to/file.pdf")
    print("   - Ask questions about your documents")
    print("   - Get help: /help")
    print("   - Check status: /stats")

    print("\n🔧 Troubleshooting:")
    print("   - Ensure Milvus is running: cd db/win32 && ./standalone.bat")
    print("   - Ensure Ollama is running: ollama serve")
    print("   - Check Open WebUI logs for errors")


def main():
    """Main deployment function"""
    print("🤖 Multimodal RAG Pipeline Deployment")
    print("=" * 50)

    # Check if we're in the right directory
    if not Path("openwebui_pipeline.py").exists():
        print("❌ Please run this script from the project root directory")
        print("   (where openwebui_pipeline.py is located)")
        return

    # Deploy pipeline
    if not deploy_pipeline():
        print("❌ Deployment failed!")
        return

    # Check dependencies
    check_dependencies()

    # Show next steps
    show_next_steps()


if __name__ == "__main__":
    main()
