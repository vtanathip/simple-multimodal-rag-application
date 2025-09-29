#!/usr/bin/env python3
"""
Test script for Open WebUI Pipeline integration
Tests the pipeline functionality before deploying to Open WebUI
"""

from openwebui_pipeline import Pipeline
import asyncio
import json
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))


async def test_pipeline():
    """Test the Open WebUI pipeline functionality"""

    print("🧪 Testing Open WebUI Pipeline Integration")
    print("=" * 50)

    # Initialize pipeline
    pipeline = Pipeline()

    # Test startup
    print("1. Testing pipeline startup...")
    try:
        await pipeline.on_startup()
        print("   ✅ Startup successful")
    except Exception as e:
        print(f"   ❌ Startup failed: {e}")
        return

    # Test valve updates
    print("\n2. Testing valve updates...")
    try:
        pipeline.valves.MAX_SEARCH_RESULTS = 3
        await pipeline.on_valves_updated()
        print("   ✅ Valve updates successful")
    except Exception as e:
        print(f"   ❌ Valve updates failed: {e}")

    # Test help command
    print("\n3. Testing help command...")
    try:
        body = {
            "messages": [
                {"role": "user", "content": "/help"}
            ]
        }
        result = pipeline.inlet(body)
        if result["messages"][0]["role"] == "assistant":
            print("   ✅ Help command works")
            print(
                f"   📝 Response preview: {result['messages'][0]['content'][:100]}...")
        else:
            print("   ❌ Help command failed")
    except Exception as e:
        print(f"   ❌ Help command error: {e}")

    # Test stats command
    print("\n4. Testing stats command...")
    try:
        body = {
            "messages": [
                {"role": "user", "content": "/stats"}
            ]
        }
        result = pipeline.inlet(body)
        if result["messages"][0]["role"] == "assistant":
            print("   ✅ Stats command works")
            print(
                f"   📊 Response preview: {result['messages'][0]['content'][:100]}...")
        else:
            print("   ❌ Stats command failed")
    except Exception as e:
        print(f"   ❌ Stats command error: {e}")

    # Test RAG enhancement (if documents exist)
    print("\n5. Testing RAG enhancement...")
    try:
        body = {
            "messages": [
                {"role": "user", "content": "What is DocQA at Your Service?"}
            ]
        }
        original_content = body["messages"][0]["content"]
        result = pipeline.inlet(body)
        enhanced_content = result["messages"][0]["content"]

        if enhanced_content != original_content:
            print("   ✅ RAG enhancement works")
            print(f"   🔍 Enhanced query length: {len(enhanced_content)} chars")
        else:
            print("   ⚠️  No RAG enhancement (no documents or low similarity)")
    except Exception as e:
        print(f"   ❌ RAG enhancement error: {e}")

    # Test document upload command
    print("\n6. Testing document upload command...")
    try:
        # Test with non-existent file (should show error message)
        body = {
            "messages": [
                {"role": "user", "content": "/add_document test_file.pdf"}
            ]
        }
        result = pipeline.inlet(body)
        if result["messages"][0]["role"] == "assistant":
            print("   ✅ Document upload command handled")
            print(
                f"   📄 Response preview: {result['messages'][0]['content'][:100]}...")
        else:
            print("   ❌ Document upload command failed")
    except Exception as e:
        print(f"   ❌ Document upload error: {e}")

    # Test regular query (should pass through)
    print("\n7. Testing regular query passthrough...")
    try:
        body = {
            "messages": [
                {"role": "user", "content": "What is the capital of France?"}
            ]
        }
        original_body = json.dumps(body, sort_keys=True)
        result = pipeline.inlet(body)

        # For non-RAG queries, body should be minimally modified
        print("   ✅ Regular query processed")
        print(f"   📝 Query: {result['messages'][-1]['content'][:50]}...")
    except Exception as e:
        print(f"   ❌ Regular query error: {e}")

    # Test outlet processing
    print("\n8. Testing outlet processing...")
    try:
        body = {
            "messages": [
                {"role": "assistant", "content": "Test response"}
            ]
        }
        result = pipeline.outlet(body)
        print("   ✅ Outlet processing works")
    except Exception as e:
        print(f"   ❌ Outlet processing error: {e}")

    # Test shutdown
    print("\n9. Testing pipeline shutdown...")
    try:
        await pipeline.on_shutdown()
        print("   ✅ Shutdown successful")
    except Exception as e:
        print(f"   ❌ Shutdown failed: {e}")

    print("\n" + "=" * 50)
    print("🎉 Pipeline testing completed!")
    print("\nNext steps:")
    print("1. Install Open WebUI if not already installed")
    print("2. Copy openwebui_pipeline.py to Open WebUI pipelines directory")
    print("3. Configure pipeline settings in Open WebUI admin panel")
    print("4. Start using the enhanced chat interface!")


if __name__ == "__main__":
    asyncio.run(test_pipeline())
