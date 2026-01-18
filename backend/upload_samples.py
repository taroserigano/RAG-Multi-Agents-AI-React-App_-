"""
Upload sample documents to the Policy RAG application.
This script uploads pre-generated policy and contract documents.
"""
import requests
import os
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"
DOCS_DIR = Path(__file__).parent / "sample_docs"

def upload_document(filepath):
    """Upload a single document to the API."""
    filename = filepath.name
    
    try:
        with open(filepath, 'rb') as f:
            files = {'file': (filename, f, 'text/plain')}
            response = requests.post(
                f"{BASE_URL}/api/docs/upload",
                files=files,
                timeout=300  # 5 minutes for large document processing
            )
            
        if response.status_code in [200, 201]:
            data = response.json()
            print(f"✅ Uploaded: {filename}")
            print(f"   ID: {data.get('doc_id') or data.get('id')}")
            return True
        else:
            print(f"❌ Failed to upload {filename}: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error uploading {filename}: {e}")
        return False

def main():
    """Upload all sample documents."""
    print("="*70)
    print("  UPLOADING SAMPLE DOCUMENTS TO POLICY RAG")
    print("="*70)
    print()
    
    # Check if backend is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ Backend API is not responding. Please start the backend first.")
            return
    except Exception as e:
        print(f"❌ Cannot connect to backend at {BASE_URL}")
        print(f"   Error: {e}")
        print("\n💡 Make sure the backend server is running:")
        print("   cd backend && python simple_server.py")
        return
    
    print(f"✅ Backend is running at {BASE_URL}\n")
    
    # Check if sample docs directory exists
    if not DOCS_DIR.exists():
        print(f"❌ Sample documents directory not found: {DOCS_DIR}")
        return
    
    # Get all text files in the directory
    doc_files = list(DOCS_DIR.glob("*.txt"))
    
    if not doc_files:
        print(f"❌ No documents found in {DOCS_DIR}")
        return
    
    print(f"📄 Found {len(doc_files)} document(s) to upload\n")
    
    # Upload each document
    success_count = 0
    for doc_file in sorted(doc_files):
        if upload_document(doc_file):
            success_count += 1
        print()
    
    # Summary
    print("="*70)
    print(f"  UPLOAD COMPLETE: {success_count}/{len(doc_files)} successful")
    print("="*70)
    print()
    
    if success_count > 0:
        print("✅ Documents are now available in the application!")
        print("\n🌐 Open your browser to:")
        print("   http://localhost:5173")
        print("\n💬 Try asking questions like:")
        print("   • What is the annual leave entitlement?")
        print("   • How many days can I work remotely?")
        print("   • What are the data retention requirements?")
        print("   • Explain the NDA confidentiality obligations")
    else:
        print("⚠️  No documents were uploaded successfully.")
        print("   Check the error messages above for details.")

if __name__ == "__main__":
    main()
