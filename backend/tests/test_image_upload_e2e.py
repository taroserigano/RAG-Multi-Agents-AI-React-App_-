"""
Comprehensive E2E tests for image upload functionality.
Tests all aspects of the multimodal RAG image upload pipeline.
"""
import pytest
import requests
import io
import time
import json
from PIL import Image

BASE_URL = "http://localhost:8000"


def wait_for_server(max_retries=30, delay=1):
    """Wait for the server to be ready."""
    for i in range(max_retries):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                print(f"Server ready after {i+1} attempts")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(delay)
    return False


def create_test_image(color='red', size=(100, 100), format='PNG'):
    """Create a test image with specified color and size."""
    img = Image.new('RGB', size, color=color)
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    buffer.seek(0)
    return buffer


class TestImageUploadE2E:
    """End-to-end tests for image upload."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Ensure server is running before tests."""
        if not wait_for_server(max_retries=10, delay=1):
            pytest.skip("Server not available")
    
    def test_upload_png_image_no_description(self):
        """Test uploading a PNG image without auto-description."""
        # Use unique color to avoid duplicate detection
        import random
        r, g, b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
        buffer = create_test_image(color=(r, g, b), format='PNG')
        
        files = {'file': ('test_unique.png', buffer, 'image/png')}
        data = {'generate_description': 'false', 'vision_provider': 'openai'}
        
        response = requests.post(
            f"{BASE_URL}/api/images/upload",
            files=files,
            data=data
        )
        
        print(f"Response status: {response.status_code}")
        print(f"Response body: {response.text}")
        
        assert response.status_code == 201, f"Expected 201, got {response.status_code}: {response.text}"
        
        resp_data = response.json()
        assert 'image_id' in resp_data
        # Filename could be original or from duplicate
        assert 'filename' in resp_data
        assert 'message' in resp_data
        
        return resp_data['image_id']
    
    def test_upload_jpg_image(self):
        """Test uploading a JPEG image."""
        buffer = create_test_image(color='green', format='JPEG')
        
        files = {'file': ('test_green.jpg', buffer, 'image/jpeg')}
        data = {'generate_description': 'false', 'vision_provider': 'openai'}
        
        response = requests.post(
            f"{BASE_URL}/api/images/upload",
            files=files,
            data=data
        )
        
        print(f"Response status: {response.status_code}")
        print(f"Response body: {response.text}")
        
        assert response.status_code == 201, f"Expected 201, got {response.status_code}: {response.text}"
        
        data = response.json()
        assert 'image_id' in data
        assert data['filename'] == 'test_green.jpg'
    
    def test_upload_with_generate_description_true(self):
        """Test uploading with generate_description=true (string form)."""
        buffer = create_test_image(color='yellow', format='PNG')
        
        files = {'file': ('test_yellow.png', buffer, 'image/png')}
        # Frontend sends 'true' as string
        data = {'generate_description': 'true', 'vision_provider': 'openai'}
        
        response = requests.post(
            f"{BASE_URL}/api/images/upload",
            files=files,
            data=data
        )
        
        print(f"Response status: {response.status_code}")
        print(f"Response body: {response.text}")
        
        # This might fail if vision API isn't configured, but shouldn't be 500
        assert response.status_code in [201, 500], f"Unexpected status: {response.status_code}"
        if response.status_code == 500:
            # Check if it's a vision API error, not a parameter parsing error
            error_data = response.json()
            assert 'vision' in str(error_data).lower() or 'api' in str(error_data).lower() or 'openai' in str(error_data).lower(), \
                f"Unexpected 500 error: {error_data}"
    
    def test_upload_duplicate_image(self):
        """Test that uploading the same image returns the existing entry."""
        buffer1 = create_test_image(color='purple', format='PNG')
        buffer2 = create_test_image(color='purple', format='PNG')  # Same color = same hash
        
        files1 = {'file': ('test_purple1.png', buffer1, 'image/png')}
        data = {'generate_description': 'false', 'vision_provider': 'openai'}
        
        response1 = requests.post(
            f"{BASE_URL}/api/images/upload",
            files=files1,
            data=data
        )
        
        assert response1.status_code == 201, f"First upload failed: {response1.text}"
        first_id = response1.json()['image_id']
        
        files2 = {'file': ('test_purple2.png', buffer2, 'image/png')}
        response2 = requests.post(
            f"{BASE_URL}/api/images/upload",
            files=files2,
            data=data
        )
        
        assert response2.status_code == 201
        second_id = response2.json()['image_id']
        
        # Should return the same image ID (duplicate detection)
        assert first_id == second_id, "Duplicate image should return same ID"
        assert "duplicate" in response2.json()['message'].lower()
    
    def test_upload_invalid_file_type(self):
        """Test uploading a non-image file."""
        buffer = io.BytesIO(b"This is not an image file")
        
        files = {'file': ('test.txt', buffer, 'text/plain')}
        data = {'generate_description': 'false', 'vision_provider': 'openai'}
        
        response = requests.post(
            f"{BASE_URL}/api/images/upload",
            files=files,
            data=data
        )
        
        print(f"Response status: {response.status_code}")
        print(f"Response body: {response.text}")
        
        assert response.status_code == 400, f"Expected 400 for invalid file, got {response.status_code}"
    
    def test_list_images(self):
        """Test listing uploaded images."""
        response = requests.get(f"{BASE_URL}/api/images/")
        
        print(f"Response status: {response.status_code}")
        print(f"Response body: {response.text[:500]}")
        
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
    
    def test_get_image_by_id(self):
        """Test getting a specific image by ID."""
        # First upload an image
        buffer = create_test_image(color='orange', format='PNG')
        
        files = {'file': ('test_orange.png', buffer, 'image/png')}
        data = {'generate_description': 'false', 'vision_provider': 'openai'}
        
        upload_response = requests.post(
            f"{BASE_URL}/api/images/upload",
            files=files,
            data=data
        )
        
        if upload_response.status_code != 201:
            # Might be duplicate
            assert upload_response.status_code == 201 or 'duplicate' in upload_response.text.lower()
        
        image_id = upload_response.json()['image_id']
        
        # Then get it by ID
        response = requests.get(f"{BASE_URL}/api/images/{image_id}")
        
        print(f"Response status: {response.status_code}")
        print(f"Response body: {response.text[:500]}")
        
        assert response.status_code == 200
        
        data = response.json()
        assert data['id'] == image_id
        assert data['filename'] == 'test_orange.png'
    
    def test_get_nonexistent_image(self):
        """Test getting an image that doesn't exist."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        
        response = requests.get(f"{BASE_URL}/api/images/{fake_id}")
        
        assert response.status_code == 404


class TestImageSearchE2E:
    """End-to-end tests for image search."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Ensure server is running before tests."""
        if not wait_for_server(max_retries=10, delay=1):
            pytest.skip("Server not available")
    
    def test_search_images_by_text(self):
        """Test searching images by text query."""
        response = requests.post(
            f"{BASE_URL}/api/images/search",
            json={"query": "blue color", "top_k": 5}
        )
        
        print(f"Response status: {response.status_code}")
        print(f"Response body: {response.text[:500]}")
        
        assert response.status_code == 200
        
        data = response.json()
        assert 'images' in data
        assert isinstance(data['images'], list)


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_endpoint(self):
        """Test that health endpoint returns correct status."""
        response = requests.get(f"{BASE_URL}/health")
        
        print(f"Response status: {response.status_code}")
        print(f"Response body: {response.text}")
        
        assert response.status_code == 200
        
        data = response.json()
        assert data.get('status') == 'healthy'


def run_quick_test():
    """Run a quick manual test without pytest."""
    print("=" * 60)
    print("QUICK IMAGE UPLOAD TEST")
    print("=" * 60)
    
    if not wait_for_server(max_retries=5, delay=1):
        print("ERROR: Server not available at http://localhost:8000")
        return False
    
    print("\n1. Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"   Status: {response.status_code}")
    if response.status_code != 200:
        print("   FAILED: Health check failed")
        return False
    print("   PASSED")
    
    print("\n2. Testing image upload (PNG, no description)...")
    buffer = create_test_image(color='cyan', format='PNG')
    files = {'file': ('test_cyan.png', buffer, 'image/png')}
    data = {'generate_description': 'false', 'vision_provider': 'openai'}
    
    response = requests.post(f"{BASE_URL}/api/images/upload", files=files, data=data)
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.text}")
    
    if response.status_code != 201:
        print("   FAILED: Image upload failed")
        return False
    
    image_id = response.json()['image_id']
    print(f"   PASSED: Image ID = {image_id}")
    
    print("\n3. Testing list images...")
    response = requests.get(f"{BASE_URL}/api/images/")
    print(f"   Status: {response.status_code}")
    if response.status_code != 200:
        print("   FAILED: List images failed")
        return False
    print(f"   PASSED: Found {len(response.json())} images")
    
    print("\n4. Testing get image by ID...")
    response = requests.get(f"{BASE_URL}/api/images/{image_id}")
    print(f"   Status: {response.status_code}")
    if response.status_code != 200:
        print("   FAILED: Get image by ID failed")
        return False
    print("   PASSED")
    
    print("\n5. Testing image search...")
    response = requests.post(
        f"{BASE_URL}/api/images/search",
        json={"query": "test image", "top_k": 5}
    )
    print(f"   Status: {response.status_code}")
    if response.status_code != 200:
        print("   FAILED: Image search failed")
        return False
    print(f"   PASSED: Found {len(response.json().get('results', []))} results")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        success = run_quick_test()
        sys.exit(0 if success else 1)
    else:
        # Run with pytest
        pytest.main([__file__, "-v", "--tb=short"])
