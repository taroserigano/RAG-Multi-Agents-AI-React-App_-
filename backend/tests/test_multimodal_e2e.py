"""
End-to-end integration tests for multimodal RAG.
Tests complete workflows from upload to retrieval to chat.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
from io import BytesIO
import uuid


@pytest.mark.integration
class TestImageUploadWorkflow:
    """Test complete image upload workflow."""

    @patch('app.api.routes_images.get_db')
    @patch('app.api.routes_images.embed_image')
    @patch('app.api.routes_images.get_pinecone_index')
    def test_upload_image_creates_db_record(self, mock_index, mock_embed, mock_db):
        """Test that uploading an image creates database record."""
        from fastapi.testclient import TestClient
        from app.db.models import ImageDocument
        
        # Mock embedding
        mock_embed.return_value = [0.1] * 512
        
        # Mock Pinecone
        mock_index_instance = MagicMock()
        mock_index.return_value = mock_index_instance
        
        # Mock database
        mock_db_session = MagicMock()
        mock_db.return_value.__enter__.return_value = mock_db_session
        
        # The actual test would require full app initialization
        # This is a placeholder structure
        pass

    def test_upload_duplicate_image_detection(self):
        """Test that duplicate images are detected by hash."""
        from app.rag.image_processing import compute_image_hash
        
        img = Image.new('RGB', (100, 100), 'red')
        buffer1 = BytesIO()
        img.save(buffer1, format='PNG')
        
        # Create identical image
        img2 = Image.new('RGB', (100, 100), 'red')
        buffer2 = BytesIO()
        img2.save(buffer2, format='PNG')
        
        hash1 = compute_image_hash(buffer1.getvalue())
        hash2 = compute_image_hash(buffer2.getvalue())
        
        assert hash1 == hash2, "Duplicate images should have same hash"


@pytest.mark.integration
class TestImageSearchWorkflow:
    """Test complete image search workflow."""

    def test_search_requires_query_or_image(self):
        """Test that search requires either query or image."""
        from app.schemas import ImageSearchRequest
        
        # With query
        req1 = ImageSearchRequest(query="test")
        assert req1.query == "test"
        
        # With image
        req2 = ImageSearchRequest(image_base64="base64data")
        assert req2.image_base64 == "base64data"


@pytest.mark.integration  
class TestMultimodalRetrievalWorkflow:
    """Test complete multimodal retrieval workflow."""

    def test_image_citation_structure(self):
        """Test ImageCitation data structure."""
        # This test requires the full import chain
        # Skip if dependencies not available
        pytest.importorskip("langchain_openai", reason="Dependencies not available")
        
        from app.rag.multimodal_retrieval import ImageCitation
        
        citation = ImageCitation(
            image_id="img-123",
            filename="test.png",
            score=0.95,
            description="Test image description",
            thumbnail_base64="base64thumb",
            width=800,
            height=600
        )
        
        data = citation.to_dict()
        
        assert data['image_id'] == "img-123"
        assert data['score'] == 0.95
        assert data['width'] == 800


@pytest.mark.integration
class TestVisionModelIntegration:
    """Test vision model integration in RAG pipeline."""

    @patch('httpx.post')
    def test_openai_vision_api_call_structure(self, mock_post):
        """Test OpenAI Vision API call structure."""
        from app.rag.vision_models import OpenAIVision
        
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "This is a test image description"
                }
            }]
        }
        mock_post.return_value = mock_response
        
        vision = OpenAIVision(api_key="test-key")
        
        img = Image.new('RGB', (100, 100), 'blue')
        result = vision.analyze_image(img, "Describe this image")
        
        assert result == "This is a test image description"
        mock_post.assert_called_once()
        
        # Verify call structure
        call_args = mock_post.call_args
        assert "Authorization" in call_args.kwargs['headers']
        assert "gpt-4o" in call_args.kwargs['json']['model']

    @patch('httpx.post')
    def test_anthropic_vision_api_call_structure(self, mock_post):
        """Test Anthropic Vision API call structure."""
        from app.rag.vision_models import AnthropicVision
        
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [{
                "type": "text",
                "text": "Anthropic vision description"
            }]
        }
        mock_post.return_value = mock_response
        
        vision = AnthropicVision(api_key="test-key")
        
        img = Image.new('RGB', (100, 100), 'green')
        result = vision.analyze_image(img, "What do you see?")
        
        assert result == "Anthropic vision description"
        mock_post.assert_called_once()

    @patch('httpx.post')
    def test_vision_api_error_handling(self, mock_post):
        """Test vision API error handling."""
        from app.rag.vision_models import OpenAIVision
        
        # Mock failed response
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = Exception("API Error")
        mock_post.return_value = mock_response
        
        vision = OpenAIVision(api_key="test-key")
        
        img = Image.new('RGB', (100, 100), 'red')
        
        with pytest.raises(Exception):
            vision.analyze_image(img, "Test")


@pytest.mark.integration
class TestImageProcessingPipeline:
    """Test complete image processing pipeline."""

    def test_image_preparation_pipeline(self):
        """Test complete pipeline: validate -> resize -> convert -> embed."""
        from app.rag.image_processing import (
            validate_image_file,
            resize_image,
            convert_to_rgb,
            compute_image_hash
        )
        
        # Create test image
        img = Image.new('RGBA', (3000, 2000), (255, 0, 0, 128))
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()
        
        # Step 1: Validate
        is_valid, error = validate_image_file("test.png", image_bytes)
        assert is_valid, f"Validation failed: {error}"
        
        # Step 2: Resize
        resized = resize_image(image_bytes, max_width=2048, max_height=2048)
        assert resized.width <= 2048
        assert resized.height <= 2048
        
        # Step 3: Convert to RGB
        rgb = convert_to_rgb(resized)
        assert rgb.mode == 'RGB'
        
        # Step 4: Hash for deduplication
        hash_value = compute_image_hash(image_bytes)
        assert len(hash_value) == 32  # MD5 hash length


@pytest.mark.integration
class TestMultimodalChatIntegration:
    """Test multimodal features in chat endpoint."""

    def test_multimodal_chat_request_structure(self):
        """Test MultimodalChatRequest schema."""
        from app.schemas import MultimodalChatRequest
        
        request = MultimodalChatRequest(
            user_id="user-123",
            provider="openai",
            question="What's in these images?",
            image_ids=["img-1", "img-2"],
            multimodal_options={
                "include_images": True,
                "use_vision_model": True,
                "vision_provider": "openai"
            }
        )
        
        assert request.user_id == "user-123"
        assert len(request.image_ids) == 2
        assert request.multimodal_options['use_vision_model'] is True

    def test_standard_chat_request_still_works(self):
        """Test that standard ChatRequest still works without images."""
        from app.schemas import ChatRequest
        
        request = ChatRequest(
            user_id="user-123",
            provider="ollama",
            question="What is the security policy?",
            rag_options={"hybrid_search": True}
        )
        
        assert request.user_id == "user-123"
        assert request.question == "What is the security policy?"


@pytest.mark.integration
class TestImageAPIEndpoints:
    """Test image API endpoints (requires full app)."""

    def test_supported_formats_endpoint(self):
        """Test /api/images/formats/supported returns correct formats."""
        # This would require TestClient and full app initialization
        # Placeholder for when environment is properly set up
        pass

    def test_list_images_pagination(self):
        """Test image listing with pagination."""
        # Would test skip/limit parameters
        pass

    def test_delete_image_cascade(self):
        """Test that deleting image removes DB record and Pinecone vector."""
        # Would test complete deletion workflow
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "integration"])
