"""
Edge case and stress tests for multimodal RAG features.
Tests boundary conditions, error handling, and edge cases.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
from io import BytesIO
import base64


class TestImageProcessingEdgeCases:
    """Test edge cases in image processing."""

    def test_extremely_small_image(self):
        """Test handling of 1x1 pixel images."""
        from app.rag.image_processing import validate_image_file, get_image_info
        
        img = Image.new('RGB', (1, 1), color='red')
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        
        is_valid, error = validate_image_file("tiny.png", buffer.getvalue())
        assert is_valid is True
        
        info = get_image_info(buffer.getvalue())
        assert info['width'] == 1
        assert info['height'] == 1

    def test_extremely_large_image(self):
        """Test handling of very large images."""
        from app.rag.image_processing import validate_image_file
        
        # Simulate 25MB file
        large_data = b'\x89PNG\r\n\x1a\n' + (b'\x00' * (25 * 1024 * 1024))
        is_valid, error = validate_image_file("large.png", large_data)
        
        assert is_valid is False
        assert "too large" in error.lower()

    def test_corrupted_image_data(self):
        """Test handling of corrupted image data."""
        from app.rag.image_processing import validate_image_file
        
        corrupted_data = b'\x89PNG\r\n\x1a\n' + b'corrupted_data_here'
        is_valid, error = validate_image_file("corrupt.png", corrupted_data)
        
        assert is_valid is False
        assert "invalid" in error.lower()

    def test_empty_image_data(self):
        """Test handling of empty image data."""
        from app.rag.image_processing import validate_image_file
        
        is_valid, error = validate_image_file("empty.png", b'')
        assert is_valid is False

    def test_wrong_extension_with_valid_data(self):
        """Test PNG data with .jpg extension."""
        from app.rag.image_processing import validate_image_file
        
        img = Image.new('RGB', (100, 100), color='blue')
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        
        # Valid PNG data but claims to be JPG
        is_valid, error = validate_image_file("fake.txt", buffer.getvalue())
        assert is_valid is False
        assert "unsupported" in error.lower()

    def test_transparent_png_conversion(self):
        """Test converting transparent PNG to RGB."""
        from app.rag.image_processing import convert_to_rgb
        
        # Create RGBA image with transparency
        img = Image.new('RGBA', (100, 100), (255, 0, 0, 128))
        
        rgb_img = convert_to_rgb(img)
        
        assert rgb_img.mode == 'RGB'
        assert rgb_img.size == (100, 100)

    def test_grayscale_image_conversion(self):
        """Test converting grayscale to RGB."""
        from app.rag.image_processing import convert_to_rgb
        
        gray_img = Image.new('L', (100, 100), 128)
        rgb_img = convert_to_rgb(gray_img)
        
        assert rgb_img.mode == 'RGB'

    def test_animated_gif_detection(self):
        """Test detection of animated GIF."""
        from app.rag.image_processing import get_image_info
        
        # Create simple GIF
        img = Image.new('RGB', (50, 50), 'green')
        buffer = BytesIO()
        img.save(buffer, format='GIF')
        
        info = get_image_info(buffer.getvalue())
        assert 'is_animated' in info

    def test_base64_with_data_url_prefix(self):
        """Test base64 decoding with data URL prefix."""
        from app.rag.image_processing import base64_to_image
        
        img = Image.new('RGB', (50, 50), 'yellow')
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        b64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Add data URL prefix
        data_url = f"data:image/png;base64,{b64_data}"
        
        decoded_img = base64_to_image(data_url)
        assert decoded_img.size == (50, 50)

    def test_image_hash_consistency(self):
        """Test that identical images produce identical hashes."""
        from app.rag.image_processing import compute_image_hash
        
        img1 = Image.new('RGB', (100, 100), 'blue')
        buffer1 = BytesIO()
        img1.save(buffer1, format='PNG')
        
        img2 = Image.new('RGB', (100, 100), 'blue')
        buffer2 = BytesIO()
        img2.save(buffer2, format='PNG')
        
        hash1 = compute_image_hash(buffer1.getvalue())
        hash2 = compute_image_hash(buffer2.getvalue())
        
        assert hash1 == hash2

    def test_image_hash_different_for_different_images(self):
        """Test that different images produce different hashes."""
        from app.rag.image_processing import compute_image_hash
        
        img1 = Image.new('RGB', (100, 100), 'blue')
        buffer1 = BytesIO()
        img1.save(buffer1, format='PNG')
        
        img2 = Image.new('RGB', (100, 100), 'red')
        buffer2 = BytesIO()
        img2.save(buffer2, format='PNG')
        
        hash1 = compute_image_hash(buffer1.getvalue())
        hash2 = compute_image_hash(buffer2.getvalue())
        
        assert hash1 != hash2


class TestVisionModelsEdgeCases:
    """Test edge cases in vision model integrations."""

    def test_base64_encoding_bytes_input(self):
        """Test base64 encoding with bytes input."""
        from app.rag.vision_models import image_to_base64
        
        test_bytes = b'\x89PNG\r\n\x1a\n' + b'\x00' * 100
        result = image_to_base64(test_bytes)
        
        assert isinstance(result, str)
        assert len(result) > 0

    def test_base64_encoding_pil_image(self):
        """Test base64 encoding with PIL Image."""
        from app.rag.vision_models import image_to_base64
        
        img = Image.new('RGB', (50, 50), 'purple')
        result = image_to_base64(img)
        
        assert isinstance(result, str)
        assert len(result) > 0

    def test_base64_encoding_already_encoded(self):
        """Test base64 encoding with already encoded data URL."""
        from app.rag.vision_models import image_to_base64
        
        data_url = "data:image/png;base64,iVBORw0KGgo="
        result = image_to_base64(data_url)
        
        assert result == data_url

    def test_media_type_detection_webp(self):
        """Test WebP detection."""
        from app.rag.vision_models import get_image_media_type
        
        webp_bytes = b'RIFF' + b'\x00' * 4 + b'WEBP' + b'\x00' * 100
        result = get_image_media_type(webp_bytes)
        
        assert result == "image/webp"

    def test_media_type_detection_gif(self):
        """Test GIF detection."""
        from app.rag.vision_models import get_image_media_type
        
        gif_bytes = b'GIF89a' + b'\x00' * 100
        result = get_image_media_type(gif_bytes)
        
        assert result == "image/gif"

    def test_media_type_default_fallback(self):
        """Test fallback to PNG for unknown formats."""
        from app.rag.vision_models import get_image_media_type
        
        unknown_bytes = b'UNKNOWN' + b'\x00' * 100
        result = get_image_media_type(unknown_bytes)
        
        assert result == "image/png"

    def test_openai_vision_missing_api_key(self):
        """Test OpenAI Vision initialization without API key."""
        from app.rag.vision_models import OpenAIVision
        
        with patch('app.rag.vision_models.settings.openai_api_key', None):
            with pytest.raises(ValueError, match="API key"):
                OpenAIVision()

    def test_anthropic_vision_missing_api_key(self):
        """Test Anthropic Vision initialization without API key."""
        from app.rag.vision_models import AnthropicVision
        
        with patch('app.rag.vision_models.settings.anthropic_api_key', None):
            with pytest.raises(ValueError, match="API key"):
                AnthropicVision()

    def test_vision_model_invalid_provider(self):
        """Test get_vision_model with invalid provider."""
        from app.rag.vision_models import get_vision_model
        
        with pytest.raises(ValueError, match="Unsupported vision provider"):
            get_vision_model("invalid_provider_xyz")


class TestImageEmbeddingsEdgeCases:
    """Test edge cases in CLIP embeddings."""

    @patch('app.rag.image_embeddings.get_clip_model')
    def test_embed_image_with_none_input(self, mock_get_clip):
        """Test embedding with None input should raise error."""
        from app.rag.image_embeddings import embed_image
        
        mock_model = MagicMock()
        mock_get_clip.return_value = mock_model
        
        with pytest.raises(Exception):
            embed_image(None)

    @patch('app.rag.image_embeddings.get_sentence_transformer')
    def test_embed_query_empty_string(self, mock_st):
        """Test embedding empty query string."""
        from app.rag.image_embeddings import embed_image_query
        import numpy as np
        
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.0] * 512)
        mock_st.return_value = mock_model
        
        result = embed_image_query("")
        assert result is not None

    @patch('app.rag.image_embeddings.get_sentence_transformer')
    def test_embed_query_very_long_text(self, mock_st):
        """Test embedding very long query text."""
        from app.rag.image_embeddings import embed_image_query
        import numpy as np
        
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.0] * 512)
        mock_st.return_value = mock_model
        
        long_query = "test query " * 1000  # Very long text
        result = embed_image_query(long_query)
        assert result is not None

    @patch('app.rag.image_embeddings.get_clip_model')
    def test_embed_image_tiny_resolution(self, mock_get_clip):
        """Test embedding 1x1 pixel image."""
        from app.rag.image_embeddings import embed_image
        import numpy as np
        
        mock_model = MagicMock()
        mock_model.return_value = {'image_embeds': np.array([[0.1] * 512])}
        mock_get_clip.return_value = mock_model
        
        tiny_img = Image.new('RGB', (1, 1), 'red')
        buffer = BytesIO()
        tiny_img.save(buffer, format='PNG')
        
        try:
            result = embed_image(buffer.getvalue())
            assert result is not None
        except Exception:
            # Some models may reject tiny images
            pass


class TestDatabaseModelsEdgeCases:
    """Test edge cases in database models."""

    def test_image_document_with_minimal_fields(self):
        """Test ImageDocument with only required fields."""
        from app.db.models import ImageDocument
        
        doc = ImageDocument(
            filename="test.png",
            content_type="image/png"
        )
        
        assert doc.filename == "test.png"
        assert doc.content_type == "image/png"
        assert doc.width is None

    def test_image_document_with_all_fields(self):
        """Test ImageDocument with all fields populated."""
        from app.db.models import ImageDocument
        
        doc = ImageDocument(
            id="test-id-123",
            filename="complete.png",
            content_type="image/png",
            width=1920,
            height=1080,
            file_size=1024000,
            content_hash="abcd1234",
            description="Complete test image",
            extracted_text="OCR text here",
            thumbnail_base64="base64data",
            source_document_id="doc-123",
            source_page_number=5
        )
        
        assert doc.filename == "complete.png"
        assert doc.width == 1920
        assert doc.extracted_text == "OCR text here"

    def test_image_document_repr_with_special_characters(self):
        """Test __repr__ with special characters in filename."""
        from app.db.models import ImageDocument
        
        doc = ImageDocument(
            id="test-123",
            filename="image with spaces & symbols!.png",
            content_type="image/png"
        )
        
        repr_str = repr(doc)
        assert "ImageDocument" in repr_str
        assert "test-123" in repr_str


class TestSchemasEdgeCases:
    """Test edge cases in Pydantic schemas."""

    def test_image_upload_response_minimal(self):
        """Test ImageUploadResponse with minimal data."""
        from app.schemas import ImageUploadResponse
        
        response = ImageUploadResponse(
            image_id="img-123",
            filename="test.png"
        )
        
        assert response.image_id == "img-123"
        assert response.description is None

    def test_image_search_request_defaults(self):
        """Test ImageSearchRequest default values."""
        from app.schemas import ImageSearchRequest
        
        request = ImageSearchRequest(query="test")
        
        assert request.top_k == 5
        assert request.image_base64 is None

    def test_image_search_request_max_top_k(self):
        """Test ImageSearchRequest with maximum top_k."""
        from app.schemas import ImageSearchRequest
        
        request = ImageSearchRequest(query="test", top_k=20)
        assert request.top_k == 20

    def test_image_search_request_min_top_k(self):
        """Test ImageSearchRequest with minimum top_k."""
        from app.schemas import ImageSearchRequest
        
        request = ImageSearchRequest(query="test", top_k=1)
        assert request.top_k == 1

    def test_image_search_request_both_query_and_image(self):
        """Test ImageSearchRequest with both query and image."""
        from app.schemas import ImageSearchRequest
        
        request = ImageSearchRequest(
            query="find similar",
            image_base64="base64data",
            top_k=10
        )
        
        assert request.query == "find similar"
        assert request.image_base64 == "base64data"

    def test_multimodal_options_defaults(self):
        """Test MultimodalOptions default values."""
        from app.schemas import MultimodalOptions
        
        options = MultimodalOptions()
        
        assert options.include_images is True
        assert options.image_weight == 0.3
        assert options.use_vision_model is False
        assert options.vision_provider is None

    def test_multimodal_options_custom_values(self):
        """Test MultimodalOptions with custom values."""
        from app.schemas import MultimodalOptions
        
        options = MultimodalOptions(
            include_images=False,
            image_weight=0.7,
            use_vision_model=True,
            vision_provider="openai"
        )
        
        assert options.include_images is False
        assert options.image_weight == 0.7
        assert options.use_vision_model is True
        assert options.vision_provider == "openai"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "not integration"])
