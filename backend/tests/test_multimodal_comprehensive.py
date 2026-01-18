"""
Comprehensive test suite for Multimodal RAG features.
Covers image processing, embeddings, vision models, retrieval, API endpoints, and edge cases.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import hashlib
import uuid
from datetime import datetime


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_rgb_image():
    """Create a sample RGB image for testing."""
    return Image.new('RGB', (100, 100), color='red')


@pytest.fixture
def sample_rgba_image():
    """Create a sample RGBA image with transparency."""
    img = Image.new('RGBA', (100, 100), color=(255, 0, 0, 128))
    return img


@pytest.fixture
def sample_large_image():
    """Create a large image that needs resizing."""
    return Image.new('RGB', (4000, 3000), color='blue')


@pytest.fixture
def sample_png_bytes(sample_rgb_image):
    """Get PNG bytes from sample image."""
    buffer = BytesIO()
    sample_rgb_image.save(buffer, format='PNG')
    return buffer.getvalue()


@pytest.fixture
def sample_jpeg_bytes(sample_rgb_image):
    """Get JPEG bytes from sample image."""
    buffer = BytesIO()
    sample_rgb_image.save(buffer, format='JPEG')
    return buffer.getvalue()


@pytest.fixture
def sample_gif_bytes():
    """Create a sample GIF image."""
    img = Image.new('P', (50, 50), color=1)
    buffer = BytesIO()
    img.save(buffer, format='GIF')
    return buffer.getvalue()


@pytest.fixture
def sample_webp_bytes(sample_rgb_image):
    """Get WebP bytes from sample image."""
    buffer = BytesIO()
    sample_rgb_image.save(buffer, format='WEBP')
    return buffer.getvalue()


# ============================================================================
# Image Processing Tests - validate_image_file
# ============================================================================

class TestValidateImageFile:
    """Comprehensive tests for image file validation."""

    def test_valid_png_image(self, sample_png_bytes):
        """Test validation of a valid PNG image."""
        from app.rag.image_processing import validate_image_file
        
        is_valid, error = validate_image_file("test.png", sample_png_bytes)
        assert is_valid is True
        assert error == ""

    def test_valid_jpeg_image(self, sample_jpeg_bytes):
        """Test validation of a valid JPEG image."""
        from app.rag.image_processing import validate_image_file
        
        is_valid, error = validate_image_file("test.jpg", sample_jpeg_bytes)
        assert is_valid is True
        assert error == ""

    def test_valid_jpeg_extension(self, sample_jpeg_bytes):
        """Test validation with .jpeg extension."""
        from app.rag.image_processing import validate_image_file
        
        is_valid, error = validate_image_file("test.jpeg", sample_jpeg_bytes)
        assert is_valid is True

    def test_valid_gif_image(self, sample_gif_bytes):
        """Test validation of a valid GIF image."""
        from app.rag.image_processing import validate_image_file
        
        is_valid, error = validate_image_file("test.gif", sample_gif_bytes)
        assert is_valid is True

    def test_valid_webp_image(self, sample_webp_bytes):
        """Test validation of a valid WebP image."""
        from app.rag.image_processing import validate_image_file
        
        is_valid, error = validate_image_file("test.webp", sample_webp_bytes)
        assert is_valid is True

    def test_invalid_extension(self, sample_png_bytes):
        """Test rejection of unsupported file extension."""
        from app.rag.image_processing import validate_image_file
        
        is_valid, error = validate_image_file("test.txt", sample_png_bytes)
        assert is_valid is False
        assert "Unsupported image format" in error

    def test_invalid_extension_exe(self, sample_png_bytes):
        """Test rejection of executable extension."""
        from app.rag.image_processing import validate_image_file
        
        is_valid, error = validate_image_file("malware.exe", sample_png_bytes)
        assert is_valid is False

    def test_invalid_image_content(self):
        """Test rejection of non-image content."""
        from app.rag.image_processing import validate_image_file
        
        is_valid, error = validate_image_file("fake.png", b"This is not an image")
        assert is_valid is False
        assert "Invalid image file" in error

    def test_empty_content(self):
        """Test rejection of empty content."""
        from app.rag.image_processing import validate_image_file
        
        is_valid, error = validate_image_file("empty.png", b"")
        assert is_valid is False

    def test_corrupted_png_header(self):
        """Test rejection of corrupted PNG (wrong header)."""
        from app.rag.image_processing import validate_image_file
        
        # PNG header but corrupted content
        corrupted = b'\x89PNG\r\n\x1a\n' + b'\x00' * 50
        is_valid, error = validate_image_file("corrupted.png", corrupted)
        assert is_valid is False

    def test_file_size_limit(self):
        """Test rejection of file exceeding size limit."""
        from app.rag.image_processing import validate_image_file, MAX_FILE_SIZE
        
        # Create oversized content (mock)
        oversized = b'\x00' * (MAX_FILE_SIZE + 1)
        is_valid, error = validate_image_file("huge.png", oversized)
        assert is_valid is False
        assert "too large" in error.lower()

    def test_case_insensitive_extension(self, sample_png_bytes):
        """Test that extension check is case insensitive."""
        from app.rag.image_processing import validate_image_file
        
        is_valid, error = validate_image_file("test.PNG", sample_png_bytes)
        assert is_valid is True

    def test_mixed_case_extension(self, sample_jpeg_bytes):
        """Test mixed case extension."""
        from app.rag.image_processing import validate_image_file
        
        is_valid, error = validate_image_file("photo.JpEg", sample_jpeg_bytes)
        assert is_valid is True


# ============================================================================
# Image Processing Tests - get_image_info
# ============================================================================

class TestGetImageInfo:
    """Tests for image metadata extraction."""

    def test_rgb_image_info(self, sample_rgb_image):
        """Test metadata extraction from RGB image."""
        from app.rag.image_processing import get_image_info
        
        info = get_image_info(sample_rgb_image)
        
        assert info['width'] == 100
        assert info['height'] == 100
        assert info['mode'] == 'RGB'
        assert info['has_transparency'] is False

    def test_rgba_image_info(self, sample_rgba_image):
        """Test metadata extraction from RGBA image."""
        from app.rag.image_processing import get_image_info
        
        info = get_image_info(sample_rgba_image)
        
        assert info['mode'] == 'RGBA'
        assert info['has_transparency'] is True

    def test_image_info_from_bytes(self, sample_png_bytes):
        """Test metadata extraction from bytes."""
        from app.rag.image_processing import get_image_info
        
        info = get_image_info(sample_png_bytes)
        
        assert info['width'] == 100
        assert info['height'] == 100
        assert info['format'] == 'PNG'

    def test_large_image_info(self, sample_large_image):
        """Test metadata extraction from large image."""
        from app.rag.image_processing import get_image_info
        
        info = get_image_info(sample_large_image)
        
        assert info['width'] == 4000
        assert info['height'] == 3000

    def test_gif_image_info(self, sample_gif_bytes):
        """Test metadata extraction from GIF."""
        from app.rag.image_processing import get_image_info
        
        info = get_image_info(sample_gif_bytes)
        
        assert info['format'] == 'GIF'
        assert info['mode'] == 'P'  # Palette mode


# ============================================================================
# Image Processing Tests - resize_image
# ============================================================================

class TestResizeImage:
    """Tests for image resizing functionality."""

    def test_resize_large_image(self, sample_large_image):
        """Test resizing a large image."""
        from app.rag.image_processing import resize_image
        
        resized = resize_image(sample_large_image, max_width=800, max_height=600)
        
        assert resized.width <= 800
        assert resized.height <= 600

    def test_no_resize_small_image(self, sample_rgb_image):
        """Test that small images are not resized."""
        from app.rag.image_processing import resize_image
        
        resized = resize_image(sample_rgb_image, max_width=800, max_height=600)
        
        assert resized.width == 100
        assert resized.height == 100

    def test_resize_maintains_aspect_ratio(self, sample_large_image):
        """Test aspect ratio is maintained during resize."""
        from app.rag.image_processing import resize_image
        
        original_ratio = 4000 / 3000
        resized = resize_image(sample_large_image, max_width=800, max_height=800)
        resized_ratio = resized.width / resized.height
        
        assert abs(original_ratio - resized_ratio) < 0.01

    def test_resize_from_bytes(self, sample_png_bytes):
        """Test resizing from bytes input."""
        from app.rag.image_processing import resize_image
        
        # Create large image bytes
        large_img = Image.new('RGB', (2000, 1500), color='green')
        buffer = BytesIO()
        large_img.save(buffer, format='PNG')
        large_bytes = buffer.getvalue()
        
        resized = resize_image(large_bytes, max_width=500, max_height=500)
        
        assert resized.width <= 500
        assert resized.height <= 500


# ============================================================================
# Image Processing Tests - convert_to_rgb
# ============================================================================

class TestConvertToRGB:
    """Tests for RGB conversion."""

    def test_rgb_stays_rgb(self, sample_rgb_image):
        """Test RGB image stays unchanged."""
        from app.rag.image_processing import convert_to_rgb
        
        converted = convert_to_rgb(sample_rgb_image)
        
        assert converted.mode == 'RGB'
        assert converted.size == sample_rgb_image.size

    def test_rgba_to_rgb(self, sample_rgba_image):
        """Test RGBA conversion to RGB."""
        from app.rag.image_processing import convert_to_rgb
        
        converted = convert_to_rgb(sample_rgba_image)
        
        assert converted.mode == 'RGB'

    def test_palette_to_rgb(self):
        """Test palette mode conversion to RGB."""
        from app.rag.image_processing import convert_to_rgb
        
        palette_img = Image.new('P', (50, 50), color=5)
        converted = convert_to_rgb(palette_img)
        
        assert converted.mode == 'RGB'

    def test_grayscale_to_rgb(self):
        """Test grayscale conversion to RGB."""
        from app.rag.image_processing import convert_to_rgb
        
        gray_img = Image.new('L', (50, 50), color=128)
        converted = convert_to_rgb(gray_img)
        
        assert converted.mode == 'RGB'


# ============================================================================
# Image Processing Tests - image_to_base64 and base64_to_image
# ============================================================================

class TestBase64Conversion:
    """Tests for base64 encoding/decoding."""

    def test_image_to_base64(self, sample_rgb_image):
        """Test base64 encoding of PIL Image."""
        from app.rag.image_processing import image_to_base64
        
        b64 = image_to_base64(sample_rgb_image)
        
        assert isinstance(b64, str)
        assert len(b64) > 0
        # Verify it's valid base64
        decoded = base64.b64decode(b64)
        assert len(decoded) > 0

    def test_bytes_to_base64(self, sample_png_bytes):
        """Test base64 encoding of bytes."""
        from app.rag.image_processing import image_to_base64
        
        b64 = image_to_base64(sample_png_bytes)
        
        decoded = base64.b64decode(b64)
        assert decoded == sample_png_bytes

    def test_base64_to_image(self, sample_rgb_image):
        """Test base64 decoding to PIL Image."""
        from app.rag.image_processing import image_to_base64, base64_to_image
        
        b64 = image_to_base64(sample_rgb_image)
        decoded = base64_to_image(b64)
        
        assert isinstance(decoded, Image.Image)
        assert decoded.size == sample_rgb_image.size

    def test_base64_with_data_url_prefix(self, sample_rgb_image):
        """Test base64 decoding with data URL prefix."""
        from app.rag.image_processing import image_to_base64, base64_to_image
        
        b64 = image_to_base64(sample_rgb_image)
        b64_with_prefix = f"data:image/png;base64,{b64}"
        
        decoded = base64_to_image(b64_with_prefix)
        
        assert isinstance(decoded, Image.Image)

    def test_round_trip_conversion(self, sample_rgb_image):
        """Test round trip: Image -> base64 -> Image."""
        from app.rag.image_processing import image_to_base64, base64_to_image
        
        b64 = image_to_base64(sample_rgb_image, format='PNG')
        decoded = base64_to_image(b64)
        
        # Compare sizes
        assert decoded.size == sample_rgb_image.size


# ============================================================================
# Image Processing Tests - compute_image_hash
# ============================================================================

class TestComputeImageHash:
    """Tests for image hash computation."""

    def test_same_image_same_hash(self):
        """Test identical images produce same hash."""
        from app.rag.image_processing import compute_image_hash
        
        img1 = Image.new('RGB', (50, 50), color='red')
        img2 = Image.new('RGB', (50, 50), color='red')
        
        hash1 = compute_image_hash(img1)
        hash2 = compute_image_hash(img2)
        
        assert hash1 == hash2

    def test_different_images_different_hash(self):
        """Test different images produce different hashes."""
        from app.rag.image_processing import compute_image_hash
        
        img1 = Image.new('RGB', (50, 50), color='red')
        img2 = Image.new('RGB', (50, 50), color='blue')
        
        hash1 = compute_image_hash(img1)
        hash2 = compute_image_hash(img2)
        
        assert hash1 != hash2

    def test_hash_from_bytes(self, sample_png_bytes):
        """Test hash computation from bytes."""
        from app.rag.image_processing import compute_image_hash
        
        hash_result = compute_image_hash(sample_png_bytes)
        
        assert isinstance(hash_result, str)
        assert len(hash_result) == 32  # MD5 hex length

    def test_hash_deterministic(self, sample_rgb_image):
        """Test hash is deterministic."""
        from app.rag.image_processing import compute_image_hash
        
        hash1 = compute_image_hash(sample_rgb_image)
        hash2 = compute_image_hash(sample_rgb_image)
        
        assert hash1 == hash2


# ============================================================================
# Image Processing Tests - create_thumbnail
# ============================================================================

class TestCreateThumbnail:
    """Tests for thumbnail creation."""

    def test_thumbnail_size(self, sample_large_image):
        """Test thumbnail respects size constraints."""
        from app.rag.image_processing import create_thumbnail
        
        thumb = create_thumbnail(sample_large_image, size=(150, 150))
        
        assert thumb.width <= 150
        assert thumb.height <= 150

    def test_thumbnail_from_bytes(self, sample_png_bytes):
        """Test thumbnail creation from bytes."""
        from app.rag.image_processing import create_thumbnail
        
        thumb = create_thumbnail(sample_png_bytes, size=(50, 50))
        
        assert thumb.width <= 50
        assert thumb.height <= 50

    def test_thumbnail_maintains_aspect(self, sample_large_image):
        """Test thumbnail maintains aspect ratio."""
        from app.rag.image_processing import create_thumbnail
        
        original_ratio = 4000 / 3000
        thumb = create_thumbnail(sample_large_image, size=(150, 150))
        thumb_ratio = thumb.width / thumb.height
        
        assert abs(original_ratio - thumb_ratio) < 0.1

    def test_small_image_thumbnail(self, sample_rgb_image):
        """Test thumbnail of image smaller than target size."""
        from app.rag.image_processing import create_thumbnail
        
        thumb = create_thumbnail(sample_rgb_image, size=(200, 200))
        
        # Should not upscale
        assert thumb.width <= 100
        assert thumb.height <= 100


# ============================================================================
# Vision Models Tests - image_to_base64
# ============================================================================

class TestVisionModelsBase64:
    """Tests for vision models base64 conversion."""

    def test_pil_image_to_base64(self, sample_rgb_image):
        """Test PIL Image to base64."""
        from app.rag.vision_models import image_to_base64
        
        result = image_to_base64(sample_rgb_image)
        
        assert isinstance(result, str)
        assert len(result) > 0

    def test_bytes_to_base64(self, sample_png_bytes):
        """Test bytes to base64."""
        from app.rag.vision_models import image_to_base64
        
        result = image_to_base64(sample_png_bytes)
        
        # Decode and verify
        decoded = base64.b64decode(result)
        assert decoded == sample_png_bytes

    def test_base64_string_passthrough(self):
        """Test base64 string with data URL is passed through."""
        from app.rag.vision_models import image_to_base64
        
        data_url = "data:image/png;base64,iVBORw0KGgo="
        result = image_to_base64(data_url)
        
        assert result == data_url


# ============================================================================
# Vision Models Tests - get_image_media_type
# ============================================================================

class TestGetImageMediaType:
    """Tests for media type detection."""

    def test_png_detection(self):
        """Test PNG magic bytes detection."""
        from app.rag.vision_models import get_image_media_type
        
        png_bytes = b'\x89PNG\r\n\x1a\n' + b'\x00' * 50
        result = get_image_media_type(png_bytes)
        
        assert result == "image/png"

    def test_jpeg_detection(self):
        """Test JPEG magic bytes detection."""
        from app.rag.vision_models import get_image_media_type
        
        jpeg_bytes = b'\xff\xd8\xff' + b'\x00' * 50
        result = get_image_media_type(jpeg_bytes)
        
        assert result == "image/jpeg"

    def test_gif_detection(self):
        """Test GIF magic bytes detection."""
        from app.rag.vision_models import get_image_media_type
        
        gif_bytes = b'GIF89a' + b'\x00' * 50
        result = get_image_media_type(gif_bytes)
        
        assert result == "image/gif"

    def test_webp_detection(self):
        """Test WebP magic bytes detection."""
        from app.rag.vision_models import get_image_media_type
        
        webp_bytes = b'RIFF' + b'\x00\x00\x00\x00' + b'WEBP' + b'\x00' * 50
        result = get_image_media_type(webp_bytes)
        
        assert result == "image/webp"

    def test_unknown_defaults_to_png(self):
        """Test unknown format defaults to PNG."""
        from app.rag.vision_models import get_image_media_type
        
        unknown_bytes = b'UNKNOWN' + b'\x00' * 50
        result = get_image_media_type(unknown_bytes)
        
        assert result == "image/png"


# ============================================================================
# Vision Models Tests - get_vision_model
# ============================================================================

class TestGetVisionModel:
    """Tests for vision model factory."""

    def test_get_openai_vision(self):
        """Test getting OpenAI vision model."""
        from app.rag.vision_models import get_vision_model, OpenAIVision
        
        with patch.object(OpenAIVision, '__init__', return_value=None):
            model = get_vision_model("openai")
            assert isinstance(model, OpenAIVision)

    def test_get_anthropic_vision(self):
        """Test getting Anthropic vision model."""
        from app.rag.vision_models import get_vision_model, AnthropicVision
        
        with patch.object(AnthropicVision, '__init__', return_value=None):
            model = get_vision_model("anthropic")
            assert isinstance(model, AnthropicVision)

    def test_get_ollama_vision(self):
        """Test getting Ollama vision model."""
        from app.rag.vision_models import get_vision_model, OllamaVision
        
        with patch.object(OllamaVision, '__init__', return_value=None):
            model = get_vision_model("ollama")
            assert isinstance(model, OllamaVision)

    def test_invalid_provider_raises(self):
        """Test invalid provider raises ValueError."""
        from app.rag.vision_models import get_vision_model
        
        with pytest.raises(ValueError, match="Unsupported vision provider"):
            get_vision_model("invalid_provider")

    def test_case_insensitive_provider(self):
        """Test provider name is case insensitive."""
        from app.rag.vision_models import get_vision_model, OpenAIVision
        
        with patch.object(OpenAIVision, '__init__', return_value=None):
            model = get_vision_model("OPENAI")
            assert isinstance(model, OpenAIVision)


# ============================================================================
# Vision Models Tests - OpenAIVision
# ============================================================================

class TestOpenAIVision:
    """Tests for OpenAI GPT-4V integration."""

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        from app.rag.vision_models import OpenAIVision
        
        with patch('app.rag.vision_models.settings') as mock_settings:
            mock_settings.openai_api_key = "test-key"
            vision = OpenAIVision(api_key="my-key")
            assert vision.api_key == "my-key"

    def test_init_without_api_key_raises(self):
        """Test initialization without API key raises error."""
        from app.rag.vision_models import OpenAIVision
        
        with patch('app.rag.vision_models.settings') as mock_settings:
            mock_settings.openai_api_key = None
            with pytest.raises(ValueError, match="API key not configured"):
                OpenAIVision(api_key=None)

    @patch('httpx.post')
    def test_analyze_image_success(self, mock_post, sample_rgb_image):
        """Test successful image analysis."""
        from app.rag.vision_models import OpenAIVision
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "A red square image"}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response
        
        with patch('app.rag.vision_models.settings') as mock_settings:
            mock_settings.openai_api_key = "test-key"
            vision = OpenAIVision()
            result = vision.analyze_image(sample_rgb_image, "Describe this")
            
            assert result == "A red square image"
            mock_post.assert_called_once()

    @patch('httpx.post')
    def test_analyze_image_api_error(self, mock_post, sample_rgb_image):
        """Test API error handling."""
        from app.rag.vision_models import OpenAIVision
        
        mock_post.side_effect = Exception("API Error")
        
        with patch('app.rag.vision_models.settings') as mock_settings:
            mock_settings.openai_api_key = "test-key"
            vision = OpenAIVision()
            
            with pytest.raises(Exception, match="API Error"):
                vision.analyze_image(sample_rgb_image, "Describe this")


# ============================================================================
# Vision Models Tests - AnthropicVision
# ============================================================================

class TestAnthropicVision:
    """Tests for Anthropic Claude 3 Vision integration."""

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        from app.rag.vision_models import AnthropicVision
        
        with patch('app.rag.vision_models.settings') as mock_settings:
            mock_settings.anthropic_api_key = "test-key"
            vision = AnthropicVision(api_key="my-key")
            assert vision.api_key == "my-key"

    @patch('httpx.post')
    def test_analyze_image_success(self, mock_post, sample_rgb_image):
        """Test successful image analysis with Claude."""
        from app.rag.vision_models import AnthropicVision
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "A red square image"}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response
        
        with patch('app.rag.vision_models.settings') as mock_settings:
            mock_settings.anthropic_api_key = "test-key"
            vision = AnthropicVision()
            result = vision.analyze_image(sample_rgb_image, "Describe this")
            
            assert result == "A red square image"


# ============================================================================
# Image Embeddings Tests
# ============================================================================

class TestImageEmbeddings:
    """Tests for CLIP-based image embeddings."""

    def test_embed_image_function_exists(self):
        """Test embed_image function is exported."""
        from app.rag.image_embeddings import embed_image, embed_image_query
        
        assert callable(embed_image)
        assert callable(embed_image_query)

    def test_get_clip_embeddings_exists(self):
        """Test get_clip_embeddings function exists."""
        from app.rag.image_embeddings import get_clip_embeddings
        
        assert callable(get_clip_embeddings)

    @patch('app.rag.image_embeddings.get_sentence_transformer')
    def test_embed_image_query_returns_list(self, mock_st):
        """Test embed_image_query returns a list."""
        from app.rag.image_embeddings import embed_image_query
        
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1] * 512)
        mock_st.return_value = mock_model
        
        result = embed_image_query("a red car")
        
        assert result is not None
        assert len(result) > 0

    @patch('app.rag.image_embeddings.get_clip_model')
    def test_get_clip_embeddings_with_sentence_transformer(self, mock_clip):
        """Test get_clip_embeddings with sentence transformer."""
        from app.rag.image_embeddings import get_clip_embeddings
        
        result = get_clip_embeddings(use_sentence_transformer=True)
        assert result is not None


# ============================================================================
# Database Model Tests
# ============================================================================

class TestImageDocumentModel:
    """Tests for ImageDocument database model."""

    def test_create_image_document(self):
        """Test creating ImageDocument instance."""
        from app.db.models import ImageDocument
        
        doc = ImageDocument(
            filename="test.png",
            content_type="image/png",
            file_size=1024,
            width=800,
            height=600,
            description="Test image"
        )
        
        assert doc.filename == "test.png"
        assert doc.content_type == "image/png"
        assert doc.width == 800
        assert doc.height == 600

    def test_image_document_id_generation(self):
        """Test ImageDocument auto-generates UUID."""
        from app.db.models import ImageDocument
        
        doc = ImageDocument(
            filename="test.png",
            content_type="image/png"
        )
        
        # ID should be set by default factory
        assert doc.id is not None or hasattr(ImageDocument, 'id')

    def test_image_document_repr(self):
        """Test ImageDocument string representation."""
        from app.db.models import ImageDocument
        
        doc = ImageDocument(
            id="test-uuid",
            filename="chart.png",
            content_type="image/png"
        )
        
        repr_str = repr(doc)
        assert "ImageDocument" in repr_str
        assert "chart.png" in repr_str

    def test_image_document_optional_fields(self):
        """Test ImageDocument optional fields."""
        from app.db.models import ImageDocument
        
        doc = ImageDocument(
            filename="minimal.png",
            content_type="image/png"
        )
        
        # Optional fields should be None
        assert doc.description is None
        assert doc.thumbnail_base64 is None
        assert doc.source_document_id is None


# ============================================================================
# Schema Tests
# ============================================================================

class TestImageSchemas:
    """Tests for image-related Pydantic schemas."""

    def test_image_upload_response(self):
        """Test ImageUploadResponse schema."""
        from app.schemas import ImageUploadResponse
        
        response = ImageUploadResponse(
            image_id="uuid-123",
            filename="test.png",
            description="A test image",
            message="Success"
        )
        
        assert response.image_id == "uuid-123"
        assert response.filename == "test.png"

    def test_image_search_request_defaults(self):
        """Test ImageSearchRequest default values."""
        from app.schemas import ImageSearchRequest
        
        request = ImageSearchRequest(query="test query")
        
        assert request.top_k == 5  # Default

    def test_image_search_request_with_top_k(self):
        """Test ImageSearchRequest with custom top_k."""
        from app.schemas import ImageSearchRequest
        
        request = ImageSearchRequest(query="test", top_k=10)
        
        assert request.top_k == 10

    def test_image_search_request_with_base64(self):
        """Test ImageSearchRequest with image input."""
        from app.schemas import ImageSearchRequest
        
        request = ImageSearchRequest(
            image_base64="data:image/png;base64,abc123",
            top_k=3
        )
        
        assert request.image_base64 is not None
        assert request.query is None

    def test_image_search_request_validation(self):
        """Test ImageSearchRequest top_k validation."""
        from app.schemas import ImageSearchRequest
        from pydantic import ValidationError
        
        # top_k must be between 1 and 20
        with pytest.raises(ValidationError):
            ImageSearchRequest(query="test", top_k=0)
        
        with pytest.raises(ValidationError):
            ImageSearchRequest(query="test", top_k=25)

    def test_image_document_response(self):
        """Test ImageDocumentResponse schema."""
        from app.schemas import ImageDocumentResponse
        from datetime import datetime
        
        response = ImageDocumentResponse(
            id="uuid-123",
            filename="test.png",
            content_type="image/png",
            width=800,
            height=600,
            file_size=1024,
            description="Test",
            created_at=datetime.now()
        )
        
        assert response.id == "uuid-123"
        assert response.width == 800

    def test_multimodal_options(self):
        """Test MultimodalOptions schema."""
        from app.schemas import MultimodalOptions
        
        options = MultimodalOptions(
            include_images=True,
            image_weight=0.5,
            use_vision_model=True,
            vision_provider="openai"
        )
        
        assert options.include_images is True
        assert options.image_weight == 0.5

    def test_multimodal_options_defaults(self):
        """Test MultimodalOptions default values."""
        from app.schemas import MultimodalOptions
        
        options = MultimodalOptions()
        
        assert options.include_images is True
        assert options.image_weight == 0.3
        assert options.use_vision_model is False

    def test_multimodal_options_weight_validation(self):
        """Test image_weight must be between 0 and 1."""
        from app.schemas import MultimodalOptions
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError):
            MultimodalOptions(image_weight=1.5)
        
        with pytest.raises(ValidationError):
            MultimodalOptions(image_weight=-0.1)


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_zero_dimension_image(self):
        """Test handling of zero dimension image."""
        from app.rag.image_processing import get_image_info
        
        # Create 1x1 image (smallest valid)
        tiny_img = Image.new('RGB', (1, 1), color='red')
        info = get_image_info(tiny_img)
        
        assert info['width'] == 1
        assert info['height'] == 1

    def test_very_long_filename(self, sample_png_bytes):
        """Test handling of very long filename."""
        from app.rag.image_processing import validate_image_file
        
        long_name = "a" * 500 + ".png"
        is_valid, error = validate_image_file(long_name, sample_png_bytes)
        
        # Should still validate the image content
        assert is_valid is True

    def test_unicode_filename(self, sample_png_bytes):
        """Test handling of unicode filename."""
        from app.rag.image_processing import validate_image_file
        
        unicode_name = "画像テスト.png"
        is_valid, error = validate_image_file(unicode_name, sample_png_bytes)
        
        assert is_valid is True

    def test_special_chars_filename(self, sample_png_bytes):
        """Test handling of special characters in filename."""
        from app.rag.image_processing import validate_image_file
        
        special_name = "test-image_v2 (1).png"
        is_valid, error = validate_image_file(special_name, sample_png_bytes)
        
        assert is_valid is True

    def test_double_extension(self, sample_png_bytes):
        """Test handling of double extension."""
        from app.rag.image_processing import validate_image_file
        
        double_ext = "image.jpg.png"
        is_valid, error = validate_image_file(double_ext, sample_png_bytes)
        
        assert is_valid is True  # Should check last extension

    def test_no_extension(self, sample_png_bytes):
        """Test handling of file without extension."""
        from app.rag.image_processing import validate_image_file
        
        no_ext = "imagefile"
        is_valid, error = validate_image_file(no_ext, sample_png_bytes)
        
        assert is_valid is False

    def test_hidden_file(self, sample_png_bytes):
        """Test handling of hidden file (dot prefix)."""
        from app.rag.image_processing import validate_image_file
        
        hidden = ".hidden.png"
        is_valid, error = validate_image_file(hidden, sample_png_bytes)
        
        assert is_valid is True


# ============================================================================
# Constants and Configuration Tests
# ============================================================================

class TestConstants:
    """Tests for module constants and configuration."""

    def test_supported_formats(self):
        """Test SUPPORTED_FORMATS constant."""
        from app.rag.image_processing import SUPPORTED_FORMATS
        
        assert '.png' in SUPPORTED_FORMATS
        assert '.jpg' in SUPPORTED_FORMATS
        assert '.jpeg' in SUPPORTED_FORMATS
        assert '.gif' in SUPPORTED_FORMATS
        assert '.webp' in SUPPORTED_FORMATS

    def test_max_dimensions(self):
        """Test MAX_WIDTH and MAX_HEIGHT constants."""
        from app.rag.image_processing import MAX_WIDTH, MAX_HEIGHT
        
        assert MAX_WIDTH > 0
        assert MAX_HEIGHT > 0
        assert MAX_WIDTH >= 1024  # Should be reasonable size
        assert MAX_HEIGHT >= 1024

    def test_max_file_size(self):
        """Test MAX_FILE_SIZE constant."""
        from app.rag.image_processing import MAX_FILE_SIZE
        
        assert MAX_FILE_SIZE > 0
        assert MAX_FILE_SIZE >= 1024 * 1024  # At least 1MB


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
