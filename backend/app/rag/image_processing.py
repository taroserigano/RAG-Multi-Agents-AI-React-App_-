"""
Image processing utilities for multimodal RAG.
Handles image validation, resizing, format conversion, and metadata extraction.
"""
from typing import Tuple, Dict, Any, Optional, Union
from PIL import Image
import io
import base64
import hashlib
from pathlib import Path

from app.core.logging import get_logger

logger = get_logger(__name__)

# Supported image formats
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}
SUPPORTED_MIME_TYPES = {
    'image/jpeg': '.jpg',
    'image/png': '.png',
    'image/gif': '.gif',
    'image/webp': '.webp',
    'image/bmp': '.bmp'
}

# Maximum dimensions for processing
MAX_WIDTH = 2048
MAX_HEIGHT = 2048
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB


def validate_image_file(filename: str, content: bytes) -> Tuple[bool, str]:
    """
    Validate image file type and content.
    
    Args:
        filename: Original filename
        content: File content as bytes
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check file extension
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        return False, f"Unsupported image format. Allowed: {', '.join(SUPPORTED_FORMATS)}"
    
    # Check file size
    if len(content) > MAX_FILE_SIZE:
        return False, f"Image too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
    
    # Verify it's actually an image
    try:
        img = Image.open(io.BytesIO(content))
        img.verify()
        return True, ""
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"


def get_image_info(image: Union[Image.Image, bytes]) -> Dict[str, Any]:
    """
    Extract metadata from an image.
    
    Args:
        image: PIL Image or bytes
    
    Returns:
        Dictionary with image metadata
    """
    if isinstance(image, bytes):
        image = Image.open(io.BytesIO(image))
    
    width, height = image.size
    
    return {
        "width": width,
        "height": height,
        "format": image.format or "Unknown",
        "mode": image.mode,
        "has_transparency": image.mode in ('RGBA', 'LA', 'P'),
        "is_animated": getattr(image, 'is_animated', False)
    }


def resize_image(
    image: Union[Image.Image, bytes],
    max_width: int = MAX_WIDTH,
    max_height: int = MAX_HEIGHT,
    maintain_aspect: bool = True
) -> Image.Image:
    """
    Resize image if it exceeds maximum dimensions.
    
    Args:
        image: PIL Image or bytes
        max_width: Maximum width
        max_height: Maximum height
        maintain_aspect: Keep aspect ratio
    
    Returns:
        Resized PIL Image
    """
    if isinstance(image, bytes):
        image = Image.open(io.BytesIO(image))
    
    width, height = image.size
    
    # Check if resizing is needed
    if width <= max_width and height <= max_height:
        return image
    
    if maintain_aspect:
        # Calculate new size maintaining aspect ratio
        ratio = min(max_width / width, max_height / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
    else:
        new_width = min(width, max_width)
        new_height = min(height, max_height)
    
    logger.info(f"Resizing image from {width}x{height} to {new_width}x{new_height}")
    
    return image.resize((new_width, new_height), Image.LANCZOS)


def convert_to_rgb(image: Union[Image.Image, bytes]) -> Image.Image:
    """
    Convert image to RGB mode (required for some models).
    
    Args:
        image: PIL Image or bytes
    
    Returns:
        RGB PIL Image
    """
    if isinstance(image, bytes):
        image = Image.open(io.BytesIO(image))
    
    if image.mode == 'RGB':
        return image
    
    if image.mode in ('RGBA', 'LA', 'P'):
        # Create white background for transparent images
        background = Image.new('RGB', image.size, (255, 255, 255))
        if image.mode == 'P':
            image = image.convert('RGBA')
        background.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
        return background
    
    return image.convert('RGB')


def image_to_base64(image: Union[Image.Image, bytes], format: str = 'PNG') -> str:
    """
    Convert image to base64 encoded string.
    
    Args:
        image: PIL Image or bytes
        format: Output format (PNG, JPEG, etc.)
    
    Returns:
        Base64 encoded string
    """
    if isinstance(image, bytes):
        return base64.b64encode(image).decode('utf-8')
    
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def base64_to_image(b64_string: str) -> Image.Image:
    """
    Convert base64 string to PIL Image.
    
    Args:
        b64_string: Base64 encoded image (with or without data URL prefix)
    
    Returns:
        PIL Image
    """
    # Remove data URL prefix if present
    if b64_string.startswith('data:image'):
        b64_string = b64_string.split(',')[1]
    
    image_data = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(image_data))


def compute_image_hash(image: Union[Image.Image, bytes]) -> str:
    """
    Compute MD5 hash of image for deduplication.
    
    Args:
        image: PIL Image or bytes
    
    Returns:
        MD5 hash string
    """
    if isinstance(image, Image.Image):
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()
    else:
        image_bytes = image
    
    return hashlib.md5(image_bytes).hexdigest()


def create_thumbnail(
    image: Union[Image.Image, bytes],
    size: Tuple[int, int] = (150, 150)
) -> Image.Image:
    """
    Create thumbnail for preview.
    
    Args:
        image: PIL Image or bytes
        size: Thumbnail size (width, height)
    
    Returns:
        Thumbnail PIL Image
    """
    if isinstance(image, bytes):
        image = Image.open(io.BytesIO(image))
    
    thumbnail = image.copy()
    thumbnail.thumbnail(size, Image.LANCZOS)
    return thumbnail


def prepare_image_for_embedding(
    image: Union[Image.Image, bytes],
    target_size: Tuple[int, int] = (224, 224)
) -> Image.Image:
    """
    Prepare image for CLIP embedding (specific size and format).
    
    Args:
        image: PIL Image or bytes
        target_size: Expected model input size
    
    Returns:
        Processed PIL Image
    """
    if isinstance(image, bytes):
        image = Image.open(io.BytesIO(image))
    
    # Convert to RGB
    image = convert_to_rgb(image)
    
    # Resize maintaining aspect ratio, then center crop
    width, height = image.size
    ratio = max(target_size[0] / width, target_size[1] / height)
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    
    image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Center crop
    left = (new_width - target_size[0]) // 2
    top = (new_height - target_size[1]) // 2
    right = left + target_size[0]
    bottom = top + target_size[1]
    
    return image.crop((left, top, right, bottom))


def prepare_image_for_vision_api(
    image: Union[Image.Image, bytes],
    max_size: int = 2048,
    max_file_size: int = 5 * 1024 * 1024  # 5MB for API calls
) -> bytes:
    """
    Prepare image for vision API (size and quality optimization).
    
    Args:
        image: PIL Image or bytes
        max_size: Maximum dimension
        max_file_size: Maximum file size in bytes
    
    Returns:
        Optimized image bytes
    """
    if isinstance(image, bytes):
        image = Image.open(io.BytesIO(image))
    
    # Convert to RGB
    image = convert_to_rgb(image)
    
    # Resize if needed
    image = resize_image(image, max_size, max_size)
    
    # Try different quality levels to fit within file size
    for quality in [95, 85, 75, 60, 45]:
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=quality, optimize=True)
        image_bytes = buffer.getvalue()
        
        if len(image_bytes) <= max_file_size:
            return image_bytes
    
    # If still too large, resize more aggressively
    scale = 0.8
    while scale > 0.2:
        new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
        resized = image.resize(new_size, Image.LANCZOS)
        
        buffer = io.BytesIO()
        resized.save(buffer, format='JPEG', quality=60, optimize=True)
        image_bytes = buffer.getvalue()
        
        if len(image_bytes) <= max_file_size:
            return image_bytes
        
        scale -= 0.1
    
    return image_bytes


def extract_image_from_pdf_page(pdf_path: str, page_num: int) -> Optional[Image.Image]:
    """
    Extract image from a PDF page (for PDF with embedded images).
    
    Args:
        pdf_path: Path to PDF file
        page_num: Page number (0-indexed)
    
    Returns:
        PIL Image if extracted, None otherwise
    """
    try:
        import pdfplumber
        
        with pdfplumber.open(pdf_path) as pdf:
            if page_num >= len(pdf.pages):
                return None
            
            page = pdf.pages[page_num]
            images = page.images
            
            if not images:
                return None
            
            # Get the first image on the page
            # In a real implementation, you might want to extract all images
            img_data = images[0]
            
            # Render the page as an image
            page_image = page.to_image(resolution=150)
            return page_image.original
            
    except Exception as e:
        logger.error(f"Error extracting image from PDF: {e}")
        return None


class ImageProcessor:
    """
    Main image processor class for multimodal RAG pipeline.
    """
    
    def __init__(self):
        self.supported_formats = SUPPORTED_FORMATS
    
    def process_for_indexing(
        self,
        image: Union[Image.Image, bytes],
        filename: str
    ) -> Dict[str, Any]:
        """
        Process image for indexing in vector store.
        
        Args:
            image: Image to process
            filename: Original filename
        
        Returns:
            Dictionary with processed data
        """
        if isinstance(image, bytes):
            original_bytes = image
            image = Image.open(io.BytesIO(image))
        else:
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            original_bytes = buffer.getvalue()
        
        return {
            "image": convert_to_rgb(image),
            "embedding_ready": prepare_image_for_embedding(image),
            "thumbnail": create_thumbnail(image),
            "hash": compute_image_hash(original_bytes),
            "info": get_image_info(image),
            "filename": filename
        }
    
    def process_for_query(self, image: Union[Image.Image, bytes]) -> Dict[str, Any]:
        """
        Process image for query (user-uploaded image for search).
        
        Args:
            image: Query image
        
        Returns:
            Processed image data
        """
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        
        return {
            "image": convert_to_rgb(image),
            "embedding_ready": prepare_image_for_embedding(image),
            "api_ready": prepare_image_for_vision_api(image)
        }


# Default processor instance
image_processor = ImageProcessor()
