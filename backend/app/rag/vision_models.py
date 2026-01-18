"""
Vision model integration for multimodal RAG.
Supports GPT-4V (OpenAI) and Claude 3 Vision (Anthropic) for image understanding.
"""
from typing import List, Optional, Union, Dict, Any
from PIL import Image
import io
import base64
import httpx

from app.core.config import get_settings
from app.core.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)


def image_to_base64(image: Union[Image.Image, bytes, str]) -> str:
    """
    Convert image to base64 encoded string.
    
    Args:
        image: PIL Image, bytes, or file path
    
    Returns:
        Base64 encoded image string
    """
    if isinstance(image, str):
        # If it's already base64 with data URL prefix
        if image.startswith('data:image'):
            return image
        # If it's a file path
        with open(image, 'rb') as f:
            image_data = f.read()
    elif isinstance(image, bytes):
        image_data = image
    elif isinstance(image, Image.Image):
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        image_data = buffer.getvalue()
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")
    
    return base64.b64encode(image_data).decode('utf-8')


def get_image_media_type(image_data: bytes) -> str:
    """Detect image media type from bytes."""
    # Check magic bytes
    if image_data[:8] == b'\x89PNG\r\n\x1a\n':
        return "image/png"
    elif image_data[:2] == b'\xff\xd8':
        return "image/jpeg"
    elif image_data[:4] == b'GIF8':
        return "image/gif"
    elif image_data[:4] == b'RIFF' and image_data[8:12] == b'WEBP':
        return "image/webp"
    else:
        return "image/png"  # Default


class OpenAIVision:
    """
    GPT-4V integration for image understanding.
    Uses OpenAI's vision-enabled models to analyze images.
    """
    
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or settings.openai_api_key
        self.model = model or "gpt-4o"  # gpt-4o has vision capabilities
        self.base_url = "https://api.openai.com/v1/chat/completions"
        
        if not self.api_key:
            raise ValueError("OpenAI API key not configured")
    
    def analyze_image(
        self,
        image: Union[Image.Image, bytes, str],
        prompt: str = "Describe this image in detail.",
        max_tokens: int = 500
    ) -> str:
        """
        Analyze an image using GPT-4V.
        
        Args:
            image: Image to analyze
            prompt: Question or instruction about the image
            max_tokens: Maximum response length
        
        Returns:
            Model's analysis/description of the image
        """
        base64_image = image_to_base64(image)
        
        # Determine if we need to add data URL prefix
        if not base64_image.startswith('data:image'):
            base64_image = f"data:image/png;base64,{base64_image}"
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": base64_image, "detail": "auto"}
                    }
                ]
            }
        ]
        
        try:
            response = httpx.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens
                },
                timeout=60.0
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            logger.error(f"OpenAI Vision API error: {e}")
            raise
    
    def analyze_with_context(
        self,
        image: Union[Image.Image, bytes, str],
        query: str,
        context: str,
        max_tokens: int = 500
    ) -> str:
        """
        Analyze image in context of RAG-retrieved documents.
        
        Args:
            image: Image to analyze
            query: User's question
            context: Retrieved document context
            max_tokens: Maximum response length
        
        Returns:
            Contextual analysis combining image and document knowledge
        """
        prompt = f"""Based on the following context from company documents and the image provided, answer the user's question.

Context from documents:
{context}

User's question: {query}

Analyze the image in relation to the question and context. If the image contains relevant information (like charts, diagrams, or text), incorporate it into your answer."""
        
        return self.analyze_image(image, prompt, max_tokens)
    
    def extract_text_from_image(self, image: Union[Image.Image, bytes, str]) -> str:
        """
        Extract text content from an image (OCR-like functionality).
        
        Args:
            image: Image potentially containing text
        
        Returns:
            Extracted text content
        """
        prompt = """Extract and transcribe all visible text from this image. 
Preserve the structure and formatting as much as possible.
If there's no text, describe what you see instead."""
        
        return self.analyze_image(image, prompt, max_tokens=1000)
    
    def generate_image_description(self, image: Union[Image.Image, bytes, str]) -> str:
        """
        Generate a detailed description for indexing.
        
        Args:
            image: Image to describe
        
        Returns:
            Detailed description for vector embedding
        """
        prompt = """Provide a detailed description of this image that would be useful for search and retrieval. Include:
1. Main subject matter
2. Visual elements (colors, shapes, layout)
3. Any text visible
4. Context and purpose (if apparent)
5. Key details that someone might search for

Be comprehensive but concise."""
        
        return self.analyze_image(image, prompt, max_tokens=500)


class AnthropicVision:
    """
    Claude 3 Vision integration for image understanding.
    Uses Anthropic's Claude 3 models with vision capabilities.
    """
    
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or settings.anthropic_api_key
        self.model = model or "claude-3-5-sonnet-latest"  # Claude 3.5 Sonnet has vision
        self.base_url = "https://api.anthropic.com/v1/messages"
        
        if not self.api_key:
            raise ValueError("Anthropic API key not configured")
    
    def analyze_image(
        self,
        image: Union[Image.Image, bytes, str],
        prompt: str = "Describe this image in detail.",
        max_tokens: int = 500
    ) -> str:
        """
        Analyze an image using Claude 3 Vision.
        
        Args:
            image: Image to analyze
            prompt: Question or instruction about the image
            max_tokens: Maximum response length
        
        Returns:
            Model's analysis/description of the image
        """
        # Get raw bytes for media type detection
        if isinstance(image, str) and not image.startswith('data:image'):
            with open(image, 'rb') as f:
                image_bytes = f.read()
        elif isinstance(image, bytes):
            image_bytes = image
        elif isinstance(image, Image.Image):
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            image_bytes = buffer.getvalue()
        else:
            image_bytes = base64.b64decode(image.split(',')[1] if image.startswith('data:') else image)
        
        media_type = get_image_media_type(image_bytes)
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64_image
                        }
                    },
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        try:
            response = httpx.post(
                self.base_url,
                headers={
                    "x-api-key": self.api_key,
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01"
                },
                json={
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "messages": messages
                },
                timeout=60.0
            )
            response.raise_for_status()
            result = response.json()
            return result["content"][0]["text"]
            
        except Exception as e:
            logger.error(f"Anthropic Vision API error: {e}")
            raise
    
    def analyze_with_context(
        self,
        image: Union[Image.Image, bytes, str],
        query: str,
        context: str,
        max_tokens: int = 500
    ) -> str:
        """
        Analyze image in context of RAG-retrieved documents.
        """
        prompt = f"""Based on the following context from company documents and the image provided, answer the user's question.

Context from documents:
{context}

User's question: {query}

Analyze the image in relation to the question and context. If the image contains relevant information (like charts, diagrams, or text), incorporate it into your answer."""
        
        return self.analyze_image(image, prompt, max_tokens)
    
    def extract_text_from_image(self, image: Union[Image.Image, bytes, str]) -> str:
        """Extract text content from an image."""
        prompt = """Extract and transcribe all visible text from this image. 
Preserve the structure and formatting as much as possible.
If there's no text, describe what you see instead."""
        
        return self.analyze_image(image, prompt, max_tokens=1000)
    
    def generate_image_description(self, image: Union[Image.Image, bytes, str]) -> str:
        """Generate a detailed description for indexing."""
        prompt = """Provide a detailed description of this image that would be useful for search and retrieval. Include:
1. Main subject matter
2. Visual elements (colors, shapes, layout)
3. Any text visible
4. Context and purpose (if apparent)
5. Key details that someone might search for

Be comprehensive but concise."""
        
        return self.analyze_image(image, prompt, max_tokens=500)


class OllamaVision:
    """
    Ollama vision model integration for local image understanding.
    Uses Ollama's llava or other vision models.
    """
    
    def __init__(self, base_url: str = None, model: str = None):
        self.base_url = base_url or settings.ollama_base_url
        self.model = model or "llava"  # LLaVA is Ollama's vision model
        self.endpoint = f"{self.base_url}/api/generate"
    
    def analyze_image(
        self,
        image: Union[Image.Image, bytes, str],
        prompt: str = "Describe this image in detail.",
        max_tokens: int = 500
    ) -> str:
        """
        Analyze an image using Ollama vision model (LLaVA).
        
        Args:
            image: Image to analyze
            prompt: Question or instruction about the image
            max_tokens: Maximum response length
        
        Returns:
            Model's analysis/description of the image
        """
        base64_image = image_to_base64(image)
        
        # Remove data URL prefix if present
        if base64_image.startswith('data:image'):
            base64_image = base64_image.split(',')[1]
        
        try:
            response = httpx.post(
                self.endpoint,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "images": [base64_image],
                    "stream": False,
                    "options": {"num_predict": max_tokens}
                },
                timeout=120.0  # Vision models can be slow
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
            
        except Exception as e:
            logger.error(f"Ollama Vision API error: {e}")
            raise
    
    def analyze_with_context(
        self,
        image: Union[Image.Image, bytes, str],
        query: str,
        context: str,
        max_tokens: int = 500
    ) -> str:
        """Analyze image in context of RAG-retrieved documents."""
        prompt = f"""Based on the following context and the image provided, answer the question.

Context:
{context}

Question: {query}

Analyze the image in relation to the question and context."""
        
        return self.analyze_image(image, prompt, max_tokens)
    
    def generate_image_description(self, image: Union[Image.Image, bytes, str]) -> str:
        """Generate a detailed description for indexing."""
        prompt = """Describe this image in detail for search and retrieval purposes.
Include: subject matter, visual elements, any text visible, and key searchable details."""
        
        return self.analyze_image(image, prompt, max_tokens=500)


def get_vision_model(provider: str = "openai") -> Union[OpenAIVision, AnthropicVision, OllamaVision]:
    """
    Factory function to get vision model by provider.
    
    Args:
        provider: "openai", "anthropic", or "ollama" (case-insensitive)
    
    Returns:
        Vision model instance
    """
    providers = {
        "openai": OpenAIVision,
        "anthropic": AnthropicVision,
        "ollama": OllamaVision
    }
    
    provider_lower = provider.lower()
    
    if provider_lower not in providers:
        raise ValueError(f"Unsupported vision provider: {provider}. Use: {list(providers.keys())}")
    
    return providers[provider_lower]()


def analyze_image_with_context(
    image: Union[Image.Image, bytes, str],
    query: str,
    context: str,
    provider: str = "openai"
) -> str:
    """
    Convenience function to analyze image with RAG context.
    
    Args:
        image: Image to analyze
        query: User's question
        context: Retrieved document context
        provider: Vision model provider
    
    Returns:
        Analysis combining image and context
    """
    vision = get_vision_model(provider)
    return vision.analyze_with_context(image, query, context)


def generate_image_description_for_indexing(
    image: Union[Image.Image, bytes, str],
    provider: str = "openai"
) -> str:
    """
    Generate image description for vector indexing.
    
    Args:
        image: Image to describe
        provider: Vision model provider
    
    Returns:
        Description text for embedding
    """
    vision = get_vision_model(provider)
    return vision.generate_image_description(image)
