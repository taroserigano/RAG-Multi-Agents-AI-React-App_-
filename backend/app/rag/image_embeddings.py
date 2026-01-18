"""
Image embeddings using CLIP model for multimodal RAG.
Supports both image and text embeddings in the same vector space.
"""
from typing import List, Optional, Union
from PIL import Image
import io
import base64
import numpy as np

from app.core.config import get_settings
from app.core.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)

# Lazy loading for heavy models
_clip_model = None
_clip_processor = None
_sentence_transformer = None


def get_clip_model():
    """Lazy load CLIP model to avoid startup delay."""
    global _clip_model, _clip_processor
    
    if _clip_model is None:
        try:
            from transformers import CLIPModel, CLIPProcessor
            
            model_name = "openai/clip-vit-base-patch32"
            logger.info(f"Loading CLIP model: {model_name}")
            
            _clip_model = CLIPModel.from_pretrained(model_name)
            _clip_processor = CLIPProcessor.from_pretrained(model_name)
            
            logger.info("CLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise
    
    return _clip_model, _clip_processor


def get_sentence_transformer():
    """Lazy load sentence transformer for text-only CLIP embeddings."""
    global _sentence_transformer
    
    if _sentence_transformer is None:
        try:
            from sentence_transformers import SentenceTransformer
            
            model_name = "clip-ViT-B-32"
            logger.info(f"Loading SentenceTransformer CLIP: {model_name}")
            
            _sentence_transformer = SentenceTransformer(model_name)
            
            logger.info("SentenceTransformer CLIP loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer: {e}")
            raise
    
    return _sentence_transformer


class CLIPEmbeddings:
    """
    CLIP-based embeddings for multimodal RAG.
    
    Generates embeddings for both images and text in a shared vector space,
    enabling cross-modal retrieval (text query → image results and vice versa).
    """
    
    def __init__(self):
        self.embedding_dim = 512  # CLIP ViT-B/32 output dimension
    
    def embed_image(self, image: Union[Image.Image, bytes, str]) -> List[float]:
        """
        Generate CLIP embedding for a single image.
        
        Args:
            image: PIL Image, bytes, or base64 string
        
        Returns:
            Embedding vector (512 dimensions)
        """
        try:
            # Convert to PIL Image if needed
            pil_image = self._to_pil_image(image)
            
            model, processor = get_clip_model()
            
            # Process and encode
            inputs = processor(images=pil_image, return_tensors="pt")
            
            import torch
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
                # Normalize embedding
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            embedding = image_features.squeeze().cpu().numpy().tolist()
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating image embedding: {e}")
            raise
    
    def embed_images(self, images: List[Union[Image.Image, bytes, str]]) -> List[List[float]]:
        """
        Generate CLIP embeddings for multiple images.
        
        Args:
            images: List of PIL Images, bytes, or base64 strings
        
        Returns:
            List of embedding vectors
        """
        embeddings = []
        for image in images:
            embedding = self.embed_image(image)
            embeddings.append(embedding)
        return embeddings
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate CLIP text embedding for cross-modal retrieval.
        
        Args:
            text: Text query or description
        
        Returns:
            Embedding vector (512 dimensions)
        """
        try:
            model, processor = get_clip_model()
            
            # Process and encode
            inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
            
            import torch
            with torch.no_grad():
                text_features = model.get_text_features(**inputs)
                # Normalize embedding
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            embedding = text_features.squeeze().cpu().numpy().tolist()
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating text embedding: {e}")
            raise
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate CLIP embeddings for multiple texts.
        
        Args:
            texts: List of text strings
        
        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in texts:
            embedding = self.embed_text(text)
            embeddings.append(embedding)
        return embeddings
    
    def _to_pil_image(self, image: Union[Image.Image, bytes, str]) -> Image.Image:
        """Convert various image formats to PIL Image."""
        if isinstance(image, Image.Image):
            return image
        elif isinstance(image, bytes):
            return Image.open(io.BytesIO(image))
        elif isinstance(image, str):
            # Assume base64 encoded
            if image.startswith('data:image'):
                # Remove data URL prefix
                image = image.split(',')[1]
            image_data = base64.b64decode(image)
            return Image.open(io.BytesIO(image_data))
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")


class SentenceTransformerCLIP:
    """
    Alternative CLIP implementation using sentence-transformers.
    More efficient for batch processing.
    """
    
    def __init__(self):
        self.embedding_dim = 512
    
    def embed_image(self, image: Union[Image.Image, bytes, str]) -> List[float]:
        """Generate embedding for a single image."""
        pil_image = self._to_pil_image(image)
        model = get_sentence_transformer()
        embedding = model.encode(pil_image)
        return embedding.tolist()
    
    def embed_images(self, images: List[Union[Image.Image, bytes, str]]) -> List[List[float]]:
        """Batch encode images efficiently."""
        pil_images = [self._to_pil_image(img) for img in images]
        model = get_sentence_transformer()
        embeddings = model.encode(pil_images)
        return [emb.tolist() for emb in embeddings]
    
    def embed_text(self, text: str) -> List[float]:
        """Generate CLIP text embedding."""
        model = get_sentence_transformer()
        embedding = model.encode(text)
        return embedding.tolist()
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Batch encode texts efficiently."""
        model = get_sentence_transformer()
        embeddings = model.encode(texts)
        return [emb.tolist() for emb in embeddings]
    
    def _to_pil_image(self, image: Union[Image.Image, bytes, str]) -> Image.Image:
        """Convert various image formats to PIL Image."""
        if isinstance(image, Image.Image):
            return image.convert('RGB')
        elif isinstance(image, bytes):
            return Image.open(io.BytesIO(image)).convert('RGB')
        elif isinstance(image, str):
            if image.startswith('data:image'):
                image = image.split(',')[1]
            image_data = base64.b64decode(image)
            return Image.open(io.BytesIO(image_data)).convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")


# Default CLIP embeddings instance (lazy loaded)
def get_clip_embeddings(use_sentence_transformer: bool = True) -> Union[CLIPEmbeddings, SentenceTransformerCLIP]:
    """
    Get CLIP embeddings instance.
    
    Args:
        use_sentence_transformer: Use sentence-transformers (more efficient for batches)
    
    Returns:
        CLIP embeddings instance
    """
    if use_sentence_transformer:
        return SentenceTransformerCLIP()
    return CLIPEmbeddings()


# Convenience functions
def embed_image(image: Union[Image.Image, bytes, str]) -> List[float]:
    """Generate CLIP embedding for an image."""
    return get_clip_embeddings().embed_image(image)


def embed_image_query(query: str) -> List[float]:
    """Generate CLIP text embedding for image retrieval."""
    return get_clip_embeddings().embed_text(query)
