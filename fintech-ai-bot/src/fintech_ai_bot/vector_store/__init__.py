# src/fintech_ai_bot/vector_store/__init__.py

# Expose the primary vector store client class

from .faiss_client import FAISSClient

__all__ = [
    "FAISSClient",
]