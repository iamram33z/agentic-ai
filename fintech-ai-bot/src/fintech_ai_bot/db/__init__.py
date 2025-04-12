# src/fintech_ai_bot/db/__init__.py

# Expose the primary database client class

from .postgres_client import PostgresClient

__all__ = [
    "PostgresClient",
]