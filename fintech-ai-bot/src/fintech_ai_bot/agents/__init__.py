# src/fintech_ai_bot/agents/__init__.py

# Expose key agent classes for easier importing

from .base import BaseAgent
from .financial import FinancialAgent
from .news import NewsAgent
from .coordinator import CoordinatorAgent
# Add other specific agents if created (e.g., RecommendationAgent)

__all__ = [
    "BaseAgent",
    "FinancialAgent",
    "NewsAgent",
    "CoordinatorAgent",
    # Add other agent class names here
]