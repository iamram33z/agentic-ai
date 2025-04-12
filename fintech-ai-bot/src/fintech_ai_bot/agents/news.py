# src/fintech_ai_bot/agents/news.py
# Corrected with Toolkit import and logger

from typing import List, Optional
from agno.tools.toolkit import Toolkit # Use Toolkit
from agno.tools.duckduckgo import DuckDuckGoTools
from .base import BaseAgent
from fintech_ai_bot.config import settings
from fintech_ai_bot.utils import get_logger

logger = get_logger(__name__)

class NewsAgent(BaseAgent):
    """Agent responsible for retrieving and summarizing market news using DuckDuckGoTools."""

    def __init__(self):
        """Initializes the NewsAgent with DuckDuckGoTools."""
        try:
            # Initialize DuckDuckGoTools without arguments
            duckduckgo_tools = DuckDuckGoTools()
            logger.debug("DuckDuckGoTools instance created for NewsAgent.")
        except Exception as e:
            logger.critical(f"Failed to initialize DuckDuckGoTools: {e}", exc_info=True)
            raise RuntimeError("Could not initialize news tools") from e

        super().__init__(model_config=settings.news_agent_model, tools=[duckduckgo_tools])
        logger.info("NewsAgent initialized.")

    @staticmethod
    def get_agent_name() -> str:
        return "News Analyst"

    @staticmethod
    def get_agent_role() -> str:
        # Role from original agent.py
        return "Retrieve and concisely summarize the top 3 most relevant market news items based on the user query and provided context (like stock symbols). Focus on market impact."

    @staticmethod
    def get_agent_instructions() -> str:
        # Instructions from original agent.py
        # Assumes the tool function name exposed is 'duckduckgo_search'
        return """Provide ONLY the top 3 most relevant news items MAXIMUM. Be extremely concise. Format each item as:
1. Headline (Source) - Sentiment: [Positive/Neutral/Negative]
2. Key Point: [One short sentence summarizing the core information and its potential market impact.]

If no highly relevant news is found, return ONLY the text: 'No significant market-moving news found relevant to the query.' Do not add any extra commentary or formatting."""