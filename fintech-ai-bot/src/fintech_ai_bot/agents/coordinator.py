# src/fintech_ai_bot/agents/coordinator.py
# Consistent imports and logger

from typing import List, Optional
from agno.tools.toolkit import Toolkit # Use Toolkit for consistency
from fintech_ai_bot.utils import get_logger
from .base import BaseAgent, AgnoAgent
from fintech_ai_bot.config import settings

logger = get_logger(__name__)

class CoordinatorAgent(BaseAgent):
    """
    Agent responsible for synthesizing information from various sources
    (other agents, documents, client context) and generating the final
    user-facing response according to a defined structure.
    """

    def __init__(self, team: Optional[List[BaseAgent]] = None):
        """
        Initializes the CoordinatorAgent.
        Args:
            team: An optional list of other BaseAgent instances.
        """
        self.agno_team = [agent._agno_agent for agent in team] if team else None
        super().__init__(model_config=settings.coordinator_agent_model, tools=None)
        team_names = ', '.join([agent.name for agent in team]) if team else "no members"
        logger.info(f"CoordinatorAgent initialized with team: [{team_names}].")

    def _create_agno_agent(self) -> AgnoAgent:
        """Creates the underlying Agno agent instance, including the team."""
        try:
            agno_instance = AgnoAgent(
                name=self.name,
                role=self.role,
                model=self.model,
                tools=self.tools,
                team=self.agno_team, # Pass the list of other Agno agents
                instructions=self.instructions,
                markdown=True
            )
            logger.debug(f"Agno agent instance created for '{self.name}' {'with' if self.agno_team else 'without'} a team.")
            return agno_instance
        except Exception as e:
            logger.critical(f"Failed to create Agno agent instance for '{self.name}': {e}", exc_info=True)
            raise RuntimeError(f"Agno agent instance creation failed for '{self.name}'") from e

    @staticmethod
    def get_agent_name() -> str:
        return "Research Coordinator"

    @staticmethod
    def get_agent_role() -> str:
        # Role from original agent.py
        return "Synthesize analysis from various inputs (client context, summarized market data/news, documents) into a final, user-friendly report answering the user's query."

    @staticmethod
    def get_agent_instructions() -> str:
        # Instructions from original agent.py
        return """Synthesize the provided information into a coherent report answering the user's query. Use clear MARKDOWN formatting.

Structure:
1.  **Executive Summary:** (1-2 sentences) Directly answer the user's core question (e.g., buy/sell opinion, analysis summary) based *only* on the provided inputs.
2.  **Analysis Context:**
    * **Market News:** Briefly mention relevant news from the provided summary, or state 'No significant relevant news found.'
    * **Financial Data:** Summarize key highlights from the provided financial data tables/messages for the symbols discussed. Mention any symbols where data retrieval failed.
    * **Knowledge Base:** Briefly mention relevant insights from the provided document context, if any.
3.  **Client Portfolio Context** (If client data was provided):
    * Briefly relate the analysis to the client's holdings (e.g., "Your portfolio holds X% in NVDA"). Mention risk profile if relevant. Use only the provided holdings list.
4.  **Recommendation & Risks:** (Optional, if sufficient data)
    * Synthesize insights from the analysis. State outlook (short/long term) concisely if possible.
    * Clearly state key risks identified *from the provided context*. If based on the recommendation agent's input, integrate it smoothly. If insufficient data, state 'Insufficient data for specific recommendations.'
5.  **Disclaimer:** Include the standard disclaimer: 'This information is for informational purposes only and not investment advice.'

**IMPORTANT**: Rely *strictly* on the information given in the prompt sections (Client Info, User Query, News Summary, Market Data, Knowledge Base). Do not infer or add external knowledge. Be objective and concise."""