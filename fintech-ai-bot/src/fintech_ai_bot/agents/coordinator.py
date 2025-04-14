# Import necessary Libraries
from typing import List, Optional
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
                team=self.agno_team,
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
        return "Synthesize analysis from various inputs (client context, summarized market data/news, documents) into a final, user-friendly report answering the user's query."

    @staticmethod
    def get_agent_instructions() -> str:
        # Financial Data Instruction
        return """Synthesize the provided information into a coherent report answering the user's query. Use clear MARKDOWN formatting.

Structure:
1.  **Executive Summary:** (1-2 sentences) Directly answer the user's core question (e.g., buy/sell opinion, analysis summary) based *only* on the provided inputs.
2.  **Analysis Context:**
    * **Market News:** Briefly mention relevant news from the provided summary, or state 'No significant relevant news found or retrieval failed.'
    * **Financial Data:**
        - If financial data (markdown tables or error messages like '⚠️ Data not available...' or '⚠️ Tool execution failed...') was retrieved for specific symbols, present this information clearly under a separate subheading for *each* symbol.
        - Directly include the markdown table or the exact error message as provided in the context for each symbol. Do NOT summarize the data into a paragraph.
        - Example Format:
            ```markdown
            **Financial Data:**

            * **AMD:**
                | Metric          | Value      |
                |-----------------|------------|
                | Current Price   | $165.00    |
                | P/E Ratio (TTM) | 35.6       |
                | ...             | ...        |

            * **GLD:**
                ⚠️ Data not available for symbol: GLD

            * **BND:**
                ⚠️ Tool execution failed for BND
            ```
        - If *no* financial data (tables or errors) was provided in the context for any symbols, state 'No specific financial data was requested or retrieved for this query.'
    * **Knowledge Base:** Briefly mention relevant insights from the provided document context, if any, including source/type. If retrieval failed or no documents were found, state 'No relevant information found in the knowledge base' or 'Error retrieving documents from knowledge base' based on the context provided.
3.  **Client Portfolio Context** (If client data was provided):
    * Briefly relate the analysis to the client's holdings (e.g., "Your portfolio holds X% in NVDA"). Mention risk profile if relevant. Use only the provided holdings list.
4.  **Recommendation & Risks:** (Optional, synthesize if sufficient data)
    * Synthesize insights from the analysis. State outlook (short/long term) concisely if possible.
    * Clearly state key risks identified *from the provided context*. If insufficient data, state 'Insufficient data for specific recommendations.'
5.  **Disclaimer:** Include the standard disclaimer: 'This information is for informational purposes only and not investment advice.'

**IMPORTANT**: Rely *strictly* on the information given in the prompt sections (Client Info, User Query, News Summary, Market Data, Knowledge Base Context). Do not infer or add external knowledge. Present the financial data exactly as provided in the context under symbol subheadings. Be objective and concise."""