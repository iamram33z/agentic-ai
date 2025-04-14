# Import necessary Libraries
from agno.tools.yfinance import YFinanceTools
from .base import BaseAgent
from fintech_ai_bot.config import settings
from fintech_ai_bot.utils import get_logger

logger = get_logger(__name__)

class FinancialAgent(BaseAgent):
    """Agent responsible for fetching financial market data using YFinanceTools."""

    def __init__(self):
        """Initializes the FinancialAgent with YFinanceTools."""
        try:
            yfinance_tools = YFinanceTools(
                stock_price=True,
                stock_fundamentals=False, # Keep false by default
                key_financial_ratios=True,
                analyst_recommendations=False # Keep false by default
            )
            logger.debug("YFinanceTools instance created for FinancialAgent using boolean flags.")
        except TypeError as te:
            # Add specific check if the flags are still invalid with current agno version
            logger.critical(f"TypeError initializing YFinanceTools. Flags might be incorrect for current agno version: {te}", exc_info=True)
            raise RuntimeError("Could not initialize financial tools due to invalid flags") from te
        except Exception as e:
            logger.critical(f"Failed to initialize YFinanceTools: {e}", exc_info=True)
            raise RuntimeError("Could not initialize financial tools") from e

        # Pass the initialized tool(s) to the base class
        super().__init__(model_config=settings.financial_agent_model, tools=[yfinance_tools])
        logger.info("FinancialAgent initialized.")

    @staticmethod
    def get_agent_name() -> str:
        return "Financial Analyst"

    @staticmethod
    def get_agent_role() -> str:
        # Reverted Role
        return "Fetch key financial metrics for specific stock symbols. Handle cases where metrics (like P/E) might not apply (e.g., crypto) by stating 'N/A'."

    @staticmethod
    def get_agent_instructions() -> str:
        # Asking the model to generate the final table directly.
        return """Provide key financial data for the requested symbol(s) in a *compact* markdown table. Include ONLY these metrics:
| Metric          | Value      |
|-----------------|------------|
| Current Price   |            |
| P/E Ratio (TTM) |            |
| Market Cap      |            |
| 52-Wk Range     | Low - High |
| Beta            |            |

Format Market Cap clearly (e.g., $1.2T, $500B, $10B, $500M). Format 52-Wk Range like '100.50 - 250.75'. If a specific data point (like P/E) is unavailable or not applicable for the asset type, state 'N/A'. If the symbol is not found or all data is unavailable, return ONLY the text: 'Financial data not available for [symbol]'."""