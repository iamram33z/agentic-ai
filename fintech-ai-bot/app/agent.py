import os
import json
from typing import Dict, Optional, List, Union
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from azure_postgres import AzurePostgresClient
from faiss_db import FAISSVectorStore
from utils import get_logger, format_markdown_response
import re
from datetime import datetime

load_dotenv()
logger = get_logger("FinancialAgents")


class FinancialAgents:
    def __init__(self):
        """Initialize financial agents system with comprehensive error handling"""
        try:
            self.db = AzurePostgresClient()
            self.vector_store = FAISSVectorStore()
            self.agents = self._initialize_agents()
            logger.info("FinancialAgents system initialized successfully")
        except Exception as e:
            logger.critical(f"Failed to initialize FinancialAgents: {str(e)}", exc_info=True)
            raise RuntimeError("Failed to initialize financial agents system") from e

    def _initialize_agents(self) -> Dict[str, Agent]:
        """Initialize all specialized agents with proper configurations"""
        try:
            # Model configuration - optimized for each agent's role
            model_config = {
                'news': {
                    'id': "llama3-8b-8192",
                    'temperature': 0.3
                },
                'financial': {
                    'id': "llama-3.3-70b-versatile",
                    'temperature': 0.1
                },
                'recommendation': {
                    'id': "llama3-70b-8192",
                    'temperature': 0.2
                },
                'coordinator': {
                    'id': "llama-3.3-70b-versatile",
                    'temperature': 0.2
                }
            }

            agents_config = {
                'news': {
                    'name': "News Analyst",
                    'role': "Get top 3 market-moving news items",
                    'model': Groq(id=model_config['news']['id']),
                    'tools': [DuckDuckGoTools()],
                    'instructions': """Format:
1. [Date] Headline (Source)
2. Sentiment: Positive/Neutral/Negative
3. Key point (1 sentence)""",
                    'markdown': True
                },
                'financial': {
                    'name': "Financial Analyst",
                    'role': "Get key financial metrics",
                    'model': Groq(id=model_config['financial']['id']),
                    'tools': [YFinanceTools(
                        stock_price=True,
                        stock_fundamentals=True,
                        key_financial_ratios=True,
                        analyst_recommendations=True
                    )],
                    'instructions': """Show data in this table format:
| Metric       | Value |
|--------------|-------|
| Price        |       |
| P/E Ratio    |       |
| 52-Week High |       |
| 52-Week Low  |       |""",
                    'markdown': True
                },
                'recommendation': {
                    'name': "Investment Advisor",
                    'role': "Provide concise recommendations",
                    'model': Groq(id=model_config['recommendation']['id']),
                    'instructions': """Provide:
1. Short-term outlook (1 sentence)
2. Long-term outlook (1 sentence)
3. Key risk (1 bullet point)""",
                    'markdown': True
                }
            }

            agents = {
                key: Agent(**config) for key, config in agents_config.items()
            }

            # Coordinator Agent
            agents['coordinator'] = Agent(
                name="Research Coordinator",
                role="Combine key insights",
                model=Groq(id=model_config['coordinator']['id']),
                team=list(agents.values()),
                instructions="""Create comprehensive report with:
1. Top News (3 items max)
2. Financial Snapshot (table)
3. Recommendation Summary
4. Portfolio Analysis (if available)""",
                markdown=True
            )

            logger.info("All specialized agents initialized successfully")
            return agents

        except Exception as e:
            logger.error(f"Agent initialization failed: {str(e)}", exc_info=True)
            raise RuntimeError("Failed to initialize specialized agents") from e

    def get_response(self, query: str, client_id: str = None, client_context: dict = None) -> str:
        """
        Get personalized financial analysis response
        Args:
            query: User's financial question
            client_id: Client identifier (optional)
            client_context: Pre-loaded client portfolio data (optional)
        Returns:
            Formatted markdown response
        """
        try:
            # Validate input
            if not query or not isinstance(query, str):
                raise ValueError("Invalid query format")

            logger.info(f"Processing query for client {client_id or 'generic'}: {query[:100]}...")

            # Extract client context from query if not provided
            if not client_context and "My portfolio:" in query:
                try:
                    portfolio_part = query.split("My portfolio:")[1].split("\n\n")[0]
                    client_context = json.loads(portfolio_part.strip())
                    query = query.split("\n\n")[-1]  # Keep only the actual question
                except json.JSONDecodeError:
                    logger.warning("Failed to parse portfolio from query")

            # Get enhanced context
            context = self._get_enhanced_context(query, client_id, client_context)

            # Build comprehensive prompt
            prompt = self._build_prompt(context)

            # Get response from coordinator
            response = self.agents['coordinator'].print_response(prompt, stream=True)

            # Format and validate response
            formatted_response = format_markdown_response(response)
            if not formatted_response or not isinstance(formatted_response, str):
                raise ValueError("Invalid response format from agent")

            return formatted_response

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}", exc_info=True)
            return self._generate_error_response(e)

    def _get_enhanced_context(self, query: str, client_id: str = None, client_context: dict = None) -> dict:
        """Build comprehensive context for the query"""
        context = {
            'client': {
                'id': client_id,
                'portfolio': None,
                'risk_profile': None,
                'portfolio_value': 0
            },
            'market_data': None,
            'relevant_documents': None,
            'query': query
        }

        # Process client context (priority to provided context over DB)
        if client_context and isinstance(client_context, dict):
            self._process_client_context(context, client_context)
        elif client_id:
            self._fetch_client_context_from_db(context, client_id)

        # Get relevant documents from vector store
        context['relevant_documents'] = self.vector_store.search(query)

        # Extract and fetch market data for mentioned symbols
        symbols = self._extract_symbols(query)
        if symbols:
            context['market_data'] = self._get_market_data(symbols[:5])  # Limit to 5 symbols

        return context

    def _process_client_context(self, context: dict, client_context: dict):
        """Process provided client context"""
        try:
            if 'holdings' in client_context:
                context['client']['portfolio'] = [
                    {
                        'symbol': h.get('symbol'),
                        'value': h.get('value', 0),
                        'allocation': h.get('allocation', 0)
                    } for h in client_context.get('holdings', []) if h.get('symbol')
                ]
                context['client']['portfolio_value'] = client_context.get('portfolio_value', 0)

            if 'risk_profile' in client_context:
                context['client']['risk_profile'] = client_context['risk_profile']
        except Exception as e:
            logger.warning(f"Error processing client context: {str(e)}")

    def _fetch_client_context_from_db(self, context: dict, client_id: str):
        """Fetch client context from database"""
        try:
            portfolio = self.db.get_client_portfolio(client_id)
            if portfolio:
                holdings = portfolio.get('holdings', [])
                context['client'].update({
                    'portfolio': [
                        {
                            'symbol': h.get('symbol'),
                            'value': h.get('current_value', 0),
                            'allocation': (h.get('current_value', 0) / portfolio['total_value'] * 100
                                           if portfolio.get('total_value', 0) > 0 else 0)
                        } for h in holdings if h.get('symbol')
                    ],
                    'risk_profile': portfolio.get('risk_profile', 'Not specified'),
                    'portfolio_value': portfolio.get('total_value', 0)
                })
        except Exception as e:
            logger.warning(f"Failed to fetch portfolio for {client_id}: {str(e)}")

    def _get_market_data(self, symbols: List[str]) -> List[str]:
        """Get market data for multiple symbols"""
        market_data = []
        for symbol in symbols:
            try:
                data = self.agents['financial'].run(f"Get summary data for {symbol}")
                if isinstance(data, str):
                    market_data.append(data)
                else:
                    market_data.append(str(data))
            except Exception as e:
                logger.warning(f"Failed to get market data for {symbol}: {str(e)}")
        return market_data

    def _build_prompt(self, context: dict) -> str:
        """Construct comprehensive prompt with context"""
        prompt_parts = []

        # Add client context if available
        if context['client']['portfolio'] or context['client']['id']:
            prompt_parts.append("**Client Context**")
            if context['client']['id']:
                prompt_parts.append(f"- Client ID: {context['client']['id']}")
            if context['client']['risk_profile']:
                prompt_parts.append(f"- Risk Profile: {context['client']['risk_profile']}")
            if context['client']['portfolio']:
                prompt_parts.extend([
                    f"- Portfolio Value: ${context['client']['portfolio_value']:,.2f}",
                    "**Current Holdings**:",
                    *[f"  - {h['symbol']}: {h.get('allocation', 0):.1f}% (${h.get('value', 0):,.2f})"
                      for h in context['client']['portfolio']]
                ])

        # Add the cleaned query
        prompt_parts.extend([
            "",
            f"**Question**: {context['query']}",
            ""
        ])

        # Add relevant documents if available
        if context['relevant_documents']:
            prompt_parts.extend([
                "**Additional Context**:",
                context['relevant_documents']
            ])

        # Add market data if available
        if context['market_data']:
            prompt_parts.extend([
                "",
                "**Market Data**:",
                *[f"- {data}" for data in context['market_data']]
            ])

        return "\n".join(prompt_parts)

    def _extract_symbols(self, text: str) -> List[str]:
        """Enhanced symbol extraction with multiple patterns"""
        # Pattern 1: Standard tickers (1-5 uppercase letters)
        ticker_pattern = r'\b[A-Z]{1,5}\b'
        # Pattern 2: Tickers with $ prefix ($AAPL)
        dollar_pattern = r'\$([A-Z]{1,5})\b'
        # Pattern 3: Common ticker mentions (e.g., "AAPL stock")
        mention_pattern = r'\b([A-Z]{1,5})\s+(stock|shares|equity)\b'

        symbols = set()
        for pattern in [ticker_pattern, dollar_pattern, mention_pattern]:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                symbol = match[0] if isinstance(match, tuple) else match
                symbols.add(symbol.upper())

        # Filter out common false positives
        common_words = {'THE', 'AND', 'FOR', 'YOU', 'ARE', 'THIS', 'WITH', 'YOUR'}
        return [s for s in symbols if s not in common_words]

    def _generate_error_response(self, error: Exception) -> str:
        """Generate user-friendly error response based on error type"""
        error_messages = {
            ValueError: "⚠️ Please provide a valid financial question",
            ConnectionError: "🔌 Connection error - please try again later",
            TimeoutError: "⏱️ Request timed out - please try a simpler query",
            Exception: "❌ An error occurred while processing your request"
        }
        return error_messages.get(type(error), error_messages[Exception])


def test_agents():
    """Test function to verify all agents are working"""
    try:
        print("Initializing FinancialAgents...")
        agents = FinancialAgents()

        print("\n🧪 Testing Financial Agent...")
        agents.agents['financial'].print_response("Give financial metrics of Tesla", stream=True)

        print("\n🧪 Testing News Agent...")
        try:
            response = agents.agents['news'].print_response("Top market-moving news about Tesla", stream=True)
            print(response)
        except Exception as e:
            print(f"❌ Test failed: {str(e)}")

        print("\n🧪 Testing Recommendation Agent...")
        agents.agents['recommendation'].print_response("Give outlook of Tesla", stream=True)

        print("\n🚀 Running Coordinator Agent...")
        agents.agents['coordinator'].print_response(
            "Provide analysis of Tesla's market position",
            stream=True
        )
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")


if __name__ == "__main__":
    test_agents()