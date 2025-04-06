# agent.py
import os
import json
from typing import Dict, Optional, List, Union
from dotenv import load_dotenv
from agno.agent import Agent
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from azure_postgres import AzurePostgresClient
from faiss_db import FAISSVectorStore
from agno.models.groq import Groq
from agno.exceptions import ModelProviderError
from agno.run.response import RunResponse
# Assuming utils.py is in the same directory or PYTHONPATH
from utils import (
    get_logger,
    format_markdown_response,
    summarize_financial_data,
    estimate_token_count,
    generate_error_html,
    generate_warning_html,  # Import warning html
    log_execution_time,
    validate_portfolio_data, validate_query_text
)
import re
import requests
from datetime import datetime
import traceback # For detailed error logging
import time # Import time

load_dotenv()
# Use the configured logger from utils
# Note: Logger name consistency helps if multiple modules use the same name
logger = get_logger("agent")

# Define constants for context limits (adjust as needed)
MAX_HOLDINGS_IN_PROMPT = 10
MAX_DOC_TOKENS_IN_PROMPT = 1000 # Tokens for document context
MAX_FINANCIAL_DATA_SUMMARY_LEN = 400 # Characters per symbol summary
MAX_NEWS_SUMMARY_LEN = 600 # Characters for overall news summary


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
            # Re-raise as a runtime error to be caught by the caller
            raise RuntimeError(f"Failed to initialize financial agents system") from e

    def _initialize_agents(self) -> Dict[str, Agent]:
        """Initialize all specialized agents with proper configurations"""
        try:
            # Model configuration (check specific model context limits and capabilities)
            # Using Llama3.1 70B as an example - ensure it's available via Groq
            # Context window size is crucial for the coordinator
            MODEL_ID_LLAMA3_3_70B = "llama-3.3-70b-versatile" # Example ID
            MODEL_ID_LLAMA3_8B = "llama-3.1-8b-instant"     # Faster, smaller context alternative

            model_config = {
                 'news': {
                     'id': MODEL_ID_LLAMA3_8B, # Use faster model for news summary
                     'temperature': 0.4,
                     'max_tokens': 800 # Limit output size
                 },
                 'financial': {
                     'id': MODEL_ID_LLAMA3_8B, # Use faster model for data extraction
                     'temperature': 0.1,
                     'max_tokens': 800
                 },
                 'recommendation': {
                     'id': MODEL_ID_LLAMA3_8B, # Faster model
                     'temperature': 0.3,
                     'max_tokens': 500
                 },
                 'coordinator': {
                     'id': MODEL_ID_LLAMA3_3_70B, # Use powerful model for coordination
                     'temperature': 0.2,
                     'max_tokens': 4000 # Max output tokens (input context is the primary limit)
                 }
            }

            # Define tools
            yfinance_tools = YFinanceTools(
                stock_price=True,
                stock_fundamentals=False, # Keep false by default
                key_financial_ratios=True,
                analyst_recommendations=False # Keep false by default
            )
            # Corrected DuckDuckGoTools initialization
            duckduckgo_tools = DuckDuckGoTools()


            # Agent Configurations
            agents_config = {
                'news': {
                    'name': "News Analyst",
                    'role': "Retrieve and concisely summarize the top 3 most relevant market news items based on the user query and provided context (like stock symbols). Focus on market impact.",
                    'model': Groq(id=model_config['news']['id'],
                                  temperature=model_config['news']['temperature'],
                                  max_tokens=model_config['news']['max_tokens']),
                    'tools': [duckduckgo_tools],
                    'instructions': """Provide ONLY the top 3 most relevant news items MAXIMUM. Be extremely concise. Format each item as:
1. Headline (Source) - Sentiment: [Positive/Neutral/Negative]
2. Key Point: [One short sentence summarizing the core information and its potential market impact.]

If no highly relevant news is found, return ONLY the text: 'No significant market-moving news found relevant to the query.' Do not add any extra commentary or formatting.""",
                    'markdown': True
                },
                'financial': {
                    'name': "Financial Analyst",
                    'role': "Fetch key financial metrics for specific stock symbols.",
                    'model': Groq(id=model_config['financial']['id'],
                                  temperature=model_config['financial']['temperature'],
                                  max_tokens=model_config['financial']['max_tokens']),
                    'tools': [yfinance_tools],
                    'instructions': """Provide key financial data for the requested symbol(s) in a *compact* markdown table. Include ONLY these metrics:
| Metric          | Value      |
|-----------------|------------|
| Current Price   |            |
| P/E Ratio (TTM) |            |
| Market Cap      |            |
| 52-Wk Range     | Low - High |
| Beta            |            |

Format Market Cap clearly (e.g., $1.2T, $500B, $10B, $500M). Format 52-Wk Range like '100.50 - 250.75'. If a specific data point (like P/E) is unavailable, state 'N/A'. If the symbol is not found or all data is unavailable, return ONLY the text: 'Financial data not available for [symbol]'.""",
                    'markdown': True
                },
                'recommendation': {
                    'name': "Investment Advisor",
                    'role': "Provide *brief* investment outlook and key risks based ONLY on the provided analysis context.",
                    'model': Groq(id=model_config['recommendation']['id'],
                                  temperature=model_config['recommendation']['temperature'],
                                  max_tokens=model_config['recommendation']['max_tokens']),
                    'instructions': """Based ONLY on the financial data and news context provided to you, give a VERY concise outlook (1 sentence each for short-term and long-term) and list ONE primary risk.

Format:
Short-term Outlook: [Sentence]
Long-term Outlook: [Sentence]
Key Risk: [Bullet point]

If data is insufficient for a meaningful recommendation, state ONLY: 'Insufficient data for specific recommendation.' DO NOT HALLUCINATE or use external knowledge.""",
                    'markdown': True
                }
            }

            agents = {
                key: Agent(**config) for key, config in agents_config.items()
            }

            # Coordinator Agent
            agents['coordinator'] = Agent(
                name="Research Coordinator",
                role="Synthesize analysis from various inputs (client context, summarized market data/news, documents) into a final, user-friendly report answering the user's query.",
                model=Groq(id=model_config['coordinator']['id'],
                           temperature=model_config['coordinator']['temperature'],
                           max_tokens=model_config['coordinator']['max_tokens']),
                team=list(agents.values()), # Team definition might still be useful for some agent frameworks
                instructions="""Synthesize the provided information into a coherent report answering the user's query. Use clear MARKDOWN formatting.

Structure:
1.  **Executive Summary:** (1-2 sentences) Directly answer the user's core question (e.g., buy/sell opinion, analysis summary) based *only* on the provided inputs.
2.  **Analysis Context:**
    * **Market News:** Briefly mention relevant news from the provided summary, or state 'No significant relevant news found.'
    * **Financial Data:** Summarize key highlights from the provided financial data tables for the symbols discussed.
    * **Knowledge Base:** Briefly mention relevant insights from the provided document context, if any.
3.  **Client Portfolio Context** (If client data was provided):
    * Briefly relate the analysis to the client's holdings (e.g., "Your portfolio holds X% in NVDA"). Mention risk profile if relevant. Use only the provided holdings list.
4.  **Recommendation & Risks:** (Optional, if sufficient data)
    * Synthesize insights from the analysis. State outlook (short/long term) concisely if possible.
    * Clearly state key risks identified *from the provided context*. If based on the recommendation agent's input, integrate it smoothly. If insufficient data, state 'Insufficient data for specific recommendations.'
5.  **Disclaimer:** Include the standard disclaimer: 'This information is for informational purposes only and not investment advice.'

**IMPORTANT**: Rely *strictly* on the information given in the prompt sections (Client Info, User Query, News Summary, Market Data, Knowledge Base). Do not infer or add external knowledge. Be objective and concise.""",
                markdown=True
            )

            logger.info("All specialized agents initialized successfully")
            return agents

        except Exception as e:
            logger.error(f"Agent configuration failed during initialization: {str(e)}", exc_info=True)
            raise RuntimeError("Failed to initialize specialized agents") from e

    @log_execution_time
    def get_response(self, query: str, client_id: str = None, client_context: dict = None) -> str:
        """
        Get personalized financial analysis response. Manages context, calls sub-agents, coordinates, and handles errors.
        """
        prompt_string = "" # Ensure prompt_string is defined in outer scope for error logging
        try:
            if not validate_query_text(query): # Use validation util
                logger.warning(f"Invalid query received: '{query}'")
                # Use generate_warning_html or similar from utils if available
                return generate_warning_html("Invalid Query", "Please provide a more specific question (3-1500 characters).")

            logger.info(f"Processing query for client '{client_id or 'generic'}': '{query[:100]}...'")

            # Prepare client context (fetches/validates if needed)
            processed_client_context = self._prepare_client_context(client_id, client_context)

            # Get enhanced context (calls sub-agents, summarizes, truncates)
            context_dict = self._get_enhanced_context(query, processed_client_context)

            # Build the final prompt string for the coordinator
            prompt_string = self._build_prompt(context_dict)

            # Estimate token count and log warning if high
            estimated_tokens = estimate_token_count(prompt_string)
            logger.info(f"Estimated token count for coordinator prompt: ~{estimated_tokens}")
            # Adjust threshold based on the coordinator model's actual limit (e.g., 8k, 32k, 128k)
            # Example for an 8k model limit:
            if estimated_tokens > 7800: # Leave some buffer
                 logger.warning(f"Prompt token estimate ({estimated_tokens}) is very high, approaching context limit (~8k).")
                 # Optionally, return a warning immediately or try to shorten further if possible

            # Call the coordinator agent
            response = self._get_coordinator_response(prompt_string)

            # Format and return the final response
            return self._format_final_response(response)

        # --- Error Handling ---
        except ModelProviderError as e:
            error_details = str(e)
            logger.error(f"Model provider error during response generation: {error_details}", exc_info=True)
            # Specific handling for context length exceeded
            if "context_length_exceeded" in error_details.lower() or (hasattr(e, 'code') and e.code == 400): # Check code if available
                final_token_estimate = estimate_token_count(prompt_string) # Recalculate for log msg
                logger.error(f"Context Length Exceeded for coordinator. Final estimated tokens: ~{final_token_estimate}")
                return generate_error_html(
                    "Request Too Complex",
                    f"The analysis generated too much context (~{final_token_estimate} tokens) for the AI model. Please try simplifying your request (e.g., fewer stocks, more specific question)."
                 )
            else: # Other model errors (rate limits, API issues)
                logger.error(f"Failed Prompt Snippet (Model Error):\n{prompt_string[:500]}...")
                return generate_error_html("AI Model Error", f"The AI model encountered an issue. Please try again. Details: {error_details}")
        except (RuntimeError, ValueError) as e: # Catch specific internal errors
             logger.error(f"Internal processing error in get_response: {str(e)}", exc_info=True)
             return generate_error_html("Processing Error", f"An internal error occurred while processing your request: {e}")
        except Exception as e: # Catch-all for truly unexpected errors
            logger.error(f"Unexpected critical error in get_response: {str(e)}", exc_info=True)
            logger.error(traceback.format_exc()) # Log full traceback
            return generate_error_html("Unexpected Error", f"An unexpected internal error occurred. Please contact support if this persists.")

    def _prepare_client_context(self, client_id: Optional[str], client_context_from_ui: Optional[dict]) -> Optional[dict]:
        """Loads, validates, and prepares client context dictionary."""
        logger.debug(f"Preparing client context. Provided ID: {client_id}, UI Context provided: {client_context_from_ui is not None}")
        final_context = None
        current_client_id = client_id

        # 1. Use UI context if valid
        if client_context_from_ui and isinstance(client_context_from_ui, dict):
            if validate_portfolio_data(client_context_from_ui):
                context_id = client_context_from_ui.get('id')
                logger.info(f"Using validated client context from UI for client: {context_id or 'ID_MISSING_IN_CONTEXT'}")
                final_context = client_context_from_ui
                current_client_id = context_id or current_client_id # Update ID
                if current_client_id and 'id' not in final_context:
                    final_context['id'] = current_client_id
            else:
                logger.warning("Invalid client context structure received from UI. Ignoring it.")
                current_client_id = client_context_from_ui.get('id', current_client_id) # Try to keep ID

        # 2. Fetch from DB if no valid UI context and client_id is available
        if final_context is None and current_client_id:
            logger.info(f"No valid UI context; fetching from DB for client ID: {current_client_id}")
            try:
                portfolio_data = self.db.get_client_portfolio(current_client_id)
                if portfolio_data and isinstance(portfolio_data, dict):
                    processed_data = self._process_raw_portfolio(current_client_id, portfolio_data)
                    if processed_data and validate_portfolio_data(processed_data):
                        logger.info(f"Successfully fetched and validated portfolio from DB for {current_client_id}")
                        final_context = processed_data
                    else:
                        logger.warning(f"Fetched/processed DB portfolio for {current_client_id} failed validation.")
                else:
                    logger.warning(f"No portfolio data found in DB for client {current_client_id}.")
            except Exception as e:
                logger.error(f"Failed to fetch/process portfolio from DB for {current_client_id}: {e}", exc_info=True)

        # 3. Final check and return
        if final_context:
            logger.debug(f"Client context prepared for ID: {final_context.get('id', 'MISSING_ID')}")
            # Ensure ID field consistency one last time
            if 'id' not in final_context and current_client_id: final_context['id'] = current_client_id
            elif final_context.get('id') != current_client_id:
                 logger.warning(f"Final context ID '{final_context.get('id')}' differs from derived client ID '{current_client_id}'.")
                 # Decide which ID takes precedence if they differ. Using context ID for now.
                 current_client_id = final_context.get('id')

        else: logger.debug("No client context available.")
        return final_context

    def _process_raw_portfolio(self, client_id: str, raw_portfolio: dict) -> Optional[dict]:
        """Processes raw portfolio data from DB into the standard context format."""
        try:
            # Structure definition matches validate_portfolio_data requirements
            portfolio = {
                "id": client_id,
                "name": f"Client {client_id[-4:]}" if len(client_id) >= 4 else f"Client {client_id}",
                "risk_profile": raw_portfolio.get('risk_profile', 'Not specified') or 'Not specified',
                "portfolio_value": 0.0,
                "holdings": [],
                "last_update": raw_portfolio.get('last_updated', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            }
            db_total_value = raw_portfolio.get('total_value')
            holdings_data = raw_portfolio.get('holdings', [])
            calculated_total_value = 0.0
            if isinstance(holdings_data, list):
                for h in holdings_data:
                     if isinstance(h, dict):
                        value = h.get('current_value', 0)
                        if isinstance(value, (int, float)): calculated_total_value += value
            # Determine final portfolio value
            if isinstance(db_total_value, (int, float)) and db_total_value > 0:
                 portfolio['portfolio_value'] = float(db_total_value)
            elif calculated_total_value >= 0:
                 portfolio['portfolio_value'] = calculated_total_value
            # Process holdings
            if isinstance(holdings_data, list):
                total_val = portfolio['portfolio_value']
                for holding in holdings_data:
                    if not isinstance(holding, dict) or not holding.get('symbol'): continue
                    symbol = str(holding['symbol']).strip().upper()
                    if not symbol: continue
                    value = holding.get('current_value', 0)
                    if not isinstance(value, (int, float)): value = 0.0
                    else: value = float(value)
                    allocation = (value / total_val * 100) if total_val > 0 else 0.0
                    portfolio['holdings'].append({"symbol": symbol, "value": value, "allocation": allocation})
            return portfolio
        except Exception as e:
             logger.error(f"Error processing raw portfolio for {client_id}: {e}", exc_info=True)
             return None

    @log_execution_time
    def _get_coordinator_response(self, formatted_prompt_string: str):
        """Sends the prompt string to the coordinator agent and returns the content."""
        coordinator = self.agents.get('coordinator')
        if not coordinator:
             logger.critical("Coordinator agent is not initialized!")
             raise RuntimeError("Coordinator agent not available.")
        logger.debug(f"Sending prompt to coordinator. Length: {len(formatted_prompt_string)} chars, Est. Tokens: ~{estimate_token_count(formatted_prompt_string)}")
        try:
            # Using run() for simpler handling, assuming final response doesn't need streaming
            response_obj = coordinator.run(formatted_prompt_string)
            if isinstance(response_obj, RunResponse):
                return response_obj.content # Extract content string
            else:
                 logger.warning(f"Coordinator agent .run() returned unexpected type: {type(response_obj)}")
                 return str(response_obj) # Fallback
        except ModelProviderError as e:
            logger.error(f"ModelProviderError calling coordinator: {e}", exc_info=True)
            raise # Re-raise to be handled by get_response
        except Exception as e:
            logger.error(f"Unexpected error calling coordinator: {e}", exc_info=True)
            logger.error(f"Failed Prompt Snippet (Coordinator):\n{formatted_prompt_string[:500]}...")
            raise RuntimeError(f"Coordinator agent failed unexpectedly: {e}") from e # Wrap original exception

    def _format_final_response(self, response: Optional[str]) -> str:
        """Formats the final response string."""
        if response is None:
            logger.warning("Coordinator returned None response.")
            return generate_error_html("Empty Response", "The AI advisor did not generate a response.")
        if not isinstance(response, str):
            logger.warning(f"Coordinator returned non-string type: {type(response)}. Converting.")
            response = str(response)
        formatted = format_markdown_response(response) # Use util function
        if not formatted.strip():
            logger.warning("Formatted response is empty.")
            return generate_warning_html("Empty Response", "The AI advisor generated an empty response after formatting.")
        return formatted

    @log_execution_time
    def _get_enhanced_context(self, query: str, client_context: Optional[dict]) -> dict:
        """Builds context dict: client data, summarized/truncated agent results, documents."""
        context_dict = {'client': client_context, 'market_data_summary': None, 'news_summary': None, 'relevant_documents': None, 'original_query': query}
        client_id_log = client_context.get('id', 'generic') if client_context else 'generic'
        # 1. Vector Store Search & Truncation
        try:
             retrieved_docs_text = self.vector_store.search(query)
             if retrieved_docs_text:
                  doc_tokens = estimate_token_count(retrieved_docs_text)
                  logger.info(f"Retrieved relevant documents (~{doc_tokens} tokens).")
                  if doc_tokens > MAX_DOC_TOKENS_IN_PROMPT:
                       logger.warning(f"Truncating relevant documents from ~{doc_tokens} to ~{MAX_DOC_TOKENS_IN_PROMPT} tokens.")
                       # Simple truncation (improve if possible)
                       chars_limit = MAX_DOC_TOKENS_IN_PROMPT * 4
                       context_dict['relevant_documents'] = retrieved_docs_text[:chars_limit].strip() + "..."
                  else: context_dict['relevant_documents'] = retrieved_docs_text
             else: logger.info("No relevant documents found.")
        except Exception as e:
             logger.error(f"Vector store search failed for client {client_id_log}: {e}", exc_info=True)
             context_dict['relevant_documents'] = "Error retrieving documents."
        # 2. Symbol Extraction
        portfolio_symbols = self._get_portfolio_symbols(client_context)
        combined_text = query + " " + portfolio_symbols
        symbols = self._extract_symbols(combined_text)
        symbols_to_fetch = symbols[:5] # Limit symbols
        # 3. Financial Data Fetching & Summarization
        if symbols_to_fetch:
            logger.info(f"Fetching market data for symbols: {symbols_to_fetch}")
            market_data_raw = self._get_market_data(symbols_to_fetch)
            summarized_market_data = [summarize_financial_data(r, MAX_FINANCIAL_DATA_SUMMARY_LEN) if isinstance(r, str) and not r.startswith("⚠️") else r for r in market_data_raw]
            context_dict['market_data_summary'] = summarized_market_data
            logger.debug(f"Summarized market data: {summarized_market_data}")
        else:
             logger.info("No symbols identified for market data fetching.")
             context_dict['market_data_summary'] = ["No specific stock symbols identified."]
        # 4. News Fetching & Summarization
        logger.info("Fetching news summary...")
        try:
            news_agent = self.agents.get('news')
            if news_agent:
                news_query = f"Top financial market news relevant to: {query}" + (f" (especially {', '.join(symbols_to_fetch)})" if symbols_to_fetch else "")
                news_response = news_agent.run(news_query) # Expects RunResponse
                if isinstance(news_response, RunResponse) and news_response.content:
                     content_str = str(news_response.content).strip()
                     if content_str and "No significant market-moving news found" not in content_str:
                          # Truncate news summary if it's too long despite instructions
                          if len(content_str) > MAX_NEWS_SUMMARY_LEN:
                               logger.warning(f"Truncating news summary from {len(content_str)} to {MAX_NEWS_SUMMARY_LEN} chars.")
                               context_dict['news_summary'] = content_str[:MAX_NEWS_SUMMARY_LEN].strip() + "..."
                          else:
                               context_dict['news_summary'] = content_str
                          logger.debug(f"News summary received: {context_dict['news_summary'][:100]}...")
                     else: logger.info("News agent returned no significant news.") # news_summary remains None
                else: logger.warning(f"News agent returned invalid/empty response: {type(news_response)}")
            else: logger.error("News agent not initialized.")
        except Exception as e:
            logger.error(f"News agent query failed for client {client_id_log}: {e}", exc_info=True)
            context_dict['news_summary'] = f"⚠️ Error retrieving news." # Keep it concise for prompt

        return context_dict

    def _get_portfolio_symbols(self, client_context: Optional[dict]) -> str:
        """Extracts symbols from portfolio context."""
        if client_context and client_context.get('holdings'):
            symbols = [str(h['symbol']) for h in client_context['holdings'] if isinstance(h, dict) and h.get('symbol')]
            return " ".join(symbols)
        return ""

    @log_execution_time
    def _get_market_data(self, symbols: List[str]) -> List[str]:
        """Gets market data using the financial agent, returns list of strings (results/errors)."""
        market_data_results = []
        financial_agent = self.agents.get('financial')
        if not financial_agent:
             logger.error("Financial agent not initialized.")
             return [f"⚠️ Error: Financial analysis module unavailable."] * len(symbols)
        for symbol in symbols:
            # Optional: time.sleep(0.1) # Small delay?
            try:
                logger.info(f"Requesting financial data for symbol: {symbol}")
                query = f"Get summary financial data table for {symbol}"
                response_obj = financial_agent.run(query) # Expect RunResponse
                if isinstance(response_obj, RunResponse) and response_obj.content:
                    content_str = str(response_obj.content).strip()
                    if f"Financial data not available for {symbol}" in content_str or not content_str:
                         logger.warning(f"Financial agent reported no data for {symbol}.")
                         market_data_results.append(f"⚠️ Data not available for symbol: {symbol}")
                    else:
                        prefix = f"**{symbol}**:\n" # Add symbol prefix for clarity
                        if not content_str.strip().startswith(("|", "**", symbol)):
                             market_data_results.append(prefix + content_str)
                        else: market_data_results.append(content_str)
                        logger.debug(f"Successfully retrieved data for {symbol}.")
                else:
                    logger.warning(f"Financial agent empty/invalid response for {symbol}: {type(response_obj)}")
                    market_data_results.append(f"⚠️ No valid data received for symbol: {symbol}")
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else 'N/A'
                user_msg = f"⚠️ Error retrieving data for {symbol} (Network Error: {status})"
                if status == 404: user_msg = f"⚠️ Data not available for symbol: {symbol} (Not Found)"
                elif status == 429: user_msg = f"⚠️ Error retrieving data for {symbol} (Rate Limit)"
                logger.warning(f"HTTP error for {symbol}: Status {status}", exc_info=True)
                market_data_results.append(user_msg)
            except ModelProviderError as e:
                 logger.error(f"ModelProviderError from financial agent for {symbol}: {e}", exc_info=True)
                 market_data_results.append(f"⚠️ Error retrieving data for {symbol} (AI Model Error)")
            except Exception as e:
                logger.error(f"Unexpected error getting data for {symbol}: {e}", exc_info=True)
                market_data_results.append(f"⚠️ Unexpected error retrieving data for {symbol}")
        return market_data_results

    @log_execution_time
    def _build_prompt(self, context_dict: dict) -> str:
        """Constructs the final prompt string for the coordinator agent."""
        prompt_parts = ["Please analyze the following information and answer the user's query."]
        prompt_parts.append("\n---\n")
        # --- Client Context ---
        client_context = context_dict.get('client')
        if client_context:
            prompt_parts.append("**Client Information**")
            prompt_parts.append(f"- Client ID: `{client_context.get('id', 'N/A')}`")
            if client_context.get('risk_profile'): prompt_parts.append(f"- Risk Profile: {client_context['risk_profile']}")
            if client_context.get('portfolio_value', 0) > 0: prompt_parts.append(f"- Portfolio Value: ${client_context['portfolio_value']:,.2f}")
            holdings = client_context.get('holdings', [])
            if holdings:
                prompt_parts.append(f"- Current Holdings ({len(holdings)} total):")
                sorted_holdings = sorted(holdings, key=lambda x: x.get('value', 0), reverse=True)
                holdings_to_show = sorted_holdings[:MAX_HOLDINGS_IN_PROMPT]
                holdings_str = [f"  - `{h.get('symbol', 'N/A')}`: {h.get('allocation', 0):.1f}% (${h.get('value', 0):,.2f})" for h in holdings_to_show]
                prompt_parts.extend(holdings_str)
                if len(holdings) > MAX_HOLDINGS_IN_PROMPT: prompt_parts.append(f"  - ... (and {len(holdings) - MAX_HOLDINGS_IN_PROMPT} more)")
            else: prompt_parts.append("- Current Holdings: None")
            prompt_parts.append("\n---\n")
        # --- Query ---
        prompt_parts.append("**User's Query**")
        prompt_parts.append(f"```\n{context_dict.get('original_query', 'N/A')}\n```")
        prompt_parts.append("\n---\n")
        # --- News ---
        if context_dict.get('news_summary'):
            prompt_parts.append("**Recent Market News Summary**")
            prompt_parts.append(context_dict['news_summary'])
            prompt_parts.append("\n---\n")
        # --- Market Data ---
        if context_dict.get('market_data_summary'):
            prompt_parts.append("**Requested Market Data Summary**")
            prompt_parts.extend([f"- {data_item}" for data_item in context_dict['market_data_summary']])
            prompt_parts.append("\n---\n")
        # --- Documents ---
        if context_dict.get('relevant_documents'):
            prompt_parts.append("**Context from Knowledge Base (Excerpt)**")
            prompt_parts.append(context_dict['relevant_documents'])
            prompt_parts.append("\n---\n")
        prompt_parts.append("Please provide a comprehensive analysis and response based *only* on the information above. Include the standard investment disclaimer.")
        final_prompt = "\n".join(prompt_parts)
        logger.debug(f"Final prompt built. Length: {len(final_prompt)} chars, Est. Tokens: ~{estimate_token_count(final_prompt)}")
        return final_prompt

    def _extract_symbols(self, text: str) -> List[str]:
        """Extracts potential financial symbols from text."""
        if not text: return []
        # Using a simplified regex approach for common patterns
        # Pattern for potential stock/ETF symbols (1-5 uppercase letters)
        stock_pattern = r'\b([A-Z]{1,5})\b'
        # Pattern for crypto pairs (e.g., BTC-USD, ETH/BTC) - adjust as needed
        crypto_pattern = r'\b([A-Z]{2,6}[-/][A-Z]{2,6})\b'
        # Pattern for prefixed symbols ($AAPL, ^GSPC)
        prefix_pattern = r'[$^.]([A-Z0-9.]+)\b' # Allow . for indices like .DJI

        symbols = set()
        proc_text = text.upper() # Process in uppercase

        for pattern in [stock_pattern, crypto_pattern, prefix_pattern]:
             try:
                 matches = re.findall(pattern, proc_text)
                 for match in matches:
                      # Simplify symbol extraction (handle tuples if groups are complex)
                      symbol = match[0] if isinstance(match, tuple) else match
                      # Basic cleaning/validation
                      symbol = symbol.strip('.') # Remove leading/trailing dots if captured
                      if symbol and len(symbol) <= 10: # Basic length check
                           symbols.add(symbol)
             except re.error as e: logger.error(f"Regex error: {e}")

        common_words = self._load_common_words()
        # Filter out common words and standalone numbers
        filtered_symbols = {s for s in symbols if s and s not in common_words and not s.isdigit()}

        final_list = sorted(list(filtered_symbols))
        logger.info(f"Extracted symbols: {final_list}")
        return final_list

    def _load_common_words(self) -> set:
         """Loads a set of common English words to filter potential symbols."""
         # Load from a file or use the extensive list provided previously
         # Example subset:
         words = {
             'A', 'I', 'IS', 'IT', 'BE', 'TO', 'DO', 'GO', 'ME', 'MY', 'NO', 'OF', 'ON', 'OR', 'SO', 'UP', 'US', 'WE',
             'ALL', 'AND', 'ANY', 'ARE', 'ASK', 'BUY', 'CAN', 'DAY', 'DID', 'DOW', 'FOR', 'GET', 'HAS', 'HAD', 'HOW',
             'LET', 'LOW', 'MAY', 'NEW', 'NOT', 'NOW', 'OFF', 'OLD', 'ONE', 'OUR', 'OUT', 'OWN', 'PUT', 'RUN', 'SAY',
             'SEE', 'SET', 'SHE', 'SIR', 'SIX', 'SUN', 'TAX', 'TEN', 'THE', 'TOO', 'TOP', 'TRY', 'TWO', 'USE', 'WAS',
             'WAY', 'WHO', 'WHY', 'YES', 'YET', 'YOU', 'YOUR', 'AREA', 'BANK', 'BASE', 'BEST', 'BOTH', 'BUSY', 'CALL',
             'CARD', 'CARE', 'CASH', 'CHAT', 'CODE', 'COME', 'COST', 'DATA', 'DATE', 'DEAL', 'DEBT', 'DONE', 'DOWN',
             'DRAW', 'DROP', 'EACH', 'EASY', 'EDGE', 'ELSE', 'EVEN', 'EVER', 'EXIT', 'FACE', 'FACT', 'FAIL', 'FALL',
             'FAST', 'FEAR', 'FEEL', 'FILE', 'FILL', 'FIND', 'FINE', 'FIRE', 'FIRM', 'FIVE', 'FLAT', 'FLOW', 'FOOD',
             'FOOT', 'FORM', 'FOUR', 'FREE', 'FROM', 'FUEL', 'FULL', 'FUND', 'GAIN', 'GAME', 'GAVE', 'GIFT', 'GIVE',
             'GOAL', 'GOLD', 'GONE', 'GOOD', 'GROW', 'HALF', 'HALL', 'HAND', 'HARD', 'HAVE', 'HEAD', 'HEAR', 'HELD',
             'HELP', 'HERE', 'HIGH', 'HOLD', 'HOME', 'HOPE', 'HOUR', 'HUGE', 'IDEA', 'INTO', 'ITEM', 'JOIN', 'JUMP',
             'JUST', 'KEEP', 'KIND', 'KNEW', 'KNOW', 'LACK', 'LAKE', 'LAND', 'LAST', 'LATE', 'LEAD', 'LEFT', 'LESS',
             'LIFE', 'LIFT', 'LIKE', 'LINE', 'LINK', 'LIST', 'LIVE', 'LOAD', 'LOAN', 'LOCK', 'LONG', 'LOOK', 'LOSE',
             'LOSS', 'LOST', 'LOUD', 'LOVE', 'LUCK', 'MADE', 'MAIL', 'MAIN', 'MAKE', 'MANY', 'MARK', 'MASS', 'MEAN',
             'MEET', 'MENU', 'MIND', 'MINE', 'MISS', 'MODE', 'MORE', 'MOST', 'MOVE', 'MUCH', 'MUST', 'NAME', 'NEAR',
             'NEED', 'NEWS', 'NEXT', 'NICE', 'NINE', 'NONE', 'NOTE', 'ONLY', 'OPEN', 'OVER', 'PAGE', 'PAID', 'PAIN',
             'PAIR', 'PART', 'PASS', 'PAST', 'PATH', 'PAY', 'PEAK', 'PLAN', 'PLAY', 'PLUS', 'POLL', 'POOL', 'POOR',
             'PORT', 'POST', 'PULL', 'PUSH', 'PUTS', 'QUIT', 'RACE', 'RAIN', 'RATE', 'READ', 'REAL', 'RENT', 'REST',
             'RICE', 'RICH', 'RIDE', 'RING', 'RISE', 'RISK', 'ROAD', 'ROCK', 'ROLE', 'ROLL', 'ROOM', 'ROOT', 'ROSE',
             'RULE', 'RUSH', 'SAFE', 'SAID', 'SAIL', 'SALE', 'SAME', 'SAND', 'SAVE', 'SEAT', 'SEED', 'SEEK', 'SEEM',
             'SEEN', 'SELF', 'SELL', 'SEND', 'SENT', 'SHIP', 'SHOP', 'SHOT', 'SHOW', 'SICK', 'SIDE', 'SIGN', 'SING',
             'SITE', 'SIZE', 'SKIN', 'SLOW', 'SNOW', 'SOFT', 'SOIL', 'SOLD', 'SOME', 'SONG', 'SOON', 'SORT', 'SPOT',
             'STAR', 'STAY', 'STEP', 'STOP', 'SUCH', 'SUIT', 'SURE', 'TAKE', 'TALK', 'TASK', 'TEAM', 'TELL', 'TEST',
             'THAN', 'THAT', 'THEN', 'THEY', 'THIS', 'THUS', 'TIME', 'TINY', 'TOUR', 'TOWN', 'TRIP', 'TRUE', 'TUNE',
             'TURN', 'TYPE', 'UNIT', 'UPON', 'USED', 'USER', 'VIEW', 'VOTE', 'WAIT', 'WAKE', 'WALK', 'WALL', 'WANT',
             'WARM', 'WASH', 'WAVE', 'WEAR', 'WEEK', 'WELL', 'WENT', 'WERE', 'WHAT', 'WHEN', 'WHOM', 'WIDE', 'WIFE',
             'WILD', 'WILL', 'WIND', 'WINE', 'WING', 'WIRE', 'WISE', 'WISH', 'WITH', 'WORD', 'WORK', 'YEAR', 'ZERO',
             'PLEASE', 'ANALYZE', 'SHOULD', 'BASED', 'CONTEXT', 'CURRENT', 'CLIENT', 'QUERY', 'REPORT', 'SUMMARY',
             'HOLDINGS', 'VALUE', 'PROFILE', 'MARKET', 'RECENTLY', 'COMPARE', 'PERFORMANCE', 'FINANCIAL',
             'METRIC', 'RATIO', 'RECOMMENDATION', 'ACTIONABLE', 'ANALYSIS', 'OVERVIEW', 'SYSTEM', 'RESPONSE',
             'OBJECT', 'CONTENT', 'ERROR', 'MESSAGE', 'WARNING', 'FAILED', 'CLIENT', 'CONTEXT', 'INVALID'
             # Add more common words or finance terms that aren't symbols
         }
         return words

# --- Standalone Test Block ---
if __name__ == "__main__":
    logger.info("--- Running Agent Standalone Test ---")
    try:
        agents = FinancialAgents()
        logger.info("FinancialAgents initialized for test.")
        # Test cases...
        query1 = "Analyze NVDA stock. Is it overvalued?"
        logger.info(f"--- Test Query 1: '{query1}' ---")
        response1 = agents.get_response(query1)
        print("\nResponse 1:\n", response1)
        print("-" * 30)

        query2 = "Compare AMD and INTC performance recently."
        logger.info(f"--- Test Query 2: '{query2}' ---")
        response2 = agents.get_response(query2)
        print("\nResponse 2:\n", response2)
        print("-" * 30)

        query3 = "Based on my aggressive profile, is adding more SOXX a good idea?"
        client_context_sim = {
            "id": "CLIENT_TEST_007", "name": "Client Test 007", "risk_profile": "Aggressive",
            "portfolio_value": 250000.00,
            "holdings": [ {"symbol": "QQQ", "value": 100000.00, "allocation": 40.0}, {"symbol": "SOXX", "value": 75000.00, "allocation": 30.0}, {"symbol": "TSLA", "value": 50000.00, "allocation": 20.0}, {"symbol": "BTC-USD", "value": 25000.00, "allocation": 10.0} ],
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S") }
        logger.info(f"--- Test Query 3: '{query3}' (with context) ---")
        response3 = agents.get_response(query3, client_id="CLIENT_TEST_007", client_context=client_context_sim)
        print("\nResponse 3:\n", response3)
        print("-" * 30)

    except RuntimeError as e: print(f"\nTEST FAILED (RuntimeError): {e}\n{traceback.format_exc()}")
    except Exception as e: print(f"\nTEST FAILED (Unexpected Error): {e}\n{traceback.format_exc()}")