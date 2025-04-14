# src/fintech_ai_bot/core/orchestrator.py
# User-provided version (returns single Markdown string)

from typing import Dict, Optional, List, Any
import time
import re
import traceback
from agno.exceptions import ModelProviderError
import requests

from fintech_ai_bot.config import settings
from fintech_ai_bot.utils import (
    get_logger,
    format_markdown_response,
    estimate_token_count,
    generate_error_html, # Still used for error responses
    generate_warning_html, # Still used for error responses
    log_execution_time,
    validate_portfolio_data,
    validate_query_text,
    validate_client_id,
    summarize_financial_data
)
from fintech_ai_bot.db.postgres_client import PostgresClient
from fintech_ai_bot.vector_store.faiss_client import FAISSClient
# Import specific agent classes using __init__.py convenience
from fintech_ai_bot.agents import BaseAgent, FinancialAgent, NewsAgent, CoordinatorAgent

logger = get_logger(__name__)

class AgentOrchestrator:
    """Orchestrates the interaction between various AI agents, data sources, and the user."""

    def __init__(self, db_client: PostgresClient, vector_store_client: Optional[FAISSClient]):
        """
        Initializes the orchestrator.

        Args:
            db_client: An initialized PostgresClient instance.
            vector_store_client: An initialized FAISSClient instance, or None if failed.
        """
        self.db_client = db_client
        self.vector_store_client = vector_store_client # Can be None
        self.agents: Dict[str, BaseAgent] = self._initialize_agents()
        logger.info("Agent Orchestrator initialized.")

    def _initialize_agents(self) -> Dict[str, BaseAgent]:
        """Initializes and returns a dictionary of all required agents."""
        try:
            financial_agent = FinancialAgent() # Uses reverted init/instructions
            news_agent = NewsAgent()
            agents_dict = {
                'financial': financial_agent,
                'news': news_agent,
            }
            coordinator_agent = CoordinatorAgent(list(agents_dict.values()))
            agents_dict['coordinator'] = coordinator_agent
            logger.info("All agents initialized successfully.")
            return agents_dict
        except Exception as e:
            logger.critical(f"Failed to initialize one or more agents: {e}", exc_info=True)
            raise RuntimeError("Core agent initialization failed.") from e

    # --- Updated Portfolio Processing Method ---
    def _process_db_portfolio(self, client_id: str, raw_portfolio: dict) -> Optional[dict]:
        """
        Processes raw portfolio data from DB into the standard context format,
        AGGREGATING holdings by symbol and PRESERVING client name.
        """
        if not raw_portfolio:  # Add check for empty raw_portfolio
            logger.warning(f"Received empty raw_portfolio for client {client_id} in processing.")
            return None
        try:
            # --- CORRECTED NAME INITIALIZATION ---
            # Get name from raw_portfolio, fallback to client_id if 'name' is missing/empty
            fetched_name = raw_portfolio.get('name')
            client_display_name = fetched_name if fetched_name else client_id
            # --- END CORRECTION ---

            portfolio = {
                "id": client_id,
                "name": client_display_name,  # Use the fetched name with fallback
                "risk_profile": raw_portfolio.get('risk_profile', 'Not specified') or 'Not specified',
                "portfolio_value": float(raw_portfolio.get('total_value', 0.0)),
                "holdings": [],  # This will store the FINAL aggregated holdings
                # Using get with a default for last_update if needed, although postgres_client should provide total_value
                # For consistency, let's assume last_updated is not a primary field here unless specifically needed
                # "last_update": raw_portfolio.get('last_updated', time.strftime("%Y-%m-%d %H:%M:%S"))
            }

            holdings_data_raw = raw_portfolio.get('holdings', [])
            total_portfolio_value = portfolio['portfolio_value']
            aggregated_holdings = {}  # Use a dict to aggregate: {symbol: total_value}

            if isinstance(holdings_data_raw, list):
                for holding in holdings_data_raw:
                    # Validate each holding item structure
                    if not isinstance(holding, dict) or 'symbol' not in holding or 'current_value' not in holding:
                        logger.warning(f"Skipping invalid raw holding item structure for client {client_id}: {holding}")
                        continue

                    symbol = str(holding['symbol']).strip().upper()
                    if not symbol:
                        logger.warning(f"Skipping raw holding item with empty symbol for client {client_id}.")
                        continue

                    value = holding.get('current_value', 0.0)
                    # Validate value type before conversion
                    if not isinstance(value, (int, float)):
                        try:
                            value = float(value)  # Attempt conversion if possible (e.g., from string decimal)
                        except (ValueError, TypeError):
                            logger.warning(
                                f"Invalid value type '{type(value)}' for symbol {symbol} in client {client_id}. Skipping value.")
                            value = 0.0
                    else:
                        value = float(value)  # Ensure it's float

                    # Add or update the aggregated value for the symbol
                    aggregated_holdings[symbol] = aggregated_holdings.get(symbol, 0.0) + value
            elif holdings_data_raw is not None:
                logger.warning(f"Holdings data for client {client_id} is not a list: {type(holdings_data_raw)}")

            # Calculate Allocations and Finalize Holdings List
            if total_portfolio_value > 0:  # Avoid division by zero
                for symbol, total_value_for_symbol in aggregated_holdings.items():
                    allocation = (total_value_for_symbol / total_portfolio_value * 100)
                    portfolio['holdings'].append({
                        "symbol": symbol,
                        "value": round(total_value_for_symbol, 2),  # Store aggregated value
                        "allocation": round(allocation, 1)  # Store calculated allocation
                    })
            else:  # Handle zero portfolio value case
                for symbol, total_value_for_symbol in aggregated_holdings.items():
                    portfolio['holdings'].append({
                        "symbol": symbol,
                        "value": round(total_value_for_symbol, 2),
                        "allocation": 0.0  # Allocation is 0 if total value is 0
                    })

            # Sort final aggregated holdings list by value
            portfolio['holdings'].sort(key=lambda x: x['value'], reverse=True)

            # Final validation before returning
            # Make sure validate_portfolio_data is imported/available
            if validate_portfolio_data(portfolio):
                logger.debug(f"Successfully processed and aggregated portfolio for client {client_id}")
                return portfolio
            else:
                logger.error(
                    f"Processed & Aggregated portfolio FAILED validation for client {client_id}. Data: {portfolio}")
                return None

        except Exception as e:
            logger.error(f"Error processing/aggregating raw portfolio for {client_id}: {e}", exc_info=True)
            return None

    def _prepare_client_context(self, client_id: Optional[str], client_context_from_ui: Optional[dict]) -> Optional[dict]:
        """Loads, validates, and prepares client context dictionary."""
        logger.debug(f"Preparing client context. Provided ID: {client_id}, UI Context: {client_context_from_ui is not None}")
        final_context = None
        current_client_id = validate_client_id(client_id) # Use imported util

        if client_context_from_ui and isinstance(client_context_from_ui, dict):
            if validate_portfolio_data(client_context_from_ui): # Use imported util
                context_id = client_context_from_ui.get('id')
                logger.info(f"Using validated client context from UI for client: {context_id or 'ID_MISSING'}")
                final_context = client_context_from_ui
                current_client_id = context_id or current_client_id
                if current_client_id and 'id' not in final_context:
                    final_context['id'] = current_client_id
            else:
                logger.warning("Invalid client context structure received from UI. Ignoring it.")
                current_client_id = client_context_from_ui.get('id', current_client_id)

        if final_context is None and current_client_id:
            logger.info(f"No valid UI context; fetching from DB for client ID: {current_client_id}")
            try:
                raw_portfolio_data = self.db_client.get_client_portfolio(current_client_id)
                if raw_portfolio_data:
                    # Call the updated processing function
                    processed_data = self._process_db_portfolio(current_client_id, raw_portfolio_data)
                    if processed_data:
                        logger.info(f"Successfully fetched and aggregated portfolio from DB for {current_client_id}")
                        final_context = processed_data
                    else:
                        logger.warning(f"Processing/aggregation of DB portfolio for {current_client_id} failed.")
                else:
                    logger.warning(f"No portfolio data found in DB for client {current_client_id}.")
            except Exception as e:
                logger.error(f"Failed to fetch/process portfolio from DB for {current_client_id}: {e}", exc_info=True)

        if final_context:
            logger.debug(f"Client context prepared for ID: {final_context.get('id', 'MISSING_ID')}")
        else:
            logger.debug("No client context available.")
        return final_context

    def _extract_symbols(self, text: str) -> List[str]:
        """Extracts potential financial symbols from text."""
        if not text: return []
        stock_pattern = r'\b([A-Z]{1,5})\b'; crypto_pattern = r'\b([A-Z]{2,6}[-/][A-Z]{2,6})\b'; prefix_pattern = r'[$^.]([A-Z0-9\.]{1,10})\b'
        symbols = set(); proc_text = text.upper()
        for pattern in [stock_pattern, crypto_pattern, prefix_pattern]:
            try:
                matches = re.findall(pattern, proc_text)
                for match in matches:
                    symbol = match[0] if isinstance(match, tuple) else match; symbol = symbol.strip('.')
                    if symbol and len(symbol) <= 15: symbols.add(symbol)
            except re.error as e: logger.error(f"Regex error: {e}")
        common_words = self._load_common_words()
        # Added 'TODAY' to common words based on logs
        filtered_symbols = {s for s in symbols if s and not s.isdigit() and ('-' not in s and '/' not in s and s not in common_words) or ('-' in s or '/' in s)}
        final_list = sorted(list(filtered_symbols)); logger.info(f"Extracted symbols: {final_list}"); return final_list

    def _load_common_words(self) -> set:
         """Loads a set of common English words to filter potential symbols."""
         # Keeping the extensive set defined previously, added TODAY
         words = {
             'A', 'ACCEPT', 'ACCOUNT', 'ACCOUNTS', 'ACCURACY', 'ACCURATE', 'ACQUIRE', 'ACTIONABLE', 'ACTIVE',
             'ACTIVITY','ABOUT', 'ADAPT', 'ADDITIONAL', 'ADDRESS', 'ADJUSTMENT', 'ADJUSTMENTS', 'ADJUSTED', 'ADJUSTING',
             'ADD', 'ADJUST', 'ADMIN', 'ADVANCE', 'AFTER', 'ALERT', 'ALL', 'ALLOCATE', 'ALLOW', 'ALGORITHM', 'ANALYSIS',
             'ANALYZE', 'AND', 'ANNUAL', 'ANY', 'APP', 'APPLICATION', 'APPLY', 'APPROVE', 'ARE', 'AREA', 'ASK', 'ASSET',
             'ASSETS', 'AUDIT', 'AUTO', 'AUTOMATE', 'AVERAGE', 'AVOID', 'BAD', 'BALANCE', 'BANK', 'BASE', 'BASED',
             'BEFORE',
             'BEGIN', 'BE', 'BEST', 'BILL', 'BILLING', 'BOTH', 'BONUS', 'BORROW', 'BRING', 'BUDGET', 'BUILD', 'BUSY',
             'BUY',
             'CALL', 'CALCULATE', 'CANCEL', 'CAN', 'CAPITAL', 'CARD', 'CARE', 'CASH', 'CATEGORY', 'CHANGE', 'CHARGE',
             'CHAT',
             'CHECK', 'CHOOSE', 'CLICK', 'CLIENT', 'CLIENTS', 'CLOSE', 'COLLECT', 'COME', 'COMMENT', 'COMMISSION',
             'COMMIT',
             'COMPANY', 'COMPARE', 'COMPLETE', 'COMPLIANT', 'CONTENT', 'CONTEXT', 'CONTRACT', 'CONTROL', 'CONVERT',
             'COPY',
             'COST', 'COSTS', 'COULD', 'CREATE', 'CREDIT', 'CRYPTO', 'CURRENT', 'CURRENTLY', 'CURRENCY', 'CUSTOMER',
             'DATA',
             'DASHBOARD', 'DATA', 'DATASET', 'DATE', 'DAY', 'DEAL', 'DEBT', 'DECIDE', 'DEFAULT', 'DEFERRED', 'DELETE',
             'DELIVER', 'DEMAND', 'DENY', 'DEPOSIT', 'DESIGN', 'DETAILS', 'DETECT', 'DID', 'DIGITAL', 'DISABLE',
             'DISPLAY',
             'DONE', 'DO', 'DOWNLOAD', 'DOW', 'DOWN', 'DRAW', 'DROP', 'DUE', 'DURATION', 'EACH', 'EARN', 'EARNING',
             'EASY',
             'ECONOMY', 'EDIT', 'EDGE', 'EFFECT', 'EFFORT', 'ELSE', 'EMPTY', 'ENABLE', 'ENTER', 'ENTRY', 'EQUITY',
             'ERROR',
             'ESTIMATE', 'EVEN', 'EVER', 'EXCHANGE', 'EXIT', 'EXPENSE', 'EXPENSES', 'EXPORT', 'EXPOSE', 'FACE', 'FACT',
             'FACTOR', 'FAIL', 'FAILED', 'FALL', 'FAST', 'FEAR', 'FEATURE', 'FEE', 'FEEL', 'FETCH', 'FIELD', 'FILE',
             'FILED',
             'FILL', 'FINAL', 'FINANCE', 'FINANCIAL', 'FIND', 'FINE', 'FINISHED', 'FIRE', 'FIRM', 'FIRST', 'FISCAL',
             'FIVE',
             'FLAT', 'FLOW', 'FOCUS', 'FOLDER', 'FOLLOW', 'FOOD', 'FOOT', 'FOR', 'FORECAST', 'FORM', 'FORMULA', 'FOUR',
             'FREE', 'FROM', 'FULL', 'FUND', 'FUNDS', 'FUEL', 'GAIN', 'GAME', 'GATEWAY', 'GAVE', 'GENERATE', 'GET',
             'GIFT',
             'GIVE', 'GOAL', 'GO', 'GOLD', 'GONE', 'GOOD', 'GROW', 'GROWTH', 'HAD', 'HALF', 'HALL', 'HAND', 'HANDLE',
             'HARD',
             'HAS', 'HAVE', 'HEAD', 'HEAR', 'HELD', 'HELP', 'HERE', 'HIGH', 'HISTORY', 'HOLD', 'HOLDINGS', 'HOME',
             'HOPE',
             'HOW', 'HUGE', 'IDEA', 'IF', 'IMPORT', 'IMPROVE', 'IN', 'INCOME', 'INCREASE', 'INDEX', 'INDICATOR', 'INFO',
             'INPUT', 'INSIGHT', 'INSTALL', 'INSTANT', 'INSURANCE', 'INTEREST', 'INTO', 'INVALID', 'INVENTORY',
             'INVOICE',
             'IS', 'ISSUE', 'ITEM', 'ITEMS', 'IT', 'JOURNAL', 'JUMP', 'JUST', 'KEEP', 'KEYWORD', 'KIND', 'KNEW', 'KNOW',
             'LABEL', 'LACK', 'LAKE', 'LAND', 'LANGUAGE', 'LAST', 'LATE', 'LATER', 'LAUNCH', 'LEAD', 'LEFT', 'LESS',
             'LET',
             'LEVEL', 'LIFE', 'LIFT', 'LIKE', 'LIMIT', 'LINE', 'LINK', 'LIST', 'LISTED', 'LIVE', 'LOAD', 'LOAN', 'LOCK',
             'LOG', 'LOGIN', 'LOGOUT', 'LONG', 'LOOK', 'LOSE', 'LOSS', 'LOST', 'LOUD', 'LOVE', 'LOW', 'LUCK', 'MADE',
             'MAIL', 'MAIN', 'MAKE', 'MANAGE', 'MANAGER', 'MANUAL', 'MANY', 'MARK', 'MARKET', 'MASS', 'MATCH', 'MEAN',
             'MEASURE', 'ME', 'MEET', 'MEMBER', 'MENU', 'MESSAGE', 'METHOD', 'METRIC', 'METRICS', 'MIGHT', 'MIND',
             'MINE',
             'MISS', 'MODE', 'MODEL', 'MODULE', 'MONITOR', 'MONEY', 'MONTHLY', 'MORE', 'MOST', 'MOVE', 'MUCH', 'MUST',
             'MY', 'NAME', 'NEAR', 'NEED', 'NET', 'NETWORK', 'NEW', 'NEWS', 'NEXT', 'NICE', 'NINE', 'NO', 'NONE',
             'NOTE',
             'NOTEBOOK', 'NOT', 'NOTIFY', 'NOW', 'OBJECT', 'OBJECTIVE', 'OF', 'OFF', 'OFFER', 'OLD', 'ON', 'ONGOING',
             'ONLY', 'OPEN', 'OPERATE', 'OPTIMIZE', 'OPTION', 'OR', 'ORDER', 'ORGANIZE', 'OUR', 'OUT', 'OUTPUT', 'OVER',
             'OVERALL', 'OVERVIEW', 'OWN', 'OWNED', 'OWNERSHIP', 'PANEL', 'PARTNER', 'PASS', 'PAUSE', 'PAY', 'PAYMENT',
             'PAYOUT', 'PENDING', 'PERCENT', 'PERFORMANCE', 'PERIOD', 'PLAN', 'PLATFORM', 'PLEASE', 'POLICY',
             'PORTFOLIO',
             'PREDICT', 'PREPARE', 'PRESS', 'PRICE', 'PRINCIPAL', 'PRINT', 'PRIVATE', 'PROCESS', 'PRODUCT', 'PROFILE',
             'PROFIT', 'PROGRAM', 'PROJECT', 'PROMPT', 'PROOF', 'PROVIDE', 'PROVIDER', 'PULL', 'PURCHASE', 'PUSH',
             'PUT',
             'PUTS', 'QUANTITY', 'QUERY', 'QUICK', 'QUIT', 'RACE', 'RAIN', 'RATE', 'READ', 'READY', 'REAL', 'REASON',
             'RECEIVE', 'RECENTLY', 'RECOMMENDATION', 'RECORD', 'RECURRING', 'REDUCE', 'REFERENCE', 'REFRESH', 'REGION',
             'REGISTER', 'RELATED', 'RELEVANT', 'REMAIN', 'REMOVE', 'REPAY', 'REPORT', 'REQUEST', 'REQUIRE', 'RESET',
             'RESOLVE', 'RESOURCE', 'RESPONSE', 'REST', 'RESULT', 'RETAIN', 'RETRIEVE', 'RETURN', 'REVENUE', 'REVIEW',
             'RICE', 'RICH', 'RIDE', 'RING', 'RISE', 'RISK', 'RISKY', 'ROAD', 'ROCK', 'ROLE', 'ROLL', 'ROOM', 'ROOT',
             'ROSE', 'ROUTINE', 'RULE', 'RULES', 'RUSH', 'RUN', 'RUNNING', 'SAFE', 'SAID', 'SAIL', 'SALE', 'SAME',
             'SAND', 'SAVE', 'SAY', 'SCENARIO', 'SCORE', 'SEARCH', 'SEAT', 'SECURE', 'SECURITY', 'SEE', 'SEED', 'SEEK',
             'SEEM', 'SEEN', 'SEGMENT', 'SELF', 'SELL', 'SELLER', 'SEND', 'SENT', 'SERVICE', 'SET', 'SETUP', 'SHALL',
             'SHAREHOLDER', 'SHARES', 'SHE', 'SHIP', 'SHOP', 'SHOT', 'SHOULD', 'SHOW', 'SICK', 'SIDE', 'SIGN', 'SIGNAL',
             'SIMPLE', 'SIMULATION', 'SING', 'SINGLE', 'SIR', 'SITE', 'SIZE', 'SIX', 'SKIN', 'SLOW', 'SMART', 'SNOW',
             'SO', 'SOFT', 'SOIL', 'SOLD', 'SOME', 'SONG', 'SOON', 'SORT', 'SOURCE', 'SPENDING', 'SPOT', 'STAR',
             'START',
             'STARTED', 'STATEMENT', 'STATUS', 'STAY', 'STEP', 'STOCK', 'STOCKS', 'STOP', 'STORE', 'STRATEGY', 'STRONG',
             'STRUCTURE', 'SUBMIT', 'SUCH', 'SUMMARY', 'SUPPORT', 'SURE', 'SURVEY', 'SUIT', 'SYMBOL', 'SYNC', 'SYSTEM',
             'SYSTEMS', 'TAG', 'TAKE', 'TALK', 'TARGET', 'TASK', 'TASKS', 'TAX', 'TAXES', 'TEAM', 'TECHNICAL',
             'TECHNOLOGY',
             'TELL', 'TEMPLATE', 'TEN', 'TERM', 'TEST', 'THAN', 'THAT', 'THE', 'THEY', 'THEN', 'THIS', 'THUS', 'TIER',
             'TIME', 'TIMEFRAME', 'TINY', 'TO', 'TODAY', 'TOKEN', 'TOOLS', 'TOO', 'TOP', 'TOTAL', 'TOUR', 'TOWN',
             'TRACK',
             'TRADE', 'TRANSACTION', 'TRANSFER', 'TRANSLATE', 'TREND', 'TRIP', 'TRUE', 'TRUST', 'TRY', 'TUNE', 'TURN',
             'TWO', 'TYPE', 'UNIT', 'UP', 'UPDATE', 'UPLOAD', 'UPON', 'USD', 'US', 'USED', 'USER', 'USERNAME',
             'UTILITY',
             'VALID', 'VALIDATE', 'VALUE', 'VARIANCE', 'VERIFY', 'VERSION', 'VIEW', 'VIEWER', 'VIRTUAL', 'VOTE', 'WAIT',
             'WAKE', 'WALK', 'WALL', 'WALLET', 'WANT', 'WARM', 'WARNING', 'WASH', 'WATCH', 'WAVE', 'WE', 'WEALTH',
             'WEAR',
             'WEEK', 'WELL', 'WENT', 'WERE', 'WHAT', 'WHEN', 'WHOM', 'WIDE', 'WIFE', 'WILD', 'WILL', 'WIND', 'WINE',
             'WING', 'WIRE', 'WISE', 'WISH', 'WITH', 'WOULD', 'WORD', 'WORK', 'WORKFLOW', 'YEAR', 'YES', 'YESTERDAY',
             'YET', 'YIELD', 'YOU', 'YOUR', 'ZERO'
         }
         return words

    @log_execution_time
    def _get_market_data(self, symbols: List[str]) -> List[str]:
        """Gets market data using the financial agent (original approach)."""
        market_data_results = []
        financial_agent = self.agents.get('financial')
        if not financial_agent:
            logger.error("Financial agent not initialized.")
            return [f"⚠️ Error: Financial analysis module unavailable."] * len(symbols)

        for symbol in symbols:
            time.sleep(settings.financial_api_delay)
            try:
                logger.info(f"Requesting financial data table for symbol: {symbol}")
                # Ask for the table directly, relying on agent's reverted instructions
                query = f"Provide key financial data table for {symbol}"
                content_str = financial_agent.run(query)

                if content_str:
                    if f"Financial data not available for {symbol}" in content_str:
                        logger.warning(f"Financial agent reported no data for {symbol}.")
                        market_data_results.append(f"⚠️ Data not available for symbol: {symbol}")
                    elif f"Metrics not applicable for {symbol}" in content_str:
                         logger.warning(f"Financial agent reported metrics N/A for {symbol}.")
                         market_data_results.append(f"⚠️ Metrics not applicable for symbol: {symbol}")
                    else:
                        # Assume valid table or info returned
                        prefix = f"**{symbol}**:\n"
                        if not content_str.strip().startswith(("|", "**", f"{symbol}:")):
                            market_data_results.append(prefix + content_str)
                        else: market_data_results.append(content_str)
                        logger.debug(f"Successfully retrieved response for {symbol}.")
                else:
                    logger.warning(f"Financial agent returned None or empty response for {symbol}.")
                    market_data_results.append(f"⚠️ No valid data received for symbol: {symbol}")

            except ModelProviderError as e:
                error_details = str(e).lower()
                # Still useful to log tool use failures specifically if they happen
                if 'tool_use_failed' in error_details:
                    logger.error(f"Tool use failed unexpectedly for {symbol}: {e}", exc_info=False)
                    market_data_results.append(f"⚠️ Tool execution failed for {symbol}")
                else:
                    logger.error(f"ModelProviderError from financial agent for {symbol}: {e}", exc_info=True)
                    market_data_results.append(f"⚠️ Error retrieving data for {symbol} (AI Model Error)")

            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else 'N/A'
                user_msg = f"⚠️ Error retrieving data for {symbol} (Network Error: {status})"
                if status == 404: user_msg = f"⚠️ Data not available for symbol: {symbol} (Not Found)"
                elif status == 429: user_msg = f"⚠️ Error retrieving data for {symbol} (Rate Limit)"
                logger.warning(f"HTTP error for {symbol}: Status {status}", exc_info=True)
                market_data_results.append(user_msg)

            except Exception as e:
                logger.error(f"Unexpected error getting data for {symbol}: {e}", exc_info=True)
                market_data_results.append(f"⚠️ Unexpected error retrieving data for {symbol}")

        return market_data_results

    @log_execution_time
    def _get_news_summary(self, query: str, symbols_to_fetch: List[str]) -> Optional[str]:
        """Gets news summary using the news agent."""
        news_agent = self.agents.get('news')
        if not news_agent:
            logger.error("News agent not initialized.")
            return "⚠️ News retrieval module unavailable."
        try:
            news_query = f"Top financial market news relevant to: {query}" + (f" (especially {', '.join(symbols_to_fetch)})" if symbols_to_fetch else "")
            logger.info("Fetching news summary...")
            news_content = news_agent.run(news_query)
            if news_content:
                if "No significant market-moving news found" in news_content:
                    logger.info("News agent returned no significant news.")
                    return None
                else:
                    if len(news_content) > settings.max_news_summary_len:
                        logger.warning(f"Truncating news summary from {len(news_content)} to {settings.max_news_summary_len} chars.")
                        return news_content[:settings.max_news_summary_len].strip() + "..."
                    else:
                        logger.debug(f"News summary received: {news_content[:100]}...")
                        return news_content
            else:
                logger.warning("News agent returned None or empty response.")
                return "⚠️ News retrieval failed or returned no content."
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code in [429, 202]: # Check for rate limit codes
                 logger.warning(f"DuckDuckGo tool hit rate limit: {e}")
                 return "⚠️ News retrieval failed due to rate limit."
            else:
                 logger.error(f"News agent HTTP error: {e}", exc_info=True)
                 return "⚠️ Error retrieving news (Network/API issue)."
        except ModelProviderError as e:
             error_details = str(e).lower();
             if 'tool_use_failed' in error_details: logger.error(f"Tool use failed for News Agent: {e}", exc_info=False); return f"⚠️ News retrieval tool execution failed."
             else: logger.error(f"ModelProviderError from news agent: {e}", exc_info=True); return f"⚠️ Error retrieving news (AI Model Error)."
        except Exception as e:
            logger.error(f"News agent query failed unexpectedly: {e}", exc_info=True)
            return "⚠️ Unexpected error retrieving news."


    @log_execution_time
    def _get_vector_search_results(self, query: str) -> Optional[str]:
        """Performs vector search and formats results."""
        if self.vector_store_client is None: logger.warning("Vector store client not available."); return None
        try:
            search_results = self.vector_store_client.search(query)
            if not search_results: logger.info("No relevant documents found."); return None
            formatted_docs = ["**Context from Knowledge Base (Excerpt)**"]; doc_text_aggregate = ""
            current_token_count = 0; max_tokens = settings.max_doc_tokens_in_prompt
            for i, res in enumerate(search_results, 1):
                source = res.get('source', 'Unk'); doc_type = res.get('type', 'doc').title()
                text_excerpt = res.get('text', ''); excerpt_tokens = estimate_token_count(text_excerpt)
                doc_entry_overhead = estimate_token_count(f"\n📄 **Doc {i} ({doc_type}) - Src: {source}**\n\n---")
                if current_token_count + excerpt_tokens + doc_entry_overhead < max_tokens:
                    doc_entry = f"\n📄 **Doc {i} ({doc_type}) - Src: {source}**\n{text_excerpt}\n---"
                    doc_text_aggregate += doc_entry; current_token_count += excerpt_tokens + doc_entry_overhead
                else: logger.warning(f"Doc inclusion token limit ({max_tokens}) reached."); break
            if not doc_text_aggregate: return None
            formatted_docs.append(doc_text_aggregate.strip()); return "\n".join(formatted_docs)
        except Exception as e: logger.error(f"Vector search/format failed: {e}", exc_info=True); return "⚠️ Error retrieving/formatting docs."

    @log_execution_time
    def _get_enhanced_context(self, query: str, client_context: Optional[dict]) -> dict:
        """Builds context dict: client data, agent results, documents."""
        context_dict = {'client': client_context, 'market_data_summary': ["No specific stock symbols identified."], 'news_summary': None, 'relevant_documents': None, 'original_query': query }
        portfolio_symbols_text = " ".join([str(h['symbol']) for h in client_context['holdings']]) if client_context and client_context.get('holdings') else ""
        combined_text = query + " " + portfolio_symbols_text
        symbols = self._extract_symbols(combined_text)
        symbols_to_fetch = symbols[:settings.max_symbols_to_fetch]
        if len(symbols) > settings.max_symbols_to_fetch: logger.warning(f"Too many symbols ({len(symbols)}), limiting fetch to first {settings.max_symbols_to_fetch}: {symbols_to_fetch}")

        # Sequential execution (can be parallelized later if needed)
        if symbols_to_fetch:
            market_data_results = self._get_market_data(symbols_to_fetch)
            context_dict['market_data_summary'] = [summarize_financial_data(r, settings.max_financial_summary_len) if isinstance(r, str) and not r.startswith("⚠️") else r for r in market_data_results]
        context_dict['news_summary'] = self._get_news_summary(query, symbols_to_fetch)
        context_dict['relevant_documents'] = self._get_vector_search_results(query)
        return context_dict

    @log_execution_time
    def _build_prompt(self, context_dict: dict) -> str:
        """Constructs the final prompt string for the coordinator agent."""
        # Using the instructions that ask coordinator to include full tables
        prompt_parts = ["Please analyze the following information and answer the user's query."]
        prompt_parts.append("\n---\n")
        client_context = context_dict.get('client')
        if client_context:
            prompt_parts.append("**Client Information**"); prompt_parts.append(f"- Client ID: `{client_context.get('id', 'N/A')}`")
            # Include name in prompt
            if client_context.get('name') and client_context['name'] != client_context.get('id'):
                 prompt_parts.append(f"- Client Name: {client_context['name']}")
            if client_context.get('risk_profile'): prompt_parts.append(f"- Risk Profile: {client_context['risk_profile']}")
            if client_context.get('portfolio_value', 0) > 0: prompt_parts.append(f"- Portfolio Value: ${client_context['portfolio_value']:,.2f}")
            holdings = client_context.get('holdings', []);
            if holdings:
                prompt_parts.append(f"- Current Holdings ({len(holdings)} total):")
                # Use aggregated holdings now
                holdings_to_show = holdings[:settings.max_holdings_in_prompt] # Already sorted by value
                holdings_str = [f"  - `{h.get('symbol', 'N/A')}`: {h.get('allocation', 0):.1f}% (${h.get('value', 0):,.2f})" for h in holdings_to_show]
                prompt_parts.extend(holdings_str)
                if len(holdings) > settings.max_holdings_in_prompt: prompt_parts.append(f"  - ... (and {len(holdings) - settings.max_holdings_in_prompt} more)")
            else: prompt_parts.append("- Current Holdings: None")
            prompt_parts.append("\n---\n")
        prompt_parts.append("**User's Query**"); prompt_parts.append(f"```\n{context_dict.get('original_query', 'N/A')}\n```"); prompt_parts.append("\n---\n")
        news_summary = context_dict.get('news_summary')
        prompt_parts.append("**Recent Market News Summary**"); prompt_parts.append(news_summary if news_summary else "No significant market-moving news found or retrieval failed."); prompt_parts.append("\n---\n")
        market_data_summary = context_dict.get('market_data_summary');
        if market_data_summary:
            prompt_parts.append("**Financial Data Context**") # Changed header slightly
            if isinstance(market_data_summary, list):
                 # Pass the raw list items (tables or errors) to the coordinator
                 for item in market_data_summary:
                     prompt_parts.append(f"{str(item).strip()}") # Pass directly
                     prompt_parts.append("---") # Add separator between symbols
            else:
                 prompt_parts.append(str(market_data_summary)) # Fallback
            # Removed the extra "\n---\n" after the last item
        relevant_documents = context_dict.get('relevant_documents');
        if relevant_documents:
            prompt_parts.append(relevant_documents) # Header included in string
            prompt_parts.append("\n---\n")
        prompt_parts.append("Please provide a comprehensive analysis and response based *only* on the information above. Follow the structure defined in your instructions (especially for presenting financial data, like using Markdown tables) and include the standard investment disclaimer.") # Added hint for Markdown tables
        final_prompt = "\n".join(prompt_parts); logger.debug(f"Final prompt built. Length: {len(final_prompt)} chars, Est. Tokens: ~{estimate_token_count(final_prompt)}"); return final_prompt

    @log_execution_time
    def get_response(self, query: str, client_id: str = None, client_context: dict = None) -> str:
        """Main method to get a response. Orchestrates context gathering, agent calls, and error handling. Returns a Markdown string."""
        final_prompt_string = ""
        try:
            # Validate query input
            if not validate_query_text(query):
                logger.warning(f"Invalid query received: '{query}'")
                # Return simple error string for UI to handle with type='error'
                return "Error: Invalid Query Format - Please provide a specific question (3-1500 characters)."

            logger.info(f"Processing query for client '{client_id or 'generic'}': '{query[:100]}...'")

            # Prepare context (fetches/processes DB data if needed)
            processed_client_context = self._prepare_client_context(client_id, client_context)

            # Enhance context with agent results (market data, news, docs)
            context_dict = self._get_enhanced_context(query, processed_client_context)

            # Build the final prompt for the coordinator agent
            final_prompt_string = self._build_prompt(context_dict)

            # Estimate token count and log warning if high
            estimated_tokens = estimate_token_count(final_prompt_string)
            logger.info(f"Est. coordinator tokens: ~{estimated_tokens}")
            # Adjust threshold based on the actual model context window
            INPUT_TOKEN_WARNING_THRESHOLD = settings.coordinator_agent_model.max_tokens * 0.8 # e.g., 80% of model limit
            if estimated_tokens > INPUT_TOKEN_WARNING_THRESHOLD:
                logger.warning(f"Prompt token estimate ({estimated_tokens}) exceeds threshold ({INPUT_TOKEN_WARNING_THRESHOLD}).")

            # Get coordinator agent and run it
            coordinator_agent = self.agents.get('coordinator')
            if not coordinator_agent:
                logger.critical("Coordinator agent missing!")
                raise RuntimeError("Coordinator agent not initialized.")

            logger.debug(f"Sending prompt to coordinator ({settings.coordinator_agent_model.id}). Length: {len(final_prompt_string)} chars")
            response_content = coordinator_agent.run(final_prompt_string) # This returns a single string

            # Format and return the response
            if response_content:
                # Format basic markdown (line breaks etc.) - assumes coordinator provides structure
                return format_markdown_response(response_content)
            else:
                logger.error("Coordinator agent returned None or empty response.")
                # Return simple error string
                return "Error: Processing Error - AI advisor failed the final response generation."

        # --- Error Handling ---
        except ModelProviderError as e:
            error_details = str(e); logger.error(f"Model provider error: {error_details}", exc_info=True)
            # Specific error checks
            if "context_length_exceeded" in error_details.lower() or (hasattr(e, 'code') and e.code == 400 and "maximum context length" in error_details.lower()):
                ft = estimate_token_count(final_prompt_string); logger.error(f"Context Length Exceeded. Est. tokens: ~{ft}")
                return f"Error: Request Too Complex - Analysis context (~{ft} tokens) exceeded model limits. Please simplify your request or ask about fewer items."
            elif "model_decommissioned" in error_details.lower():
                 logger.error(f"Model Decommissioned: {error_details}");
                 return f"Error: AI Model Error - The AI model is currently unavailable or decommissioned. Please check configuration. Details: {error_details}"
            elif "tool_use_failed" in error_details.lower():
                 logger.error(f"Tool Use Failed Error: {error_details}");
                 return f"Error: AI Tool Error - An underlying AI tool failed during execution. Details: {error_details}"
            else:
                 logger.error(f"Failed Prompt Snippet:\n{final_prompt_string[:500]}...");
                 return f"Error: AI Model Error - An issue occurred with the AI model. Please try again. Details: {error_details}"
        except (RuntimeError, ValueError) as e:
             logger.error(f"Internal processing error: {str(e)}", exc_info=True);
             return f"Error: Processing Error - An internal error occurred: {e}"
        except Exception as e:
             logger.critical(f"Unexpected critical error: {str(e)}", exc_info=True);
             logger.critical(traceback.format_exc());
             return "Error: Unexpected Error - An unexpected internal error occurred. Please contact support."

        def _try_parse_markdown_table(self, text_block: str) -> Optional[List[Dict[str, str]]]:
            """Attempts to parse the first Markdown table found in a text block."""
            lines = text_block.strip().split('\n')
            table_lines = []
            in_table = False
            header = []
            col_align = []

            for i, line in enumerate(lines):
                stripped = line.strip()
                # Look for table row markers or separator lines
                is_separator = '---' in stripped and stripped.replace('-', '').replace('|', '').replace(' ',
                                                                                                        '').replace(':',
                                                                                                                    '') == ''
                is_potential_data_row = stripped.startswith('|') and stripped.endswith('|') and not is_separator

                if is_separator:
                    # Check if the previous line looks like a header
                    if i > 0 and lines[i - 1].strip().startswith('|') and not ('---' in lines[i - 1]):
                        header_line = lines[i - 1].strip()
                        header = [h.strip() for h in header_line.strip('|').split('|')]
                        col_align = [a.strip() for a in stripped.strip('|').split('|')]
                        if len(header) > 0 and len(header) == len(col_align):
                            in_table = True
                            # Clear any previous non-table lines mistaken for data
                            table_lines = []
                        else:
                            header = []  # Reset if mismatch or empty header
                            in_table = False
                    continue  # Move to next line after processing separator

                elif is_potential_data_row and in_table:  # Only collect data rows if we are confirmed to be in a table
                    table_lines.append(stripped)

                elif in_table:  # A non-table line encountered after table started, assume table ended
                    break

            if not in_table or not header or not table_lines:
                logger.debug("No valid markdown table found in block.")
                return None  # No valid table found

            parsed_data = []
            num_cols = len(header)
            for line_num, line in enumerate(table_lines):
                cols = [c.strip() for c in line.strip('|').split('|')]
                if len(cols) == num_cols:
                    # Ensure header keys are not empty strings if parsing produced them
                    clean_header = [h if h else f"Column_{idx + 1}" for idx, h in enumerate(header)]
                    row_dict = dict(zip(clean_header, cols))
                    parsed_data.append(row_dict)
                else:
                    logger.warning(
                        f"Skipping table row {line_num + 1} due to column count mismatch. Expected {num_cols}, got {len(cols)}. Row: '{line}'")

            logger.debug(f"Parsed table with {len(parsed_data)} rows and columns: {header}")
            return parsed_data if parsed_data else None

        def _parse_coordinator_response(self, markdown_text: str) -> List[Dict[str, Any]]:
            """
            Parses the coordinator's markdown response into a structured list of dictionaries.
            Handles headers (like **Header:**) and attempts to parse markdown tables.
            """
            if not markdown_text or not isinstance(markdown_text, str):
                logger.warning("Received empty or invalid text for parsing.")
                # Return a structured representation even for empty/invalid input
                return [{"type": "markdown", "text": markdown_text or "[No content available]"}]

            # Define potential headers (case-insensitive matching, strip ':')
            # Expanded list based on common patterns
            known_headers = [
                "Executive Summary", "Summary",
                "Analysis Context", "Analysis Context: Market News", "Market News", "News Summary",
                "Financial Data", "Data", "Metrics",
                "Knowledge Base", "Document Context",
                "Client Portfolio Context", "Portfolio Context", "Client Context",
                "Recommendation & Risks", "Recommendations", "Risks", "Analysis", "Response",
                "Disclaimer", "Important Information"
            ]
            # Normalize headers: store lowercase version as key, original format with colon as value
            normalized_headers = {h.lower().strip(':'): h + ":" for h in known_headers}

            structured_list = []
            lines = markdown_text.strip().split('\n')
            current_content_lines = []
            current_header_text = None  # Holds the text of the *last* detected header
            found_first_header = False

            for line in lines:
                stripped_line = line.strip()
                matched_header = None

                # Check if line matches a known header format (e.g., **Header:** or #### Header)
                # More robust header matching
                header_match_bold = re.match(r'\*\*(.+?):\*\*', stripped_line)
                header_match_h4 = re.match(r'####\s*(.+)', stripped_line)

                potential_header_text = None
                if header_match_bold:
                    potential_header_text = header_match_bold.group(1).strip()
                elif header_match_h4:
                    potential_header_text = header_match_h4.group(1).strip().rstrip(
                        ':')  # Remove trailing colon if present

                # Normalize and check against known headers
                if potential_header_text:
                    normalized_potential = potential_header_text.lower()
                    if normalized_potential in normalized_headers:
                        matched_header = normalized_headers[normalized_potential]  # Get the canonical header text

                # If a new known header is found
                if matched_header:
                    found_first_header = True
                    # Process the content collected for the *previous* header
                    if current_header_text and current_content_lines:
                        content_block = "\n".join(current_content_lines).strip()
                        if content_block:  # Only add if there's actual content
                            # Special handling for Financial Data sections
                            if any(term in current_header_text.lower() for term in ["financial data", "metrics"]):
                                # Try to parse a table within this block
                                parsed_table = self._try_parse_markdown_table(content_block)
                                if parsed_table:
                                    # Find text *before* the table (if any)
                                    table_start_index = content_block.find('\n|')  # Find first newline followed by pipe
                                    if table_start_index == -1: table_start_index = content_block.find(
                                        '|')  # Or just first pipe

                                    markdown_before = content_block[
                                                      :table_start_index].strip() if table_start_index != -1 else None
                                    if markdown_before:
                                        structured_list.append({"type": "markdown", "text": markdown_before})
                                    structured_list.append({"type": "table", "data": parsed_table})
                                    # TODO: Handle text *after* the table if needed? Currently ignored.
                                else:
                                    # No table found, add the whole block as markdown
                                    structured_list.append({"type": "markdown", "text": content_block})
                            else:
                                # For other sections, just add the markdown block
                                structured_list.append({"type": "markdown", "text": content_block})

                    # Start the new section with the matched header
                    structured_list.append({"type": "header", "level": 4, "text": matched_header})
                    current_header_text = matched_header
                    current_content_lines = []  # Reset content for the new section
                elif found_first_header:  # Only collect content *after* the first header is found
                    # Add the line to the current section's content, preserving original spacing/formatting
                    current_content_lines.append(line)
                # Optional: Collect content before the first header if needed
                # elif not found_first_header and stripped_line:
                #     structured_list.append({"type": "markdown", "text": line}) # Add intro lines

            # Add the content of the very last section
            if current_header_text and current_content_lines:
                content_block = "\n".join(current_content_lines).strip()
                if content_block:
                    if any(term in current_header_text.lower() for term in ["financial data", "metrics"]):
                        parsed_table = self._try_parse_markdown_table(content_block)
                        if parsed_table:
                            table_start_index = content_block.find('\n|')
                            if table_start_index == -1: table_start_index = content_block.find('|')
                            markdown_before = content_block[
                                              :table_start_index].strip() if table_start_index != -1 else None
                            if markdown_before:
                                structured_list.append({"type": "markdown", "text": markdown_before})
                            structured_list.append({"type": "table", "data": parsed_table})
                        else:
                            structured_list.append({"type": "markdown", "text": content_block})
                    else:
                        structured_list.append({"type": "markdown", "text": content_block})

            # If parsing failed to identify any known headers, return the original text as one block
            if not found_first_header:
                logger.warning(
                    "Could not parse coordinator response into known sections. Returning as single markdown block.")
                return [{"type": "markdown", "text": markdown_text}]

            # Filter out empty markdown elements that might have been added
            return [item for item in structured_list if
                    item.get("type") != "markdown" or (item.get("text") and item.get("text").strip())]

        # Now, modify the get_response method like this:
        @log_execution_time
        def get_response(self, query: str, client_id: str = None, client_context: dict = None) -> List[
            Dict[str, Any]]:  # <<< Return type changed
            """
            Main method to get a response. Orchestrates context gathering, agent calls, and error handling.
            Returns a structured list of dictionaries representing the response content.
            """
            final_prompt_string = ""  # Keep for potential error context
            try:
                # Validate query input
                if not validate_query_text(query):
                    logger.warning(f"Invalid query received: '{query}'")
                    # Return structured error
                    return [{"type": "error",
                             "content": "Invalid Query Format - Please provide a specific question (3-1500 characters)."}]

                logger.info(f"Processing query for client '{client_id or 'generic'}': '{query[:100]}...'")

                # Prepare context (fetches/processes DB data if needed)
                processed_client_context = self._prepare_client_context(client_id, client_context)

                # Enhance context with agent results (market data, news, docs)
                context_dict = self._get_enhanced_context(query, processed_client_context)

                # Build the final prompt for the coordinator agent
                final_prompt_string = self._build_prompt(context_dict)

                # Estimate token count and log warning if high
                estimated_tokens = estimate_token_count(final_prompt_string)
                logger.info(f"Est. coordinator tokens: ~{estimated_tokens}")
                INPUT_TOKEN_WARNING_THRESHOLD = settings.coordinator_agent_model.max_tokens * 0.8
                if estimated_tokens > INPUT_TOKEN_WARNING_THRESHOLD:
                    logger.warning(
                        f"Prompt token estimate ({estimated_tokens}) exceeds threshold ({INPUT_TOKEN_WARNING_THRESHOLD}).")

                # Get coordinator agent and run it
                coordinator_agent = self.agents.get('coordinator')
                if not coordinator_agent:
                    logger.critical("Coordinator agent missing!")
                    # Return structured error
                    return [{"type": "error", "content": "System Configuration Error - Coordinator agent unavailable."}]

                logger.debug(
                    f"Sending prompt to coordinator ({settings.coordinator_agent_model.id}). Length: {len(final_prompt_string)} chars")
                response_content = coordinator_agent.run(final_prompt_string)  # Gets the markdown string

                # Format and return the response
                if response_content:
                    # *** NEW: Parse the markdown into structured list ***
                    try:
                        structured_response = self._parse_coordinator_response(response_content)
                        if not structured_response:  # Handle case where parsing returns empty
                            logger.error("Parsing coordinator response resulted in an empty structure. Falling back.")
                            return [
                                {"type": "markdown", "text": format_markdown_response(response_content)}]  # Fallback
                        logger.info("Successfully parsed coordinator response into structured format.")
                        return structured_response  # <<< Return the list[dict]
                    except Exception as parse_error:
                        logger.error(f"Failed to parse coordinator response into structured format: {parse_error}",
                                     exc_info=True)
                        # Fallback to returning the raw (but formatted) markdown within structure
                        return [{"type": "markdown", "text": format_markdown_response(response_content)}]
                else:
                    logger.error("Coordinator agent returned None or empty response.")
                    # Return structured error
                    return [{"type": "error",
                             "content": "Processing Error - AI advisor failed the final response generation."}]

            # --- Error Handling (Returning Structured Errors) ---
            except ModelProviderError as e:
                error_details = str(e);
                logger.error(f"Model provider error: {error_details}", exc_info=True)
                error_msg = f"AI Model Error - An issue occurred. Details: {error_details}"  # Default
                if "context_length_exceeded" in error_details.lower() or (
                        hasattr(e, 'code') and e.code == 400 and "maximum context length" in error_details.lower()):
                    ft = estimate_token_count(final_prompt_string);
                    logger.error(f"Context Length Exceeded. Est. tokens: ~{ft}")
                    error_msg = f"Request Too Complex - Analysis context (~{ft} tokens) exceeded model limits. Please simplify your request."
                elif "model_decommissioned" in error_details.lower():
                    error_msg = f"AI Model Error - Model unavailable/decommissioned. Check config. Details: {error_details}"
                elif "tool_use_failed" in error_details.lower():
                    error_msg = f"AI Tool Error - An underlying tool failed. Details: {error_details}"
                else:
                    logger.error(
                        f"Failed Prompt Snippet:\n{final_prompt_string[:500]}...");  # Log snippet for general errors
                return [{"type": "error", "content": error_msg}]
            except (RuntimeError, ValueError) as e:
                logger.error(f"Internal processing error: {str(e)}", exc_info=True);
                return [{"type": "error", "content": f"Processing Error - An internal error occurred: {e}"}]
            except Exception as e:
                logger.critical(f"Unexpected critical error: {str(e)}", exc_info=True);
                logger.critical(traceback.format_exc());
                return [{"type": "error",
                         "content": "Unexpected Error - An unexpected internal error occurred. Please contact support."}]