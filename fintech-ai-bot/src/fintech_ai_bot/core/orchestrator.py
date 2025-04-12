# src/fintech_ai_bot/core/orchestrator.py
# No major changes needed based on agent reversions, ensure imports are correct

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
    generate_error_html,
    generate_warning_html,
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
        self.db_client = db_client
        self.vector_store_client = vector_store_client
        self.agents: Dict[str, BaseAgent] = self._initialize_agents()
        logger.info("Agent Orchestrator initialized.")

    def _initialize_agents(self) -> Dict[str, BaseAgent]:
        """Initializes and returns a dictionary of all required agents."""
        try:
            financial_agent = FinancialAgent() # Now uses original init/instructions
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

    # _process_db_portfolio, _prepare_client_context, _extract_symbols,
    # _load_common_words methods remain the same as the last correct version...

    def _process_db_portfolio(self, client_id: str, raw_portfolio: dict) -> Optional[dict]:
        """Processes raw portfolio data from DB into the standard context format."""
        try:
            portfolio = {
                "id": client_id,
                "name": f"Client {client_id[-4:]}" if len(client_id) >= 4 else f"Client {client_id}",
                "risk_profile": raw_portfolio.get('risk_profile', 'Not specified') or 'Not specified',
                "portfolio_value": float(raw_portfolio.get('total_value', 0.0)),
                "holdings": [],
                "last_update": raw_portfolio.get('last_updated', time.strftime("%Y-%m-%d %H:%M:%S"))
            }
            holdings_data = raw_portfolio.get('holdings', [])
            total_val = portfolio['portfolio_value']

            if isinstance(holdings_data, list):
                for holding in holdings_data:
                    if not isinstance(holding, dict) or not holding.get('symbol'): continue
                    symbol = str(holding['symbol']).strip().upper()
                    if not symbol: continue
                    value = holding.get('current_value', 0.0)
                    if not isinstance(value, (int, float)): value = 0.0
                    else: value = float(value)
                    allocation = (value / total_val * 100) if total_val > 0 else 0.0
                    portfolio['holdings'].append({"symbol": symbol, "value": value, "allocation": allocation})

            if validate_portfolio_data(portfolio):
                return portfolio
            else:
                 logger.error(f"Processed portfolio FAILED validation for client {client_id}.")
                 return None
        except Exception as e:
            logger.error(f"Error processing raw portfolio for {client_id}: {e}", exc_info=True)
            return None

    def _prepare_client_context(self, client_id: Optional[str], client_context_from_ui: Optional[dict]) -> Optional[dict]:
        """Loads, validates, and prepares client context dictionary."""
        logger.debug(f"Preparing client context. Provided ID: {client_id}, UI Context: {client_context_from_ui is not None}")
        final_context = None
        current_client_id = validate_client_id(client_id)

        if client_context_from_ui and isinstance(client_context_from_ui, dict):
            if validate_portfolio_data(client_context_from_ui):
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
                    processed_data = self._process_db_portfolio(current_client_id, raw_portfolio_data)
                    if processed_data:
                        logger.info(f"Successfully fetched and validated portfolio from DB for {current_client_id}")
                        final_context = processed_data
                    else:
                        logger.warning(f"Processing/validation of DB portfolio for {current_client_id} failed.")
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
        filtered_symbols = {s for s in symbols if s and not s.isdigit() and ('-' not in s and '/' not in s and s not in common_words) or ('-' in s or '/' in s)}
        final_list = sorted(list(filtered_symbols)); logger.info(f"Extracted symbols: {final_list}"); return final_list

    def _load_common_words(self) -> set:
         """Loads a set of common English words to filter potential symbols."""
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
             'NEED', 'NEWS', 'NEXT', 'NICE', 'NINE', 'NONE', 'NOTE', 'ONLY', 'OPEN', 'OVER','PULL', 'PUSH', 'PUTS',
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
             'OBJECT', 'CONTENT', 'ERROR', 'MESSAGE', 'WARNING', 'FAILED', 'CLIENT', 'CONTEXT', 'INVALID',
             'IDEA', 'GOOD', 'BAD', 'INVEST', 'ADD', 'MORE', 'INTO', 'CHECK', 'LOOK', 'STOCK', 'PRICE', 'SHARES',
             'USD', 'QUIT', 'RACE', 'RAIN', 'RATE', 'READ', 'REAL', 'RENT', 'REST', 'TODAY' # Added TODAY based on logs
         }
         return words

    # _get_market_data method: Logic remains same, relying on Financial Agent's ability to produce table
    @log_execution_time
    def _get_market_data(self, symbols: List[str]) -> List[str]:
        """Gets market data using the financial agent (original approach)."""
        market_data_results = []
        financial_agent = self.agents.get('financial')
        if not financial_agent:
            logger.error("Financial agent not initialized.")
            return [f"⚠️ Error: Financial analysis module unavailable."] * len(symbols)

        for symbol in symbols:
            time.sleep(settings.financial_api_delay) # Still useful
            try:
                logger.info(f"Requesting financial data table for symbol: {symbol}")
                # Use the original prompt style asking for the table
                query = f"Provide key financial data table for {symbol}"
                content_str = financial_agent.run(query) # Returns Optional[str]

                if content_str:
                    # Check if agent returned the specific 'not available' message
                    if f"Financial data not available for {symbol}" in content_str:
                        logger.warning(f"Financial agent reported no data for {symbol}.")
                        market_data_results.append(f"⚠️ Data not available for symbol: {symbol}")
                    # Check if agent returned the 'N/A' message for non-stocks
                    elif f"Metrics not applicable for {symbol}" in content_str:
                         logger.warning(f"Financial agent reported metrics N/A for {symbol}.")
                         market_data_results.append(f"⚠️ Metrics not applicable for symbol: {symbol}")
                    else:
                        # Assume the content is the desired markdown table or relevant info
                        prefix = f"**{symbol}**:\n"
                        if not content_str.strip().startswith(("|", "**", f"{symbol}:")):
                            market_data_results.append(prefix + content_str)
                        else:
                            market_data_results.append(content_str)
                        logger.debug(f"Successfully retrieved response for {symbol}.")
                else:
                    logger.warning(f"Financial agent returned None or empty response for {symbol}.")
                    market_data_results.append(f"⚠️ No valid data received for symbol: {symbol}")

            except ModelProviderError as e:
                # Keep specific handling for tool_use_failed, even if less likely now
                error_details = str(e).lower()
                if 'tool_use_failed' in error_details:
                    logger.error(f"Tool use failed unexpectedly for {symbol}: {e}", exc_info=False)
                    market_data_results.append(f"⚠️ Tool execution failed for {symbol}")
                else:
                    logger.error(f"ModelProviderError from financial agent for {symbol}: {e}", exc_info=True)
                    market_data_results.append(f"⚠️ Error retrieving data for {symbol} (AI Model Error)")

            except requests.exceptions.HTTPError as e: # Keep HTTP error handling
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

    # _get_news_summary method remains the same as the last correct version...
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
        except requests.exceptions.HTTPError as e: # Catch specific rate limit or network errors
            if e.response is not None and e.response.status_code == 429:
                 logger.warning(f"DuckDuckGo tool hit rate limit: {e}")
                 return "⚠️ News retrieval failed due to rate limit."
            else:
                 logger.error(f"News agent HTTP error: {e}", exc_info=True)
                 return "⚠️ Error retrieving news (Network/API issue)."
        except ModelProviderError as e: # Catch potential tool use errors here too
             error_details = str(e).lower();
             if 'tool_use_failed' in error_details: logger.error(f"Tool use failed for News Agent: {e}", exc_info=False); return f"⚠️ News retrieval tool execution failed."
             else: logger.error(f"ModelProviderError from news agent: {e}", exc_info=True); return f"⚠️ Error retrieving news (AI Model Error)."
        except Exception as e:
            logger.error(f"News agent query failed unexpectedly: {e}", exc_info=True)
            return "⚠️ Unexpected error retrieving news."


    # _get_vector_search_results method remains the same as the last correct version...
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
                doc_entry_overhead = estimate_token_count(f"\n📄 **Doc {i} ({doc_type}) - Src: {source}**\n\n---") # Shorter header
                if current_token_count + excerpt_tokens + doc_entry_overhead < max_tokens:
                    doc_entry = f"\n📄 **Doc {i} ({doc_type}) - Src: {source}**\n{text_excerpt}\n---"
                    doc_text_aggregate += doc_entry; current_token_count += excerpt_tokens + doc_entry_overhead
                else: logger.warning(f"Doc inclusion token limit ({max_tokens}) reached."); break
            if not doc_text_aggregate: return None
            formatted_docs.append(doc_text_aggregate.strip()); return "\n".join(formatted_docs)
        except Exception as e: logger.error(f"Vector search/format failed: {e}", exc_info=True); return "⚠️ Error retrieving/formatting docs."

    # _get_enhanced_context method remains the same as the last correct version...
    @log_execution_time
    def _get_enhanced_context(self, query: str, client_context: Optional[dict]) -> dict:
        """Builds context dict: client data, agent results, documents."""
        context_dict = {'client': client_context, 'market_data_summary': ["No specific stock symbols identified."], 'news_summary': None, 'relevant_documents': None, 'original_query': query }
        portfolio_symbols_text = " ".join([str(h['symbol']) for h in client_context['holdings']]) if client_context and client_context.get('holdings') else ""
        combined_text = query + " " + portfolio_symbols_text
        symbols = self._extract_symbols(combined_text)
        symbols_to_fetch = symbols[:settings.max_symbols_to_fetch]
        if len(symbols) > settings.max_symbols_to_fetch: logger.warning(f"Too many symbols ({len(symbols)}), limiting fetch to first {settings.max_symbols_to_fetch}: {symbols_to_fetch}")
        if symbols_to_fetch:
            market_data_results = self._get_market_data(symbols_to_fetch)
            context_dict['market_data_summary'] = [summarize_financial_data(r, settings.max_financial_summary_len) if isinstance(r, str) and not r.startswith("⚠️") else r for r in market_data_results]
        context_dict['news_summary'] = self._get_news_summary(query, symbols_to_fetch)
        context_dict['relevant_documents'] = self._get_vector_search_results(query)
        return context_dict

    # _build_prompt method remains the same as the last correct version...
    @log_execution_time
    def _build_prompt(self, context_dict: dict) -> str:
        """Constructs the final prompt string for the coordinator agent."""
        prompt_parts = ["Please analyze the following information and answer the user's query."]
        prompt_parts.append("\n---\n")
        client_context = context_dict.get('client')
        if client_context:
            prompt_parts.append("**Client Information**"); prompt_parts.append(f"- Client ID: `{client_context.get('id', 'N/A')}`")
            if client_context.get('risk_profile'): prompt_parts.append(f"- Risk Profile: {client_context['risk_profile']}")
            if client_context.get('portfolio_value', 0) > 0: prompt_parts.append(f"- Portfolio Value: ${client_context['portfolio_value']:,.2f}")
            holdings = client_context.get('holdings', []);
            if holdings:
                prompt_parts.append(f"- Current Holdings ({len(holdings)} total):")
                sorted_holdings = sorted(holdings, key=lambda x: x.get('value', 0), reverse=True)
                holdings_to_show = sorted_holdings[:settings.max_holdings_in_prompt]
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
            prompt_parts.append("**Requested Market Data Summary**")
            if isinstance(market_data_summary, list):
                 for data_item in market_data_summary: prompt_parts.append(f"- {str(data_item).strip()}")
            else: prompt_parts.append(str(market_data_summary))
            prompt_parts.append("\n---\n")
        relevant_documents = context_dict.get('relevant_documents');
        if relevant_documents: prompt_parts.append(relevant_documents); prompt_parts.append("\n---\n")
        prompt_parts.append("Please provide a comprehensive analysis and response based *only* on the information above. Follow the structure defined in your instructions and include the standard investment disclaimer.")
        final_prompt = "\n".join(prompt_parts); logger.debug(f"Final prompt built. Length: {len(final_prompt)} chars, Est. Tokens: ~{estimate_token_count(final_prompt)}"); return final_prompt

    # get_response method remains the same as the last correct version...
    @log_execution_time
    def get_response(self, query: str, client_id: str = None, client_context: dict = None) -> str:
        """Main method to get a response. Orchestrates context gathering, agent calls, and error handling."""
        final_prompt_string = ""
        try:
            if not validate_query_text(query): logger.warning(f"Invalid query: '{query}'"); return generate_warning_html("Invalid Query", "Please provide a specific question (3-1500 characters).")
            logger.info(f"Processing query for client '{client_id or 'generic'}': '{query[:100]}...'")
            processed_client_context = self._prepare_client_context(client_id, client_context)
            context_dict = self._get_enhanced_context(query, processed_client_context)
            final_prompt_string = self._build_prompt(context_dict)
            estimated_tokens = estimate_token_count(final_prompt_string); logger.info(f"Est. coordinator tokens: ~{estimated_tokens}")
            INPUT_TOKEN_WARNING_THRESHOLD = 30000
            if estimated_tokens > INPUT_TOKEN_WARNING_THRESHOLD: logger.warning(f"Prompt token estimate ({estimated_tokens}) high.")
            coordinator_agent = self.agents.get('coordinator');
            if not coordinator_agent: logger.critical("Coordinator agent missing!"); raise RuntimeError("Coordinator agent not initialized.")
            logger.debug(f"Sending prompt to coordinator ({settings.coordinator_agent_model.id}). Len: {len(final_prompt_string)}")
            response_content = coordinator_agent.run(final_prompt_string)
            if response_content: return format_markdown_response(response_content)
            else: logger.error("Coordinator agent returned None/empty."); return generate_error_html("Processing Error", "AI advisor failed final generation.")
        except ModelProviderError as e:
            error_details = str(e); logger.error(f"Model provider error: {error_details}", exc_info=True)
            if "context_length_exceeded" in error_details.lower() or (hasattr(e, 'code') and e.code == 400 and "maximum context length" in error_details.lower()):
                ft = estimate_token_count(final_prompt_string); logger.error(f"Context Length Exceeded. Est. tokens: ~{ft}")
                return generate_error_html("Request Too Complex", f"Analysis context (~{ft} tokens) exceeded model limits. Simplify request.")
            elif "model_decommissioned" in error_details.lower(): logger.error(f"Model Decommissioned: {error_details}"); return generate_error_html("AI Model Error", f"AI model unavailable/decommissioned. Check config. Details: {error_details}")
            elif "tool_use_failed" in error_details.lower(): logger.error(f"Tool Use Failed Error: {error_details}"); return generate_error_html("AI Tool Error", f"Underlying AI tool failed. Details: {error_details}")
            else: logger.error(f"Failed Prompt Snippet:\n{final_prompt_string[:500]}..."); return generate_error_html("AI Model Error", f"AI model issue. Try again. Details: {error_details}")
        except (RuntimeError, ValueError) as e: logger.error(f"Internal processing error: {str(e)}", exc_info=True); return generate_error_html("Processing Error", f"Internal error occurred: {e}")
        except Exception as e: logger.critical(f"Unexpected critical error: {str(e)}", exc_info=True); logger.critical(traceback.format_exc()); return generate_error_html("Unexpected Error", "Unexpected internal error. Contact support.")