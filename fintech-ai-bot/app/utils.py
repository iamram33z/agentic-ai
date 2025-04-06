# utils.py
import logging
import os
import requests # Still potentially useful for other things, keep for now
import json
from functools import wraps
from datetime import datetime
import re
import markdown
from bs4 import BeautifulSoup # Keep if needed for other formatting later
import time
from collections import defaultdict
from typing import Optional, Callable, Any, Union
import sys # Import sys
from dotenv import load_dotenv
import html # For escaping HTML in error messages
import traceback # For logging tracebacks

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
log_levels = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
numeric_level = log_levels.get(LOG_LEVEL, logging.INFO)

# Use a flag to ensure setup happens only once per logger name
_loggers_configured = set()

def get_logger(name: str) -> logging.Logger:
    """Enhanced logger with file and console (UTF-8) logging."""
    logger = logging.getLogger(name)

    # Check if this specific logger has already been configured
    if name in _loggers_configured and logger.hasHandlers():
        return logger

    logger.setLevel(numeric_level)
    logger.propagate = False # Prevent duplicate logs in parent/root logger

    # --- Console handler with UTF-8 ---
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        try:
            console_handler = logging.StreamHandler(sys.stdout) # Use sys.stdout
            console_formatter = logging.Formatter(
                '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            console_handler.setFormatter(console_formatter)
            # Attempt to set UTF-8 encoding for the stream
            console_handler.stream.reconfigure(encoding='utf-8')
        except AttributeError:
            print(f"Warning: Logger '{name}' could not reconfigure stream encoding. Ensure console supports UTF-8 or set PYTHONIOENCODING=utf-8.", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Logger '{name}' failed to set console handler encoding to UTF-8: {e}", file=sys.stderr)
        finally:
             if 'console_handler' in locals(): # Ensure handler was created before adding
                 logger.addHandler(console_handler)


    # --- File handler with UTF-8 ---
    try:
        # Define log directory (adjust path as needed, e.g., relative to project root)
        # Assuming utils.py is in app/, logs is sibling to app/
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_dir = os.path.join(project_root, "logs")
        os.makedirs(log_dir, exist_ok=True)
        # Sanitize logger name for use in filename
        safe_log_name = re.sub(r'[^\w\-_\. ]', '_', name)
        log_file_path = os.path.join(log_dir, f"{safe_log_name}.log")

        # Check if a file handler for this path already exists
        if not any(isinstance(h, logging.FileHandler) and hasattr(h, 'baseFilename') and h.baseFilename == log_file_path for h in logger.handlers):
            # Use UTF-8 encoding for the file handler
            file_handler = logging.FileHandler(log_file_path, encoding='utf-8', mode='a') # Append mode
            file_handler.setLevel(numeric_level) # Set level for this handler
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(message)s',
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
    except Exception as e:
        # Log error regarding file handler setup to console as fallback
        print(f"Error: Logger '{name}' failed to create file handler at '{log_dir}': {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)


    # --- BetterStack Handler Removed ---

    # Mark this logger name as configured
    _loggers_configured.add(name)

    return logger


# Initialize a logger for utils functions themselves
util_logger = get_logger(__name__)

# --- Utility Functions ---

def retry_db_operation(max_retries: int = 3, delay: float = 1.0, allowed_exceptions=(Exception,)):
    """Decorator for retrying database operations with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except allowed_exceptions as e:
                    last_exception = e
                    wait_time = delay * (2 ** attempt) # Exponential backoff
                    util_logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed for {func.__name__}. "
                        f"Error: {e}. Retrying in {wait_time:.2f}s..."
                    )
                    if attempt < max_retries - 1:
                        time.sleep(wait_time)
                        continue
                    else:
                         util_logger.error(
                             f"All {max_retries} retry attempts failed for {func.__name__}. Last error: {e}",
                             exc_info=True # Log traceback on final failure
                         )
                         raise last_exception # Re-raise the last exception
                except Exception as e: # Catch unexpected exceptions
                    util_logger.error(f"Unexpected error during retry attempt for {func.__name__}: {e}", exc_info=True)
                    raise # Re-raise immediately

        return wrapper
    return decorator

def validate_client_id(client_id: Optional[str]) -> Optional[str]:
    """Validate and normalize client ID."""
    if not client_id: return None
    try:
        client_id_str = str(client_id).strip().upper()
        # Example pattern: CLIENT followed by 3 or more digits
        if re.fullmatch(r'CLIENT\d{3,}', client_id_str): # Use fullmatch
            util_logger.debug(f"Client ID '{client_id_str}' validated successfully.")
            return client_id_str
        else:
            util_logger.warning(f"Invalid client ID format received: '{client_id_str}'")
            return None
    except Exception as e:
        util_logger.error(f"Error during client ID validation for input '{client_id}': {e}")
        return None

def format_markdown_response(response: Union[str, dict, list]) -> str:
    """Format agent response into markdown, handling various types."""
    if isinstance(response, str):
        text = response.strip()
        text = re.sub(r'\n{3,}', '\n\n', text) # Collapse excessive newlines
        return text
    if isinstance(response, (dict, list)):
        try:
            # Format as a markdown JSON code block
            return f"```json\n{json.dumps(response, indent=2, ensure_ascii=False)}\n```"
        except Exception as e:
            util_logger.warning(f"Failed to format dict/list as JSON: {e}")
            return f"```\n{str(response)}\n```" # Fallback string representation
    util_logger.debug(f"Formatting non-standard type {type(response)} as string.")
    return str(response) # Fallback for other types

def log_execution_time(func: Callable) -> Callable:
    """Decorator to log function execution time."""
    # Create logger specific to the function's module for better organization
    logger_perf = get_logger(f"Performance.{func.__module__}")

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.monotonic()
        try:
            result = func(*args, **kwargs)
            duration = time.monotonic() - start_time
            logger_perf.info(f"{func.__name__} executed in {duration:.4f}s")
            return result
        except Exception as e:
            duration = time.monotonic() - start_time
            logger_perf.error(f"{func.__name__} failed after {duration:.4f}s: {type(e).__name__} - {str(e)}", exc_info=True)
            raise
    return wrapper

def estimate_token_count(text: str) -> int:
    """Estimate token count using a simple character heuristic."""
    if not text or not isinstance(text, str): return 0
    # Heuristic: Approx 1 token per 4 chars (adjust if using specific tokenizer like tiktoken)
    return max(1, len(text) // 4)

def validate_portfolio_data(portfolio: dict) -> bool:
    """Validate portfolio data structure and basic types."""
    if not isinstance(portfolio, dict):
        util_logger.warning("Portfolio validation failed: Input is not a dictionary.")
        return False
    required_fields = {'holdings', 'portfolio_value', 'id', 'risk_profile'}
    if not all(field in portfolio for field in required_fields):
        missing = required_fields - set(portfolio.keys())
        util_logger.warning(f"Portfolio validation failed: Missing required fields: {missing}")
        return False
    # Type checks (allow None for optional fields like risk_profile if intended)
    if not isinstance(portfolio.get('id'), str) or not portfolio['id'].strip():
        util_logger.warning("Portfolio validation failed: 'id' is missing or invalid.")
        return False
    if not isinstance(portfolio.get('portfolio_value'), (int, float)):
        util_logger.warning(f"Portfolio validation failed: 'portfolio_value' not num: {type(portfolio.get('portfolio_value'))}).")
        return False
    if not isinstance(portfolio.get('risk_profile'), str): # Check risk_profile is string (can be empty)
        util_logger.warning(f"Portfolio validation failed: 'risk_profile' not str: {type(portfolio.get('risk_profile'))}).")
        # return False # Decide if empty string is allowed
    holdings = portfolio.get('holdings')
    if not isinstance(holdings, list):
        util_logger.warning(f"Portfolio validation failed: 'holdings' is not a list ({type(holdings)}).")
        return False
    for i, holding in enumerate(holdings):
        if not isinstance(holding, dict):
            util_logger.warning(f"Portfolio validation failed: Holding index {i} not a dict.")
            return False
        holding_req_fields = {'symbol', 'value', 'allocation'}
        if not all(hf in holding for hf in holding_req_fields):
            h_missing = holding_req_fields - set(holding.keys())
            util_logger.warning(f"Portfolio validation failed: Holding index {i} missing fields: {h_missing}.")
            return False
        if not isinstance(holding['symbol'], str) or not holding['symbol'].strip():
            util_logger.warning(f"Portfolio validation failed: Holding index {i} invalid 'symbol'.")
            return False
        if not isinstance(holding['value'], (int, float)):
            util_logger.warning(f"Portfolio validation failed: Holding '{holding['symbol']}' non-num 'value'.")
            return False
        if not isinstance(holding['allocation'], (int, float)):
            util_logger.warning(f"Portfolio validation failed: Holding '{holding['symbol']}' non-num 'allocation'.")
            return False
    util_logger.debug(f"Portfolio data for client '{portfolio['id']}' validated successfully.")
    return True

def summarize_financial_data(data: str, max_length: int = 500) -> str:
    """Summarize financial data, prioritizing markdown table content."""
    if not isinstance(data, str):
        util_logger.warning(f"Cannot summarize non-string data: {type(data)}")
        return str(data)[:max_length] # Fallback
    try:
        # Extract markdown table lines (excluding separators)
        table_pattern = re.compile(r'^\s*\|(?!-+--)(.+?)\|\s*$', re.MULTILINE)
        matches = table_pattern.findall(data)
        if matches:
            table_lines = [f"|{match.strip()}|" for match in matches]
            summary = "\n".join(table_lines)
            if len(summary) > max_length: # Truncate if needed
                summary = summary[:max_length].rsplit('\n', 1)[0] + "\n| ... (truncated) |" # Truncate cleanly at line break
            if len(summary.strip()) > 10:
                 util_logger.debug("Summarized financial data using table extraction.")
                 return summary
        # Fallback: If no table or extraction failed, just truncate
        util_logger.debug("Summarizing financial data by simple truncation.")
        return data[:max_length].strip() + ("..." if len(data) > max_length else "")
    except Exception as e:
        util_logger.error(f"Failed to summarize financial data: {str(e)}", exc_info=True)
        return data[:max_length].strip() + ("..." if len(data) > max_length else "") # Safe fallback

def generate_error_html(message: str, details: str = "") -> str:
    """Generate formatted HTML error message using assumed Streamlit CSS classes."""
    escaped_details = html.escape(details) if details else ""
    details_html = f'<p style="font-size: 0.85rem; margin-top: 8px; font-family: monospace; white-space: pre-wrap; color: #fca5a5;">{escaped_details}</p>' if escaped_details else ''
    escaped_message = html.escape(message)
    # Ensure class name matches CSS in main.py
    return f"""
    <div class="error-message">
        <strong>⚠️ Error: {escaped_message}</strong>
        {details_html}
    </div>
    """

def generate_warning_html(message: str, details: str = "") -> str:
    """Generate formatted HTML warning message using assumed Streamlit CSS classes."""
    escaped_details = html.escape(details) if details else ""
    details_html = f'<p style="font-size: 0.9rem; margin-top: 8px; color: #fde68a;">{escaped_details}</p>' if escaped_details else ''
    escaped_message = html.escape(message)
    # Ensure class name matches CSS in main.py
    return f"""
    <div class="token-warning">
        <strong>⚠️ Warning: {escaped_message}</strong>
        {details_html}
    </div>
    """

def validate_query_text(text: str) -> bool:
    """Validate user query text for basic quality checks."""
    if not isinstance(text, str):
         util_logger.warning("Query validation failed: Input is not a string.")
         return False
    trimmed_text = text.strip()
    MIN_QUERY_LENGTH = 3
    MAX_QUERY_LENGTH = 1500 # Increased slightly
    if not trimmed_text:
         util_logger.warning("Query validation failed: Query is empty.")
         return False
    if len(trimmed_text) < MIN_QUERY_LENGTH:
         util_logger.warning(f"Query validation failed: Query too short ('{trimmed_text}').")
         return False
    if len(trimmed_text) > MAX_QUERY_LENGTH:
        util_logger.warning(f"Query validation failed: Query exceeds max length {MAX_QUERY_LENGTH}.")
        return False
    util_logger.debug("Query text passed basic validation.")
    return True

# Define asset type constants
ASSET_TYPE_STOCK = "stock"
ASSET_TYPE_ETF = "etf"
ASSET_TYPE_CRYPTO = "crypto"
ASSET_TYPE_FOREX = "forex"
ASSET_TYPE_INDEX = "index"
ASSET_TYPE_MUTUAL_FUND = "mutual_fund"
ASSET_TYPE_OPTION = "option"
ASSET_TYPE_FUTURES = "futures"
ASSET_TYPE_UNKNOWN = "unknown"

# Expanded known lists for asset type detection
KNOWN_ETFS = {
    'SPY', 'IVV', 'VOO', 'QQQ', 'VTI', 'VEA', 'VWO', 'GLD', 'SLV', 'USO', 'UNG', 'TLT', 'HYG', 'LQD',
    'XLE', 'XLF', 'XLK', 'XLV', 'XLI', 'XLU', 'XLB', 'XLP', 'XLY', 'DIA', 'IWM', 'EFA', 'EEM', 'AGG',
    'BND', 'ARKK', 'ARKG', 'ARKW', 'ARKF', 'ARKQ', 'SOXX', 'SMH', 'IBB', 'XBI', 'GDX', 'GDXJ', 'IYR',
    'VNQ', 'SCHD', 'VIG', 'VYM', 'VO', 'VB', 'VUG', 'VGT', 'FXI', 'EWZ', 'EWW', 'INDA', 'RSX', 'URA',
    'TAN', 'ICLN', 'PBW', 'JETS', 'PEJ', 'IYT', 'IAU', 'DBC', 'DBA', 'TIP', 'SHY', 'IEF', 'GOVT'
}
KNOWN_CRYPTO = {
    'BTC', 'ETH', 'XRP', 'LTC', 'BCH', 'ADA', 'DOT', 'SOL', 'DOGE', 'SHIB', 'LINK', 'MATIC', 'XLM',
    'USDT', 'USDC', 'BUSD', 'DAI', 'AVAX', 'TRX', 'WBTC', 'ETC', 'XMR', 'ATOM', 'UNI', 'FIL'
}
KNOWN_INDICES = { 'SPX', 'NDX', 'DJI', 'VIX', 'FTSE', 'DAX', 'N225', 'HSI', 'RUT', 'STOXX50E' }
KNOWN_CCY = {'USD', 'EUR', 'JPY', 'GBP', 'CHF', 'CAD', 'AUD', 'NZD', 'CNY', 'HKD', 'SGD', 'KRW', 'INR', 'RUB', 'BRL', 'ZAR', 'MXN'}

def get_asset_type(symbol: str) -> str:
    """Determine asset type based on symbol conventions and known lists."""
    if not symbol or not isinstance(symbol, str): return ASSET_TYPE_UNKNOWN
    symbol = symbol.strip().upper()
    symbol_len = len(symbol)
    # 1. Prefixes/Suffixes/Separators
    if symbol.startswith('^') or symbol.startswith('.'): return ASSET_TYPE_INDEX
    if symbol.endswith('=F'): return ASSET_TYPE_FUTURES
    if '=X' in symbol:
        base = symbol.split('=X')[0]
        if len(base) == 6 and base[:3] in KNOWN_CCY and base[3:] in KNOWN_CCY: return ASSET_TYPE_FOREX
    if '/' in symbol and symbol_len == 7: # EUR/USD
         parts = symbol.split('/')
         if len(parts) == 2 and parts[0] in KNOWN_CCY and parts[1] in KNOWN_CCY: return ASSET_TYPE_FOREX
    if '-' in symbol: # Crypto Pair (e.g., BTC-USD)
        parts = symbol.split('-')
        if len(parts) == 2 and (parts[0] in KNOWN_CRYPTO or parts[1] in KNOWN_CRYPTO or parts[1] in ['USD', 'EUR', 'GBP', 'USDT']): return ASSET_TYPE_CRYPTO
    # 2. Known Lists
    if symbol in KNOWN_ETFS: return ASSET_TYPE_ETF
    if symbol in KNOWN_CRYPTO: return ASSET_TYPE_CRYPTO
    if symbol in KNOWN_INDICES: return ASSET_TYPE_INDEX
    # 3. Pattern Matching
    option_pattern = r'^[A-Z]{1,5}\d{6}[CP]\d{8}$'
    if re.fullmatch(option_pattern, symbol): return ASSET_TYPE_OPTION
    futures_pattern = r'^[A-Z]{1,4}[FGHJKMNQUVXZ]\d{1,2}$' # Allows 4 char root like ZW WN
    if re.fullmatch(futures_pattern, symbol): return ASSET_TYPE_FUTURES
    if symbol_len == 6 and symbol.isalpha() and symbol[:3] in KNOWN_CCY and symbol[3:] in KNOWN_CCY: return ASSET_TYPE_FOREX # GBPJPY
    if symbol_len == 5 and symbol.endswith('X') and symbol[:-1].isalpha(): return ASSET_TYPE_MUTUAL_FUND
    # 4. Fallback to Stock (common case for 1-5 alpha chars)
    if 1 <= symbol_len <= 5 and symbol.isalpha(): return ASSET_TYPE_STOCK
    # Consider 1-4 char symbols with numbers as stocks too (e.g., BRK.B -> need custom handle, 7203.T)
    # For now, keep it simple.
    util_logger.debug(f"Could not determine asset type for symbol '{symbol}', defaulting to unknown.")
    return ASSET_TYPE_UNKNOWN