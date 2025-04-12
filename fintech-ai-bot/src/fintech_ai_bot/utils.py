# src/fintech_ai_bot/utils.py

import logging
import os
import sys
import re
import time
import json # Added for format_markdown_response
from functools import wraps
from fintech_ai_bot.config import settings # Import the settings instance
import traceback
import html
from typing import Optional, Callable, Any, Union, Dict # Added types

# --- Logger Setup ---
log_levels = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
numeric_level = log_levels.get(settings.log_level.upper(), logging.INFO)
_loggers_configured = set()

def get_logger(name: str) -> logging.Logger:
    """Enhanced logger setup using centralized configuration."""
    logger = logging.getLogger(name)
    if name in _loggers_configured and logger.hasHandlers():
        return logger

    logger.setLevel(numeric_level)
    logger.propagate = False

    # Console Handler
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        try:
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = logging.Formatter(
                '[%(asctime)s] %(levelname)s in %(module)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S"
            )
            console_handler.setFormatter(console_formatter)
            # Add encoding attempt if needed (often depends on environment)
            # if hasattr(console_handler.stream, 'reconfigure'):
            #     try: console_handler.stream.reconfigure(encoding='utf-8')
            #     except Exception as e_enc: print(f"Console UTF-8 warning: {e_enc}")
            logger.addHandler(console_handler)
        except Exception as e:
            print(f"Warning: Logger '{name}' failed console handler setup: {e}", file=sys.stderr)


    # File Handler
    try:
        settings.log_dir.mkdir(parents=True, exist_ok=True)
        safe_log_name = re.sub(r'[^\w\-_\. ]', '_', name)
        log_file_path = settings.log_dir / f"{safe_log_name}.log"

        if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', None) == str(log_file_path) for h in logger.handlers):
            file_handler = logging.FileHandler(log_file_path, encoding='utf-8', mode='a')
            file_handler.setLevel(numeric_level)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s',
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
    except Exception as e:
        print(f"Error: Logger '{name}' failed to create file handler at '{settings.log_dir}': {e}", file=sys.stderr)
        # traceback.print_exc(file=sys.stderr) # Optional: print full traceback for file handler error

    _loggers_configured.add(name)
    return logger

# Initialize a logger for utils functions themselves
util_logger = get_logger(__name__)


# --- Decorators ---

def log_execution_time(func: Callable) -> Callable:
    """Decorator to log function execution time."""
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
            # Log less detail by default for performance logs, but include type/msg
            logger_perf.error(f"{func.__name__} failed after {duration:.4f}s: {type(e).__name__} - {str(e)}")
            raise # Re-raise the original exception
    return wrapper

# --- Formatting ---

def format_markdown_response(response: Union[str, dict, list]) -> str:
    """Format agent response into markdown, handling various types."""
    if isinstance(response, str):
        text = response.strip()
        text = re.sub(r'\n{3,}', '\n\n', text) # Collapse excessive newlines
        # Basic check to avoid applying markdown conversion to already formatted HTML errors
        if not text.startswith('<div class="error-message">') and not text.startswith('<div class="token-warning">'):
            # You might use a markdown library here if needed, but often raw LLM output is already markdown
            # Example using 'markdown' library (install it: pip install markdown)
            # try:
            #     import markdown
            #     # Convert basic markdown to HTML - adjust extensions as needed
            #     text = markdown.markdown(text, extensions=['tables', 'fenced_code'])
            # except ImportError:
            #     util_logger.warning("Markdown library not installed. Returning raw text.")
            #     pass # Fallback to raw text if library not available
            pass # Keep as raw text which Streamlit markdown usually handles well
        return text
    elif isinstance(response, (dict, list)):
        try:
            return f"```json\n{json.dumps(response, indent=2, ensure_ascii=False)}\n```"
        except Exception as e:
            util_logger.warning(f"Failed to format dict/list as JSON: {e}")
            return f"```\n{str(response)}\n```" # Fallback
    util_logger.debug(f"Formatting non-standard type {type(response)} as string.")
    return str(response)

def generate_error_html(message: str, details: str = "") -> str:
    """Generate formatted HTML error message using consistent styling."""
    escaped_details = html.escape(details) if details else ""
    details_html = f'<p style="font-size: 0.85rem; margin-top: 8px; font-family: monospace; white-space: pre-wrap; color: #fca5a5;">{escaped_details}</p>' if escaped_details else ''
    escaped_message = html.escape(message)
    # Matches CSS class defined in main.py's injected styles
    return f"""
    <div class="error-message">
        <strong>⚠️ Error: {escaped_message}</strong>
        {details_html}
    </div>"""

def generate_warning_html(message: str, details: str = "") -> str:
    """Generate formatted HTML warning message using consistent styling."""
    escaped_details = html.escape(details) if details else ""
    details_html = f'<p style="font-size: 0.9rem; margin-top: 8px; color: #fde68a;">{escaped_details}</p>' if escaped_details else ''
    escaped_message = html.escape(message)
    # Matches CSS class defined in main.py's injected styles
    return f"""
    <div class="token-warning">
        <strong>⚠️ Warning: {escaped_message}</strong>
        {details_html}
    </div>"""

# --- Validation Functions ---

def validate_client_id(client_id: Optional[str]) -> Optional[str]:
    """Validate and normalize client ID."""
    if not client_id: return None
    try:
        client_id_str = str(client_id).strip().upper()
        # Simple validation: alphanumeric, min length 4, starts with letter
        if re.fullmatch(r'[A-Z][A-Z0-9]{3,}', client_id_str):
            util_logger.debug(f"Client ID '{client_id_str}' validated successfully.")
            return client_id_str
        else:
            util_logger.warning(f"Invalid client ID format received: '{client_id_str}'")
            return None
    except Exception as e:
        util_logger.error(f"Error during client ID validation for input '{client_id}': {e}")
        return None

def validate_query_text(text: str) -> bool:
    """Validate user query text for basic quality checks."""
    if not isinstance(text, str):
        util_logger.warning("Query validation failed: Input is not a string.")
        return False
    trimmed_text = text.strip()
    MIN_QUERY_LENGTH = 3
    MAX_QUERY_LENGTH = 1500 # Use config if preferred: settings.max_query_length

    if not trimmed_text:
        util_logger.warning("Query validation failed: Query is empty.")
        return False
    if len(trimmed_text) < MIN_QUERY_LENGTH:
        util_logger.warning(f"Query validation failed: Query too short ('{trimmed_text[:50]}...'). Length: {len(trimmed_text)}")
        return False
    if len(trimmed_text) > MAX_QUERY_LENGTH:
        util_logger.warning(f"Query validation failed: Query exceeds max length {MAX_QUERY_LENGTH}. Length: {len(trimmed_text)}")
        return False
    # Basic check for repetitive characters (optional)
    # if len(set(trimmed_text)) < min(5, len(trimmed_text)):
    #     util_logger.warning("Query validation failed: Appears to be gibberish (low char variety).")
    #     return False

    util_logger.debug("Query text passed basic validation.")
    return True

def validate_portfolio_data(portfolio: Dict[str, Any]) -> bool:
    """Validate portfolio data structure and basic types."""
    if not isinstance(portfolio, dict):
        util_logger.warning("Portfolio validation failed: Input is not a dictionary.")
        return False
    required_fields = {'id', 'portfolio_value', 'holdings'}
    if not required_fields.issubset(portfolio.keys()):
        missing = required_fields - set(portfolio.keys())
        util_logger.warning(f"Portfolio validation failed: Missing required fields: {missing}")
        return False

    # Type checks
    if not isinstance(portfolio.get('id'), str) or not portfolio['id'].strip():
        util_logger.warning("Portfolio validation failed: 'id' is missing or invalid.")
        return False
    if not isinstance(portfolio.get('portfolio_value'), (int, float)):
        util_logger.warning(f"Portfolio validation failed: 'portfolio_value' not num: {type(portfolio.get('portfolio_value'))}).")
        return False
    if 'risk_profile' in portfolio and not isinstance(portfolio.get('risk_profile'), str):
        util_logger.warning(f"Portfolio validation failed: 'risk_profile' not str: {type(portfolio.get('risk_profile'))}).")
        # return False # Decide if this is fatal

    holdings = portfolio.get('holdings')
    if not isinstance(holdings, list):
        util_logger.warning(f"Portfolio validation failed: 'holdings' is not a list ({type(holdings)}).")
        return False

    for i, holding in enumerate(holdings):
        if not isinstance(holding, dict):
            util_logger.warning(f"Portfolio validation failed: Holding index {i} not a dict.")
            return False
        holding_req_fields = {'symbol', 'value', 'allocation'}
        if not holding_req_fields.issubset(holding.keys()):
            h_missing = holding_req_fields - set(holding.keys())
            util_logger.warning(f"Portfolio validation failed: Holding index {i} missing fields: {h_missing}.")
            return False
        if not isinstance(holding.get('symbol'), str) or not holding['symbol'].strip():
            util_logger.warning(f"Portfolio validation failed: Holding index {i} invalid 'symbol'.")
            return False
        if not isinstance(holding.get('value'), (int, float)):
            util_logger.warning(f"Portfolio validation failed: Holding '{holding.get('symbol', 'N/A')}' non-num 'value'.")
            return False
        if not isinstance(holding.get('allocation'), (int, float)):
            util_logger.warning(f"Portfolio validation failed: Holding '{holding.get('symbol', 'N/A')}' non-num 'allocation'.")
            return False

    util_logger.debug(f"Portfolio data for client '{portfolio.get('id', 'N/A')}' validated successfully.")
    return True


# --- Data Processing Helpers ---

def summarize_financial_data(data: str, max_length: int) -> str:
    """Summarize financial data string, attempting to preserve markdown table structure if truncating."""
    if not isinstance(data, str):
        util_logger.warning(f"Cannot summarize non-string data: {type(data)}")
        return str(data)[:max_length] # Fallback

    if len(data) <= max_length:
        return data

    try:
        lines = data.strip().split('\n')
        # Basic check for markdown table structure
        is_table = len(lines) > 2 and \
                   all(line.strip().startswith('|') and line.strip().endswith('|')
                       for line in lines if line.strip() and not line.strip().startswith('|--'))

        if is_table:
            summary = ""
            current_length = 0
            # Include header and separator
            header_sep = lines[0] + "\n" + lines[1] + "\n"
            summary += header_sep
            current_length += len(header_sep)
            truncation_msg = "| ... (truncated) |\n"
            truncation_len = len(truncation_msg)

            for line in lines[2:]:
                # Check if adding the next line plus truncation message fits
                if current_length + len(line) + 1 <= max_length - truncation_len:
                    summary += line + "\n"
                    current_length += len(line) + 1
                else:
                    summary += truncation_msg
                    break
            # If loop finished without adding truncation message, ensure it fits
            if not summary.endswith(truncation_msg) and current_length > max_length:
                 # This case is less likely with the check inside loop, but as fallback:
                 # Rebuild summary ensuring truncation fits
                 summary = header_sep
                 current_length = len(header_sep)
                 for line in lines[2:]:
                     if current_length + len(line) + 1 <= max_length - truncation_len:
                         summary += line + "\n"
                         current_length += len(line) + 1
                     else: break
                 summary += truncation_msg

            util_logger.debug("Summarized financial data by truncating table rows.")
            return summary.strip()
        else:
            # Fallback: Simple character truncation
            util_logger.debug("Summarizing financial data by simple character truncation.")
            # Try to cut at last space before max_length
            truncated_point = data[:max_length - 3].rfind(' ') # Leave space for "..."
            if truncated_point > max_length * 0.7: # Cut at space if reasonable
                return data[:truncated_point] + "..."
            else: # Otherwise hard cut
                return data[:max_length - 3] + "..."

    except Exception as e:
        util_logger.error(f"Failed to summarize financial data: {str(e)}", exc_info=True)
        return data[:max_length - 3].strip() + "..." # Safe fallback


# --- Token Estimation ---

def estimate_token_count(text: str) -> int:
     """Estimate token count using a simple character heuristic."""
     if not text or not isinstance(text, str): return 0
     # Heuristic: Approx 1 token per 4 chars (adjust if using specific tokenizer like tiktoken)
     return max(1, len(text) // 4)