# src/fintech_ai_bot/utils.py

import logging
import os
import sys
import re
import time
import json
import html
from functools import wraps
from pathlib import Path
from typing import Optional, Callable, Any, Union, Dict, List
from transformers import PreTrainedTokenizerBase # Added for optional token counting

# --- Initialize Logger ---
# Attempt to import settings, but handle failure gracefully for utils standalone use
try:
    from fintech_ai_bot.config import settings
    _log_level_str = settings.log_level
    _log_dir = settings.log_dir
except ImportError:
    settings = None # Indicate settings are not available
    _log_level_str = "INFO" # Default log level
    _log_dir = Path("./logs") # Default log directory relative to execution
    print(f"Warning [utils.py]: Could not import config. Using default logger settings (Level: {_log_level_str}, Dir: {_log_dir}).", file=sys.stderr)

# --- Logger Setup ---
_log_levels = { # Renamed to avoid conflict
    "DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING,
    "ERROR": logging.ERROR, "CRITICAL": logging.CRITICAL,
}
_numeric_level = _log_levels.get(_log_level_str.upper(), logging.INFO)
_loggers_configured = set()

def get_logger(name: str) -> logging.Logger:
    """
    Sets up and returns a logger instance with console and file handlers.
    Uses settings from config.py if available, otherwise defaults.
    Ensures handlers are not added multiple times.
    """
    logger = logging.getLogger(name)
    # Check if logger already configured (by name) and has handlers
    if name in _loggers_configured and logger.hasHandlers():
        return logger

    # Configure logger (level, propagation)
    logger.setLevel(_numeric_level)
    logger.propagate = False # Prevent root logger from handling messages again

    # Remove existing handlers if reconfiguring (e.g., level change) - safer to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # --- Console Handler ---
    try:
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s in %(module)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    except Exception as e:
        print(f"Warning [utils.py]: Logger '{name}' failed console handler setup: {e}", file=sys.stderr)

    # --- File Handler ---
    try:
        # Use _log_dir which is set based on settings availability
        _log_dir.mkdir(parents=True, exist_ok=True)
        # Sanitize logger name for filename
        safe_log_name = re.sub(r'[^\w\-_\. ]', '_', name)
        log_file_path = _log_dir / f"{safe_log_name}.log"

        file_handler = logging.FileHandler(log_file_path, encoding='utf-8', mode='a')
        file_handler.setLevel(_numeric_level) # Set level on file handler too
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s',
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        # Log error to stderr if file logging fails
        print(f"Error [utils.py]: Logger '{name}' failed file handler setup at '{_log_dir}': {e}", file=sys.stderr)

    _loggers_configured.add(name) # Mark logger as configured
    return logger

# Initialize a logger for utils functions themselves
util_logger = get_logger(__name__)
util_logger.debug("Utils module logger initialized.")

# --- Decorators ---
def log_execution_time(func: Callable) -> Callable:
    """Decorator to log the execution time of a function."""
    # Create a specific performance logger if needed, or use util_logger
    perf_logger = get_logger(f"performance.{func.__module__}")
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.monotonic()
        try:
            result = func(*args, **kwargs)
            duration = time.monotonic() - start_time
            perf_logger.debug(f"'{func.__name__}' executed in {duration:.4f}s") # Use DEBUG for less noise
            return result
        except Exception as e:
            duration = time.monotonic() - start_time
            # Log error with duration and exception details
            perf_logger.error(f"'{func.__name__}' failed after {duration:.4f}s: {type(e).__name__} - {e}", exc_info=False) # exc_info=False for brevity
            raise # Re-raise the original exception
    return wrapper

# --- Formatting ---
def format_markdown_response(response: Union[str, dict, list]) -> str:
    """
    Formats various response types into a markdown string suitable for display.
    Handles basic cleanup and JSON formatting.
    """
    try:
        if isinstance(response, str):
            text = response.strip()
            # Collapse excessive newlines, but preserve single/double newlines
            text = re.sub(r'\n{3,}', '\n\n', text)
            # Basic HTML tag stripping (optional, adjust regex if needed)
            # text = re.sub(r'<[^>]+>', '', text)
            # Ensure basic code blocks are preserved if already present
            if '```' in text:
                 return text # Assume already formatted if it contains code blocks
            # Otherwise, treat as plain text (Streamlit markdown usually handles it)
            return text
        elif isinstance(response, (dict, list)):
            # Pretty-print JSON within a markdown code block
            return f"```json\n{json.dumps(response, indent=2, ensure_ascii=False)}\n```"
        else:
            # Fallback for other types
            util_logger.debug(f"Formatting non-standard type {type(response)} as string.")
            return f"```\n{str(response)}\n```"
    except Exception as e:
        util_logger.error(f"Error formatting response: {e}", exc_info=True)
        return f"```\nError formatting response: {str(response)}\n```" # Return error indication

def _generate_html_message(level: str, message: str, details: str = "") -> str:
    """Internal helper to generate styled HTML messages."""
    level_emoji = {"ERROR": "⚠️", "WARNING": "ℹ️"}.get(level.upper(), "ℹ️")
    level_color = {"ERROR": "#fca5a5", "WARNING": "#fde68a"}.get(level.upper(), "#fde68a") # text colors
    level_bg_color = {"ERROR": "#fee2e2", "WARNING": "#fef9c3"}.get(level.upper(), "#fef9c3") # background colors
    level_border_color = {"ERROR": "#ef4444", "WARNING": "#facc15"}.get(level.upper(), "#facc15") # border colors

    escaped_message = html.escape(message)
    escaped_details = html.escape(details) if details else ""
    details_html = f'<p style="font-size: 0.85rem; margin-top: 8px; font-family: monospace; white-space: pre-wrap; color: #57534e;">{escaped_details}</p>' if escaped_details else '' # details color

    # Define CSS classes or use inline styles (inline styles used here for simplicity)
    return f"""
    <div style="border-left: 4px solid {level_border_color}; background-color: {level_bg_color}; padding: 10px 15px; margin: 10px 0; border-radius: 4px;">
        <strong style="color: {level_border_color};">{level_emoji} {level.capitalize()}: {escaped_message}</strong>
        {details_html}
    </div>"""

def generate_error_html(message: str, details: str = "") -> str:
    """Generates formatted HTML error message."""
    return _generate_html_message("ERROR", message, details)

def generate_warning_html(message: str, details: str = "") -> str:
    """Generates formatted HTML warning message."""
    return _generate_html_message("WARNING", message, details)


# --- Validation Functions ---
@log_execution_time
def validate_client_id(client_id: Optional[str]) -> Optional[str]:
    """Validates and normalizes a client ID string."""
    if not client_id or not isinstance(client_id, str):
        util_logger.warning(f"Client ID validation failed: Input invalid or not string: {client_id}")
        return None
    try:
        client_id_str = client_id.strip().upper()
        # Example validation: Must be alphanumeric, start with letter, length 4-10
        if re.fullmatch(r'[A-Z][A-Z0-9]{3,9}', client_id_str):
            util_logger.debug(f"Client ID '{client_id_str}' validated.")
            return client_id_str
        else:
            util_logger.warning(f"Invalid client ID format rejected: '{client_id_str}'")
            return None
    except Exception as e:
        util_logger.error(f"Error during client ID validation for input '{client_id}': {e}", exc_info=True)
        return None

@log_execution_time
def validate_query_text(text: str) -> bool:
    """Performs basic validation checks on user query text."""
    if not isinstance(text, str):
        util_logger.warning("Query validation failed: Input not a string.")
        return False
    trimmed_text = text.strip()
    # Use config for limits if available, otherwise use constants
    min_len = getattr(settings, 'min_query_length', 3) if settings else 3
    max_len = getattr(settings, 'max_query_length', 2000) if settings else 2000

    if not trimmed_text:
        util_logger.warning("Query validation failed: Query is empty.")
        return False
    if len(trimmed_text) < min_len:
        util_logger.warning(f"Query validation failed: Query too short (< {min_len} chars).")
        return False
    if len(trimmed_text) > max_len:
        util_logger.warning(f"Query validation failed: Query too long (> {max_len} chars).")
        return False

    util_logger.debug("Query text passed basic validation.")
    return True

@log_execution_time
def validate_portfolio_data(portfolio: Optional[Dict[str, Any]]) -> bool:
    """Validates the structure and basic types of portfolio data."""
    if not portfolio or not isinstance(portfolio, dict):
        util_logger.warning(f"Portfolio validation failed: Input not a non-empty dict: {type(portfolio)}")
        return False

    # Check top-level required fields
    required_fields = {'client_id', 'client_name', 'portfolio_value', 'holdings'} # Adjusted based on common needs
    if not required_fields.issubset(portfolio.keys()):
        missing = required_fields - set(portfolio.keys())
        util_logger.warning(f"Portfolio validation failed: Missing top-level fields: {missing}")
        return False

    # Type and value checks for top-level fields
    if not isinstance(portfolio['client_id'], str) or not portfolio['client_id'].strip(): util_logger.warning("Invalid 'client_id'"); return False
    if not isinstance(portfolio['client_name'], str) or not portfolio['client_name'].strip(): util_logger.warning("Invalid 'client_name'"); return False
    if not isinstance(portfolio['portfolio_value'], (int, float)) or portfolio['portfolio_value'] < 0: util_logger.warning("Invalid 'portfolio_value'"); return False
    if 'risk_profile' in portfolio and not isinstance(portfolio['risk_profile'], str): util_logger.warning("Invalid 'risk_profile' type"); # Optional: return False

    # Check holdings list
    holdings = portfolio.get('holdings')
    if not isinstance(holdings, list):
        util_logger.warning(f"Portfolio validation failed: 'holdings' is not a list ({type(holdings)}).")
        return False

    # Check each holding within the list
    holding_req_fields = {'symbol', 'value', 'allocation'}
    for i, holding in enumerate(holdings):
        if not isinstance(holding, dict): util_logger.warning(f"Holding {i} not a dict."); return False
        if not holding_req_fields.issubset(holding.keys()): util_logger.warning(f"Holding {i} missing fields: {holding_req_fields - set(holding.keys())}."); return False
        if not isinstance(holding['symbol'], str) or not holding['symbol'].strip(): util_logger.warning(f"Holding {i} invalid 'symbol'."); return False
        if not isinstance(holding['value'], (int, float)) or holding['value'] < 0: util_logger.warning(f"Holding '{holding['symbol']}' invalid 'value'."); return False
        if not isinstance(holding['allocation'], (int, float)) or not (0 <= holding['allocation'] <= 100): util_logger.warning(f"Holding '{holding['symbol']}' invalid 'allocation'."); return False
        # Add more checks per holding if needed (e.g., 'name', 'type')

    util_logger.debug(f"Portfolio data for client '{portfolio['client_id']}' validated successfully.")
    return True


# --- Data Processing Helpers ---
@log_execution_time
def summarize_financial_data(data: str, max_length: int) -> str:
    """Summarize financial data string, attempting smart truncation."""
    if not isinstance(data, str): util_logger.warning(f"Cannot summarize non-string: {type(data)}"); return str(data)[:max_length]
    if len(data) <= max_length: return data
    if max_length <= 3: return "..." # Handle edge case

    util_logger.debug(f"Attempting to summarize data of length {len(data)} to {max_length} chars.")
    # Simple character truncation, trying to preserve whole words/lines
    try:
        # Try cutting at the last newline before max_length
        newline_cut = data[:max_length - 3].rfind('\n')
        if newline_cut > max_length * 0.6: # If newline found reasonably far in
            return data[:newline_cut].strip() + "\n..."

        # Try cutting at the last space before max_length
        space_cut = data[:max_length - 3].rfind(' ')
        if space_cut > max_length * 0.6: # If space found reasonably far in
            return data[:space_cut].strip() + "..."

        # Fallback: Hard character cut
        return data[:max_length - 3] + "..."
    except Exception as e:
        util_logger.error(f"Error during summarization: {e}", exc_info=True)
        return data[:max_length - 3] + "..." # Safe fallback


# --- Token Estimation ---
def estimate_token_count(text: str, tokenizer: Optional[PreTrainedTokenizerBase] = None) -> int:
    """
    Estimate token count. Uses provided tokenizer if available, otherwise a character heuristic.
    """
    if not text or not isinstance(text, str): return 0

    if tokenizer:
        try:
            # Use tokenizer for accurate count (handles special tokens)
            # Note: Depending on usage, you might want add_special_tokens=True/False
            return len(tokenizer.encode(text, add_special_tokens=True))
        except Exception as e:
            util_logger.warning(f"Tokenizer encode failed during estimation: {e}. Falling back to heuristic.")
            # Fallback to heuristic if tokenizer fails unexpectedly

    # Heuristic: Approx 1 token per 4 chars (adjust factor based on typical language/model)
    char_to_token_factor = 4
    estimated_count = max(1, len(text) // char_to_token_factor)
    # util_logger.debug(f"Estimated token count using heuristic: {estimated_count}")
    return estimated_count