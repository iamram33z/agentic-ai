import logging
import os
import requests
import json
from typing import Optional, Callable, Any
from functools import wraps
from datetime import datetime
import re
import markdown
from bs4 import BeautifulSoup
import time


class BetterStackHandler(logging.Handler):
    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key
        self.endpoint = "https://in.logs.betterstack.com/v1/logs"
        self.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))

    def emit(self, record):
        log_entry = {
            "message": self.format(record),
            "level": record.levelname,
            "metadata": {
                "service": "fintech-ai-bot",
                "module": record.name,
                "timestamp": datetime.utcnow().isoformat()
            }
        }

        try:
            requests.post(
                self.endpoint,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=log_entry,
                timeout=3
            )
        except Exception as e:
            print(f"BetterStack logging failed: {e}")


def get_logger(name: str) -> logging.Logger:
    """Enhanced logger with file and remote logging"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
    ))
    logger.addHandler(console_handler)

    # File handler
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(file_handler)

    # BetterStack handler if configured
    if api_key := os.getenv("BETTERSTACK_API_KEY"):
        logger.addHandler(BetterStackHandler(api_key))

    return logger


def retry_db_operation(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retrying database operations"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        time.sleep(delay * (attempt + 1))
                        continue
                    raise last_exception

        return wrapper

    return decorator

def validate_client_id(client_id: Optional[str]) -> Optional[str]:
    """Validate and normalize client ID"""
    if not client_id:
        return None

    client_id = str(client_id).strip().upper()
    if re.match(r'^CLIENT\d{3}$', client_id):
        return client_id
    return None


def format_markdown_response(text: str) -> str:
    """Enhanced markdown formatting with HTML sanitization"""
    if not text:
        return ""

    try:
        # Convert markdown to HTML
        html = markdown.markdown(text)

        # Sanitize HTML
        soup = BeautifulSoup(html, 'html.parser')

        # Allow only certain tags and attributes
        allowed_tags = {
            'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
            'p', 'br', 'div', 'span',
            'ul', 'ol', 'li',
            'strong', 'em', 'b', 'i', 'u',
            'table', 'thead', 'tbody', 'tr', 'th', 'td',
            'a'
        }

        for tag in soup.find_all(True):
            if tag.name not in allowed_tags:
                tag.unwrap()
            elif tag.name == 'a':
                tag.attrs['target'] = '_blank'
                tag.attrs['rel'] = 'noopener noreferrer'
            else:
                tag.attrs = {}

        return str(soup)

    except Exception as e:
        logger = get_logger("Formatting")
        logger.warning(f"Failed to format markdown: {str(e)}")
        return text


def log_execution_time(func: Callable) -> Callable:
    """Decorator to log function execution time"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger("Performance")
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(f"{func.__name__} executed in {duration:.2f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{func.__name__} failed after {duration:.2f}s: {str(e)}")
            raise

    return wrapper