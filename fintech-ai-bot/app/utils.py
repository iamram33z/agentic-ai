import logging
import os
import requests
import json
from datetime import datetime


class BetterStackHandler(logging.Handler):
    def __init__(self, api_token):
        super().__init__()
        self.api_token = api_token
        self.url = "https://in.logs.betterstack.com/v1/logs"

    def emit(self, record):
        log_entry = {
            "dt": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "message": self.format(record),
            "metadata": {
                "service": "fintech-ai-bot",
                "module": record.name
            }
        }

        try:
            requests.post(
                self.url,
                headers={"Authorization": f"Bearer {self.api_token}"},
                json=log_entry,
                timeout=3
            )
        except Exception as e:
            print(f"Failed to send log to BetterStack: {e}")


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # BetterStack if configured
    if api_token := os.getenv("BETTERSTACK_API_TOKEN"):
        logger.addHandler(BetterStackHandler(api_token))

    # Console logging
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger