# src/fintech_ai_bot/config.py
# CORRECTED - Updated coordinator model ID

import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, PostgresDsn, HttpUrl
from dotenv import load_dotenv

# Load .env file from the root of the 'fintech-ai-bot' directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

class AgentModelConfig(BaseSettings):
    id: str = "llama3-8b-8192" # Default to a common, available model
    temperature: float = 0.5
    max_tokens: int = 1000 # Default max_tokens

class Settings(BaseSettings):
    # --- Project Paths ---
    project_root: Path = Field(default=PROJECT_ROOT)
    log_dir: Path = Field(default=PROJECT_ROOT / "logs")
    faiss_dir: Path = Field(default=PROJECT_ROOT / "faiss")
    data_dir: Path = Field(default=PROJECT_ROOT / "data")
    faiss_index_path: Path | None = None
    faiss_docs_path: Path | None = None
    policies_dir: Path | None = None
    products_dir: Path | None = None

    # --- Logging ---
    log_level: str = "INFO"

    # --- Database ---
    azure_pg_host: str | None = None
    azure_pg_db: str | None = None
    azure_pg_user: str | None = None
    azure_pg_password: str | None = None
    azure_pg_ssl: str = "require"
    azure_pg_schema: str = "profiles"
    db_connection_string: Optional[PostgresDsn] = None # Made Optional explicitly
    db_max_retries: int = 3
    db_retry_delay: float = 1.0

    # --- Vector Store & Embeddings ---
    hf_api_key: str | None = Field(default=None, validation_alias="HUGGINGFACE_API_KEY")
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    # Construct URL dynamically later if model name changes? For now, keep default.
    embedding_api_url: HttpUrl = Field(default=f"https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2")
    embedding_request_timeout: int = 15
    vector_search_k: int = 3

    # --- LLM / Agent Models ---
    # Use currently available Groq models (as of early 2024/2025 based on logs)
    # Consider Llama3 models on Groq
    news_agent_model: AgentModelConfig = AgentModelConfig(
        id="llama3-8b-8192", temperature=0.4, max_tokens=800 # Llama3 8b is fast
    )
    financial_agent_model: AgentModelConfig = AgentModelConfig(
        id="llama3-8b-8192", temperature=0.1, max_tokens=800 # Llama3 8b is fast
    )
    recommendation_agent_model: AgentModelConfig = AgentModelConfig(
        id="llama3-8b-8192", temperature=0.3, max_tokens=500
    )
    # *** UPDATED COORDINATOR MODEL ***
    coordinator_agent_model: AgentModelConfig = AgentModelConfig(
        # id="llama-3.1-70b-versatile", # Decommissioned!
        id="llama3-70b-8192", # Use current Llama3 70b model on Groq
        temperature=0.2,
        max_tokens=4000 # Adjust based on model limits if needed
    )

    # --- Agent Behavior ---
    max_holdings_in_prompt: int = 10
    max_doc_tokens_in_prompt: int = 1000
    max_financial_summary_len: int = 400
    max_news_summary_len: int = 600
    max_symbols_to_fetch: int = 5
    financial_api_delay: float = 0.2

    # --- Streamlit App ---
    app_title: str = "FinTech AI Advisor"
    app_icon: str = "💹"
    user_avatar: str = "👤"
    assistant_avatar: str = "🤖"

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / '.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )

    def __init__(self, **values):
        super().__init__(**values)
        # Construct derived paths and connection string after loading
        self.faiss_index_path = self.faiss_dir / "faiss_index"
        self.faiss_docs_path = self.faiss_dir / "documents.json"
        self.policies_dir = self.data_dir / "policies"
        self.products_dir = self.data_dir / "products"
        # Construct embedding URL based on model name
        self.embedding_api_url = HttpUrl(f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.embedding_model_name}")


        if self.azure_pg_user and self.azure_pg_password and self.azure_pg_host and self.azure_pg_db:
             # Ensure path starts with / if db name present
             db_path = f"/{self.azure_pg_db}" if self.azure_pg_db else ""
             self.db_connection_string = PostgresDsn(
                 f"postgresql://{self.azure_pg_user}:{self.azure_pg_password}@{self.azure_pg_host}{db_path}"
             )

settings = Settings()

# Add a check after initialization
if not settings.db_connection_string:
    print("WARNING: Database connection string could not be constructed. DB operations will fail.", file=sys.stderr)
if not settings.hf_api_key:
    print("WARNING: Hugging Face API key not found. Embedding generation will fail.", file=sys.stderr)