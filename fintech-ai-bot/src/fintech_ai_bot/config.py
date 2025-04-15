# config.py

import sys
import os # Added for robust key loading message
from pathlib import Path
from typing import Optional, Union
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, NonNegativeInt, PostgresDsn, DirectoryPath, FilePath, validator, PositiveInt, \
    NonNegativeFloat
from dotenv import load_dotenv
import logging # Added for early logging

# --- Early Logger Setup (for config loading issues) ---
# Basic config before settings are loaded
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
config_logger = logging.getLogger(__name__)

# --- Project Root Setup ---
try:
    # Assuming config.py is in src/fintech_ai_bot/config.py
    # Adjust parent calls if your structure is different
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
except NameError:
    # Fallback for environments where __file__ might not be defined (e.g., some notebooks)
    PROJECT_ROOT = Path(".").resolve()
    config_logger.warning(f"__file__ not defined, setting PROJECT_ROOT to current working directory: {PROJECT_ROOT}")

# --- Load .env ---
dotenv_path = PROJECT_ROOT / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)
    config_logger.info(f"Loaded environment variables from: {dotenv_path}")
else:
    config_logger.warning(f".env file not found at expected location: {dotenv_path}")

# --- Model Definitions ---
class AgentModelConfig(BaseSettings):
    """Configuration for LLM agent models."""
    id: str = "llama3-8b-8192"  # Default model ID
    temperature: NonNegativeFloat = 0.5
    max_tokens: PositiveInt = 1000

# --- Main Settings Class ---
class Settings(BaseSettings):
    """Main application settings, loaded from environment variables and .env file."""

    # --- Core Paths ---
    project_root: DirectoryPath = Field(default=PROJECT_ROOT)
    log_dir: Path = Field(default=PROJECT_ROOT / "logs") # Path object, validated in __init__
    faiss_dir: Path = Field(default=PROJECT_ROOT / "faiss") # Path object, validated in __init__
    data_dir: Path = Field(default=PROJECT_ROOT / "data")   # Path object, validated in __init__

    # Derived paths (set in __init__)
    faiss_index_path: Optional[Path] = None
    faiss_docs_path: Optional[Path] = None
    policies_dir: Optional[Path] = None
    products_dir: Optional[Path] = None

    # --- Logging ---
    log_level: str = "INFO" # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL

    # --- Database (PostgreSQL) ---
    azure_pg_host: Optional[str] = None
    azure_pg_db: Optional[str] = None
    azure_pg_user: Optional[str] = None
    azure_pg_password: Optional[str] = Field(default=None, repr=False) # Prevent logging password
    azure_pg_ssl: str = "require" # Default SSL mode for Azure PG
    azure_pg_schema: str = "profiles" # Default DB schema
    db_connection_string: Optional[PostgresDsn] = None # Set in __init__
    db_max_retries: PositiveInt = 3
    db_retry_delay: NonNegativeFloat = 1.0 # Seconds

    # --- Vector Store (FAISS + Local Embeddings) ---
    hf_api_key: Optional[str] = Field(default=None, validation_alias="HUGGINGFACE_API_KEY", repr=False) # For model download auth
    embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2" # HF model ID
    embedding_dimension: PositiveInt = 768 # Must match model output dim (all-mpnet-base-v2 is 768)
    vector_search_k: PositiveInt = 3 # Number of relevant chunks to retrieve

    # NEW: Token-based chunking settings
    chunk_size_tokens: PositiveInt = 450 # Target chunk size in tokens (must be <= model max length)
    chunk_overlap_tokens: NonNegativeInt = 50 # Overlap between chunks in tokens
    embedding_batch_size: PositiveInt = 32 # Number of chunks to embed in one go

    # --- Agent Models ---
    news_agent_model: AgentModelConfig = AgentModelConfig(id="llama3-8b-8192", temperature=0.4, max_tokens=1000)
    financial_agent_model: AgentModelConfig = AgentModelConfig(id="llama3-8b-8192", temperature=0.1, max_tokens=1000)
    coordinator_agent_model: AgentModelConfig = AgentModelConfig(id="llama3-8b-8192", temperature=0.2, max_tokens=4000)

    # --- Agent Logic ---
    max_holdings_in_prompt: PositiveInt = 10 # Limit number of holdings shown to LLM
    # Max *characters* from retrieved doc chunks to include in agent context
    # Estimate based on avg token length if needed, e.g., 5000 tokens * ~5 chars/token
    max_doc_chars_in_prompt: PositiveInt = 25000
    max_financial_summary_len: PositiveInt = 500 # Target max tokens for financial summary
    max_news_summary_len: PositiveInt = 600 # Target max tokens for news summary
    max_symbols_to_fetch: PositiveInt = 10 # Limit for parallel external API calls (e.g., yfinance)
    financial_api_delay: NonNegativeFloat = 0.2 # Seconds delay between API calls

    # --- Streamlit UI ---
    app_title: str = "FinTech AI Advisor"
    app_icon: str = "💹"
    app_description: str = "Your AI partner for financial insights and portfolio analysis."
    user_avatar: str = "🧔🏻‍♂️"
    assistant_avatar: str = "🪼"
    cache_ttl_seconds: PositiveInt = 300 # Cache duration for Streamlit functions

    # --- Pydantic Model Config ---
    model_config = SettingsConfigDict(
        env_file=dotenv_path if dotenv_path.exists() else None, # Load .env if found
        env_file_encoding='utf-8',
        extra='ignore', # Ignore extra environment variables
        validate_assignment=True # Validate on attribute assignment
    )

    # --- Initialization and Validation ---
    def __init__(self, **values):
        super().__init__(**values)
        # --- Resolve and Validate Paths ---
        # Ensure base directories are Path objects before joining
        self.log_dir = Path(self.log_dir).resolve()
        self.faiss_dir = Path(self.faiss_dir).resolve()
        self.data_dir = Path(self.data_dir).resolve()

        # Create derived paths
        self.faiss_index_path = self.faiss_dir / "faiss_index"
        self.faiss_docs_path = self.faiss_dir / "documents.json"
        self.policies_dir = self.data_dir / "policies"
        self.products_dir = self.data_dir / "products"

        # --- Construct DB Connection String ---
        self._construct_db_connection_string()

        # --- Initial Log Messages ---
        # Log key paths being used
        config_logger.info(f"Project Root: {self.project_root}")
        config_logger.info(f"Log Directory: {self.log_dir}")
        config_logger.info(f"FAISS Directory: {self.faiss_dir}")
        config_logger.info(f"Data Directory: {self.data_dir}")
        # Log DB connection status (avoid logging the string itself)
        config_logger.info(f"DB Connection String Set: {'Yes' if self.db_connection_string else 'No'}")
        # Log HF API Key status without exposing the key
        hf_key_status = "Not Set"
        if self.hf_api_key: hf_key_status = "Set (Loaded)"
        elif os.getenv("HUGGINGFACE_API_KEY"): hf_key_status = "Set (Env Var)"
        config_logger.info(f"Hugging Face API Key Status: {hf_key_status}")

    def _construct_db_connection_string(self):
        """Helper to construct the DB connection string."""
        if self.azure_pg_user and self.azure_pg_password and self.azure_pg_host:
            db_path = f"/{self.azure_pg_db}" if self.azure_pg_db else ""
            conn_str = f"postgresql://{self.azure_pg_user}:{self.azure_pg_password}@{self.azure_pg_host}:5432{db_path}?sslmode={self.azure_pg_ssl}"
            try:
                self.db_connection_string = PostgresDsn(conn_str)
            except Exception as e:
                 config_logger.warning(f"Error constructing/validating DB connection string: {e}")
                 self.db_connection_string = None
        else:
            # Warn only if some Azure details were provided but not all required ones
            if self.azure_pg_host or self.azure_pg_user or self.azure_pg_db or self.azure_pg_password:
                 config_logger.warning("Partial Azure PG details provided. Host, User, Password required.")
            self.db_connection_string = None

# --- Instantiate Settings Globally ---
try:
    settings = Settings()
    config_logger.info("Settings loaded successfully.")
except Exception as e:
    config_logger.critical(f"Failed to initialize Settings: {e}", exc_info=True)
    # Provide minimal fallback settings if critical failure during init
    settings = Settings( # Use default values which might allow parts of app to run
        azure_pg_host=None, azure_pg_user=None, azure_pg_password=None, azure_pg_db=None,
        hf_api_key=None
    )
    config_logger.warning("Initialized with minimal fallback settings.")

# --- Post-Initialization Checks ---
# These run when the config module is imported elsewhere

# Log directory creation check (moved here from original location)
if not settings.log_dir.exists():
    try:
        settings.log_dir.mkdir(parents=True, exist_ok=True)
        config_logger.info(f"Log directory created at: {settings.log_dir}")
    except Exception as e:
        config_logger.error(f"Could not create log directory at {settings.log_dir}: {e}", exc_info=True)
elif not settings.log_dir.is_dir():
     config_logger.error(f"Log path exists but is not a directory: {settings.log_dir}")

# Database connection warning
if not settings.db_connection_string:
    config_logger.warning("DB connection string not configured. Database operations will fail.")

# Hugging Face API key warning
if not settings.hf_api_key and not os.getenv("HUGGINGFACE_API_KEY"):
    config_logger.warning("Hugging Face API key (HUGGINGFACE_API_KEY) not found in env or .env. Model download might fail if private/gated or not cached.")

# Embedding dimension check (example)
if settings.embedding_model_name == "sentence-transformers/all-mpnet-base-v2" and settings.embedding_dimension != 768:
     config_logger.warning(f"Configured embedding dimension {settings.embedding_dimension} does not match standard 768 for {settings.embedding_model_name}.")
elif settings.embedding_model_name == "sentence-transformers/all-MiniLM-L6-v2" and settings.embedding_dimension != 384:
     config_logger.warning(f"Configured embedding dimension {settings.embedding_dimension} does not match standard 384 for {settings.embedding_model_name}.")
# Add more checks as needed

config_logger.debug("Config module loaded.")