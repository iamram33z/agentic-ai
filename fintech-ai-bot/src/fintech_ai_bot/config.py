# config.py

# Importing necessary libraries
import sys
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
# HttpUrl might not be needed anymore if not used elsewhere
# from pydantic import Field, PostgresDsn, HttpUrl
from pydantic import Field, PostgresDsn
from dotenv import load_dotenv

# Load .env file from the root of the 'fintech-ai-bot' directory
# Use resolve() for robustness, ensure correct parent levels based on file location
# Assuming config.py is in src/fintech_ai_bot/config.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

class AgentModelConfig(BaseSettings):
    id: str = "llama3-8b-8192"
    temperature: float = 0.5
    max_tokens: int = 1000

class Settings(BaseSettings):
    # Project directories
    project_root: Path = Field(default=PROJECT_ROOT)
    log_dir: Path = Field(default=PROJECT_ROOT / "logs")
    faiss_dir: Path = Field(default=PROJECT_ROOT / "faiss")
    data_dir: Path = Field(default=PROJECT_ROOT / "data")
    # Derived paths will be set in __init__
    faiss_index_path: Path | None = None
    faiss_docs_path: Path | None = None
    policies_dir: Path | None = None
    products_dir: Path | None = None

    # Logging
    log_level: str = "INFO"

    # Database settings
    azure_pg_host: str | None = None
    azure_pg_db: str | None = None
    azure_pg_user: str | None = None
    azure_pg_password: str | None = None
    azure_pg_ssl: str = "require"
    azure_pg_schema: str = "profiles"
    db_connection_string: Optional[PostgresDsn] = None
    db_max_retries: int = 3
    db_retry_delay: float = 1.0

    # Vector database settings (using local pipeline)
    hf_api_key: str | None = Field(default=None,
                                   validation_alias="HUGGINGFACE_API_KEY")  # Still useful for authenticated downloads
    embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    embedding_dimension: int = 768
    # embedding_api_url: Optional[HttpUrl] = None # Removed: No longer needed for local pipeline
    # embedding_request_timeout: int = 15      # Removed: No longer needed for local pipeline
    vector_search_k: int = 3                   # Number of relevant docs to retrieve

    # Agent Models
    # Configuration for different agent models
    news_agent_model: AgentModelConfig = AgentModelConfig(
        id="llama3-8b-8192", temperature=0.4, max_tokens=1000
    )
    financial_agent_model: AgentModelConfig = AgentModelConfig(
        id="llama3-8b-8192", temperature=0.1, max_tokens=1000
    )
    coordinator_agent_model: AgentModelConfig = AgentModelConfig(
        id="llama3-8b-8192",
        temperature=0.2,
        max_tokens=4000,
    )

    # Agent settings
    max_holdings_in_prompt: int = 10
    # Max approx characters from retrieved docs to include in agent context (adjust multiplier as needed)
    max_doc_chars_in_prompt: int = 5000 * 5
    max_financial_summary_len: int = 500  # Target token length for financial agent summary
    max_news_summary_len: int = 600       # Target token length for news agent summary
    max_symbols_to_fetch: int = 10        # Max stock symbols to fetch news/data for at once
    financial_api_delay: float = 0.2      # Delay between financial API calls (e.g., yfinance)

    # Streamlit settings
    app_title: str = "FinTech AI Advisor"
    app_icon: str = "💹"
    app_description: str = "Your AI partner for financial insights and portfolio analysis."
    user_avatar: str = "🧔🏻‍♂️"
    assistant_avatar: str = "🪼"
    cache_ttl_seconds: int = 300          # Cache time for Streamlit functions

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / '.env',
        env_file_encoding='utf-8',
        extra='ignore',                   # Ignore extra fields from .env
        validate_assignment=True          # Validate fields on assignment
    )

    # Using __init__ and post-init logic for derived fields
    def __init__(self, **values):
        super().__init__(**values)
        # Construct derived paths after initial loading
        # Ensure base directories are Path objects before joining
        self.faiss_dir = Path(self.faiss_dir)
        self.data_dir = Path(self.data_dir)
        self.log_dir = Path(self.log_dir)

        self.faiss_index_path = self.faiss_dir / "faiss_index"
        self.faiss_docs_path = self.faiss_dir / "documents.json"
        self.policies_dir = self.data_dir / "policies"
        self.products_dir = self.data_dir / "products"

        # Construct DB connection string if Azure details are present
        if self.azure_pg_user and self.azure_pg_password and self.azure_pg_host:
            db_path = f"/{self.azure_pg_db}" if self.azure_pg_db else ""
            conn_str = f"postgresql://{self.azure_pg_user}:{self.azure_pg_password}@{self.azure_pg_host}:5432{db_path}?sslmode={self.azure_pg_ssl}"
            try:
                self.db_connection_string = PostgresDsn(conn_str)
            except Exception as e:
                 print(f"WARNING: Error constructing/validating DB connection string: {e}", file=sys.stderr)
                 self.db_connection_string = None
        else:
            if self.azure_pg_host or self.azure_pg_user or self.azure_pg_db:
                 print("WARNING: Partial Azure PostgreSQL connection details provided. Full details (host, user, password) are required.", file=sys.stderr)
            self.db_connection_string = None

# Instantiate settings globally
settings = Settings()

# Add checks after initialization
# These checks run when the module is imported
if not settings.db_connection_string:
    print("WARNING: Database connection string could not be constructed or is missing/incomplete. DB operations will fail.", file=sys.stderr)

if not settings.hf_api_key:
    print("WARNING: Hugging Face API key (HUGGINGFACE_API_KEY) not found. Model download might fail if the model isn't cached locally or if it's a private/gated model.", file=sys.stderr)

# Create log directory if it doesn't exist (using the path derived in __init__)
# Check log_dir directly from the settings instance
if not settings.log_dir.exists():
    try:
        settings.log_dir.mkdir(parents=True, exist_ok=True)
        print(f"Log directory created at: {settings.log_dir}")
    except Exception as e:
        print(f"ERROR: Could not create log directory at {settings.log_dir}: {e}", file=sys.stderr)
elif not settings.log_dir.is_dir():
     print(f"ERROR: Path exists but is not a directory: {settings.log_dir}", file=sys.stderr)