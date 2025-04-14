# Importing necessary libraries
import sys
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, PostgresDsn, HttpUrl
from dotenv import load_dotenv

# Load .env file from the root of the 'fintech-ai-bot' directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
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

    # Vector database settings
    hf_api_key: str | None = Field(default=None, validation_alias="HUGGINGFACE_API_KEY")
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # Construct URL dynamically later in __init__
    embedding_api_url: Optional[HttpUrl] = None
    embedding_request_timeout: int = 15
    vector_search_k: int = 3

    # Agent Models
    # Configuration for different agent models
    news_agent_model: AgentModelConfig = AgentModelConfig(
        id="llama3-8b-8192", temperature=0.4, max_tokens=800
    )
    financial_agent_model: AgentModelConfig = AgentModelConfig(
        id="llama3-8b-8192", temperature=0.1, max_tokens=800
    )
    recommendation_agent_model: AgentModelConfig = AgentModelConfig(
        id="llama3-8b-8192", temperature=0.3, max_tokens=800
    )
    coordinator_agent_model: AgentModelConfig = AgentModelConfig(
        id="llama3-70b-8192",
        temperature=0.2,
        max_tokens=4000
    )

    # Agent settings
    max_holdings_in_prompt: int = 10
    max_doc_tokens_in_prompt: int = 5000
    max_financial_summary_len: int = 500
    max_news_summary_len: int = 600
    max_symbols_to_fetch: int = 10
    financial_api_delay: float = 0.2

    # Streamlit settings
    app_title: str = "FinTech AI Advisor"
    app_icon: str = "💹"
    app_description: str = "Your AI partner for financial insights and portfolio analysis."
    user_avatar: str = "🧔🏻‍♂️"
    assistant_avatar: str = "🪼"
    cache_ttl_seconds: int = 300

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / '.env',
        env_file_encoding='utf-8',
        extra='ignore',
        validate_assignment=True
    )

    # Using __init__ and post-init logic for derived fields
    def __init__(self, **values):
        super().__init__(**values)
        # Construct derived paths after initial loading
        self.faiss_index_path = self.faiss_dir / "faiss_index"
        self.faiss_docs_path = self.faiss_dir / "documents.json"
        self.policies_dir = self.data_dir / "policies"
        self.products_dir = self.data_dir / "products"

        # Construct embedding URL based on model name
        self.embedding_api_url = HttpUrl(f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.embedding_model_name}")

        # Construct DB connection string if Azure details are present
        if self.azure_pg_user and self.azure_pg_password and self.azure_pg_host:
            # Ensure path starts with / if db name present, otherwise empty
            db_path = f"/{self.azure_pg_db}" if self.azure_pg_db else ""
            # Pydantic v2 requires URL scheme for PostgresDsn
            conn_str = f"postgresql://{self.azure_pg_user}:{self.azure_pg_password}@{self.azure_pg_host}:5432{db_path}"
            try:
                # Validate and assign using Pydantic's validation
                self.db_connection_string = PostgresDsn(conn_str)
            except Exception as e:
                 # Handle potential validation errors during construction
                 print(f"WARNING: Error constructing/validating DB connection string: {e}", file=sys.stderr)
                 self.db_connection_string = None # Ensure it's None if construction fails
        else:
            self.db_connection_string = None # Ensure it's None if parts are missing

# Instantiate settings globally
settings = Settings()

# Add checks after initialization
# These checks run when the module is imported
if not settings.db_connection_string:
    print("WARNING: Database connection string could not be constructed or is missing. DB operations will fail.", file=sys.stderr)

if not settings.hf_api_key:
    print("WARNING: Hugging Face API key (HUGGINGFACE_API_KEY) not found in environment variables or .env file. Embedding generation will fail.", file=sys.stderr)

# Create log directory if it doesn't exist
if not settings.log_dir.exists():
    try:
        settings.log_dir.mkdir(parents=True, exist_ok=True)
        print(f"Log directory created at: {settings.log_dir}")
    except Exception as e:
        print(f"ERROR: Could not create log directory at {settings.log_dir}: {e}", file=sys.stderr)