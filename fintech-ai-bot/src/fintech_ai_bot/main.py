# src/fintech_ai_bot/main.py
# REVERTED TO STANDARD STREAMLIT UI
from typing import Optional
import streamlit as st
import traceback
import sys
from datetime import datetime

# Import configuration and utility functions first
from fintech_ai_bot.config import settings
from fintech_ai_bot.utils import get_logger, generate_error_html # Keep generate_error_html for potential non-UI errors

# Import core components and UI modules
from fintech_ai_bot.db.postgres_client import PostgresClient
from fintech_ai_bot.vector_store.faiss_client import FAISSClient
from fintech_ai_bot.core.orchestrator import AgentOrchestrator
from fintech_ai_bot.ui import sidebar, chat_interface # Import UI modules

# Initialize root logger for main application scope
logger = get_logger("StreamlitApp")

# --- UI Setup ---
def setup_ui():
    """Configure Streamlit page settings using standard defaults."""
    try:
        st.set_page_config(
            page_title=settings.app_title,
            layout="wide", # You might change this to "centered" for the absolute default
            page_icon=settings.app_icon, # Use icon from settings
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'mailto:support@example.com',
                'Report a bug': "mailto:support@example.com",
                'About': f"### {settings.app_title}\nYour AI-powered financial guidance."
            }
        )

        # --- Custom CSS Block Removed ---
        # The large st.markdown(<style>...) block has been removed
        # to restore the default Streamlit look and feel.
        # Streamlit's built-in light and dark themes will now apply.

    except Exception as e:
        logger.error(f"UI setup failed: {e}", exc_info=True)
        # Use st.error directly for simpler error display without custom HTML
        st.error(f"Failed to initialize application UI settings: {e}")

# --- Resource Initialization (Cached) ---
# (get_db_client, get_vector_store_client, get_agent_orchestrator functions remain the same)
@st.cache_resource(show_spinner="Connecting to Database...")
def get_db_client() -> PostgresClient:
    logger.info("Initializing PostgresClient...")
    try:
        client = PostgresClient()
        return client
    except Exception as e:
        logger.critical(f"CRITICAL: Failed to initialize PostgresClient: {e}", exc_info=True)
        # Use standard st.error
        st.error(f"🚨 Database Connection Failed! Could not connect. Error: {e}")
        st.stop()

@st.cache_resource(show_spinner="Loading Knowledge Base...")
def get_vector_store_client() -> FAISSClient:
    logger.info("Initializing FAISSClient...")
    try:
        client = FAISSClient()
        if client.index is None or client.index.ntotal == 0:
             logger.warning("FAISS index seems empty or failed to load properly.")
        return client
    except Exception as e:
        logger.critical(f"CRITICAL: Failed to initialize FAISSClient: {e}", exc_info=True)
         # Use standard st.error
        st.error(f"🚨 Knowledge Base Failed! Could not load vector store. Contextual answers might be impaired. Error: {e}")
        return None # Allow app to potentially continue without vector store

@st.cache_resource(show_spinner="Initializing AI Advisor...")
def get_agent_orchestrator(_db_client: PostgresClient, _vector_store_client: Optional[FAISSClient]) -> AgentOrchestrator:
    logger.info("Initializing AgentOrchestrator...")
    try:
        orchestrator = AgentOrchestrator(_db_client, _vector_store_client)
        return orchestrator
    except Exception as e:
        logger.critical(f"CRITICAL: Failed to initialize AgentOrchestrator: {e}", exc_info=True)
         # Use standard st.error
        st.error(f"🚨 AI System Failed! Core AI components could not be initialized. Error: {e}")
        st.stop()

# --- Main Application ---
def main():
    """Main function to orchestrate the Streamlit application."""
    setup_ui()

    st.title(settings.app_title)
    # Use st.caption or st.write for simpler subtitle without custom HTML
    st.caption("Your AI partner for financial insights and portfolio analysis.")

    orchestrator = None
    db_client = None
    vector_store_failed = False
    try:
        db_client = get_db_client()
        vector_store_client = get_vector_store_client()
        if vector_store_client is None:
            vector_store_failed = True # Flag that VS failed but continue

        # Pass clients to the orchestrator initializer
        orchestrator = get_agent_orchestrator(db_client, vector_store_client)

        if vector_store_failed:
             # Use standard st.warning
             st.warning("⚠️ Knowledge base failed to load. Contextual answers from documents may be unavailable.")

    except Exception as init_error:
         logger.critical(f"Failed during component initialization sequence: {init_error}", exc_info=True)
         # Error should have been displayed by cached functions if they failed and called st.stop()
         if not getattr(st, 'errors_displayed', False):
             # Use standard st.error
             st.error(f"💥 Application Initialization Error! A core component failed to start. Error: {init_error}")
             st.errors_displayed = True
         st.stop()

    # --- Sidebar ---
    if db_client:
        # Assuming sidebar module uses standard streamlit widgets
        sidebar.manage_sidebar(db_client)
    else:
        st.sidebar.error("🚨 Database connection unavailable.")

    # --- Main Chat Area ---
    if orchestrator and db_client:
        # Assuming chat_interface module uses standard st.chat_message, st.chat_input etc.
        chat_interface.display_chat_messages()
        chat_interface.handle_chat_input(orchestrator, db_client)
    else:
        # Use standard st.divider instead of custom HTML hr
        st.divider()
        st.warning("⚠️ AI Advisor features are unavailable due to an initialization error.")


# --- Application Entry Point & Final Exception Handler ---
# (Final exception handler remains largely the same, but use st.error directly)
if __name__ == "__main__":
    st.errors_displayed = False # Initialize flag
    try:
        main()
    except SystemExit:
        logger.info("Application stopped via st.stop() or SystemExit.")
        pass
    except Exception as e:
        critical_logger = get_logger("MainCriticalError")
        error_type = type(e).__name__; error_message = str(e); full_traceback = traceback.format_exc()
        critical_logger.critical(f"Unhandled error: {error_type} - {error_message}", exc_info=True)
        critical_logger.critical(full_traceback)
        try:
            if not getattr(st, 'errors_displayed', False):
                # Use standard st.error
                st.error(f"💥 Critical Application Error! Error: {error_type}. Check logs.")
                st.errors_displayed = True
        except Exception as st_err:
            critical_logger.critical(f"!!! FAILED to display critical error via st.error: {st_err}", exc_info=True)
            # Fallback to print for absolutely critical errors
            print(f"\n--- CRITICAL UNHANDLED ERROR ---", file=sys.stderr)
            print(f"Timestamp: {datetime.now()}", file=sys.stderr)
            print(f"Original Error: {error_type} - {error_message}\nTraceback:\n{full_traceback}", file=sys.stderr)
            print(f"Error during st.error display: {type(st_err).__name__} - {st_err}", file=sys.stderr)
            print(f"--- END CRITICAL ERROR ---\n", file=sys.stderr)