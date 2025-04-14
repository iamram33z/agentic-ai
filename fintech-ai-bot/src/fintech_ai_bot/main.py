# src/fintech_ai_bot/main.py
# ENHANCED UI/UX using standard components

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
            layout="wide", # Keep wide layout for more space
            page_icon=settings.app_icon,
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'mailto:support@example.com', # Placeholder
                'Report a bug': "mailto:support@example.com", # Placeholder
                'About': f"### {settings.app_title}\n{settings.app_description}" # Use description from settings
            }
        )
        # No custom CSS block - uses Streamlit's default light/dark themes

    except Exception as e:
        logger.error(f"UI setup failed: {e}", exc_info=True)
        st.error(f"Failed to initialize application UI settings: {e}") # Use standard error display

# --- Resource Initialization (Cached) ---
# (Cached functions remain the same - error handling uses st.error/st.stop)
@st.cache_resource(show_spinner="Connecting to Database...")
def get_db_client() -> PostgresClient:
    logger.info("Initializing PostgresClient...")
    try:
        client = PostgresClient()
        return client
    except Exception as e:
        logger.critical(f"CRITICAL: Failed to initialize PostgresClient: {e}", exc_info=True)
        st.error(f"🚨 Database Connection Failed! Could not connect. Please check configuration or contact support. Error: {e}")
        st.stop() # Stop execution if DB fails

@st.cache_resource(show_spinner="Loading Knowledge Base...")
def get_vector_store_client() -> Optional[FAISSClient]: # Allow return None
    logger.info("Initializing FAISSClient...")
    try:
        client = FAISSClient()
        if client.index is None or client.index.ntotal == 0:
             logger.warning("FAISS index seems empty or failed to load properly.")
             # Don't raise critical error here, allow app to continue
        return client
    except Exception as e:
        # Log as error, but don't stop the app if only vector store fails
        logger.error(f"Failed to initialize FAISSClient: {e}", exc_info=True)
        st.toast("⚠️ Knowledge Base failed to load. Contextual answers may be limited.", icon="📉")
        return None # Allow app to potentially continue without vector store

@st.cache_resource(show_spinner="Initializing AI Advisor...")
def get_agent_orchestrator(_db_client: PostgresClient, _vector_store_client: Optional[FAISSClient]) -> AgentOrchestrator:
    logger.info("Initializing AgentOrchestrator...")
    try:
        orchestrator = AgentOrchestrator(_db_client, _vector_store_client)
        return orchestrator
    except Exception as e:
        logger.critical(f"CRITICAL: Failed to initialize AgentOrchestrator: {e}", exc_info=True)
        st.error(f"🚨 AI System Failed! Core AI components could not be initialized. Please contact support. Error: {e}")
        st.stop() # Stop execution if core orchestrator fails

# --- Main Application ---
def main():
    """Main function to orchestrate the Streamlit application."""
    setup_ui()

    st.title(f"{settings.app_icon} {settings.app_title}")
    st.header("Your AI partner for financial insights")
    st.caption(settings.app_description) # Use description from settings

    orchestrator = None
    db_client = None
    vector_store_client = None # Keep track of it
    init_error_occurred = False

    try:
        # Initialize components sequentially
        db_client = get_db_client()
        vector_store_client = get_vector_store_client() # Might be None
        orchestrator = get_agent_orchestrator(db_client, vector_store_client)

        if vector_store_client is None:
             # Display a persistent warning if VS failed but app continues
             st.warning("⚠️ Knowledge base (Vector Store) could not be loaded. Answers based on uploaded documents might be unavailable.", icon="📉")

    except Exception as init_error:
         # This catches errors from st.stop() or subsequent issues if st.stop() failed
         logger.critical(f"Unhandled exception during component initialization sequence: {init_error}", exc_info=True)
         if not getattr(st, 'stop_called', False): # Avoid double error if st.stop worked
             st.error(f"💥 Application Initialization Error! A critical component failed to start. Please refresh or contact support. Error: {init_error}")
         init_error_occurred = True
         # st.stop() should have been called by the failing cached function

    st.divider() # Visual separator before main content areas

    # --- Sidebar ---
    if db_client and not init_error_occurred:
        # Pass the DB client to the sidebar management function
        sidebar.manage_sidebar(db_client)
    elif not db_client:
        st.sidebar.error("🚨 Database connection is unavailable. Cannot load client data.")
    else:
        st.sidebar.warning("Application initialization failed. Sidebar features disabled.")

    # --- Main Chat Area ---
    if orchestrator and db_client and not init_error_occurred:
        # Pass orchestrator and db_client to the chat interface functions
        chat_interface.display_chat_messages()
        chat_interface.handle_chat_input(orchestrator, db_client)
    else:
        # Display message indicating core functionality is unavailable
        st.warning("⚠️ AI Advisor features are unavailable due to an initialization error. Please check the sidebar or refresh the application.", icon="🤖")


# --- Application Entry Point & Final Exception Handler ---
if __name__ == "__main__":
    st.stop_called = False # Flag to track if st.stop was called
    try:
        main()
    except SystemExit:
        # This catches st.stop() calls gracefully
        st.stop_called = True
        logger.info("Application stopped via st.stop() or SystemExit.")
        pass # Normal exit/stop
    except Exception as e:
        # Log critical errors that weren't caught or handled by st.stop()
        critical_logger = get_logger("MainCriticalError")
        error_type = type(e).__name__; error_message = str(e); full_traceback = traceback.format_exc()
        critical_logger.critical(f"Unhandled error at main level: {error_type} - {error_message}", exc_info=True)
        critical_logger.critical(full_traceback)

        # Attempt to display a final error message in the Streamlit app if possible
        try:
            if not st.stop_called: # Only show if app didn't stop normally
                st.error(f"💥 Critical Application Error! An unexpected issue occurred. Please check the logs or contact support. Error: {error_type}.")
        except Exception as st_err:
            critical_logger.critical(f"!!! FAILED to display critical error via st.error in final handler: {st_err}", exc_info=True)
            # Fallback to stderr printing if Streamlit display fails
            print(f"\n--- CRITICAL UNHANDLED ERROR (FALLBACK) ---", file=sys.stderr)
            print(f"Timestamp: {datetime.now()}", file=sys.stderr)
            print(f"Original Error: {error_type} - {error_message}\nTraceback:\n{full_traceback}", file=sys.stderr)
            print(f"Error during st.error display: {type(st_err).__name__} - {st_err}", file=sys.stderr)
            print(f"--- END CRITICAL ERROR ---\n", file=sys.stderr)