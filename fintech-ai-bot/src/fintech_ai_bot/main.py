# src/fintech_ai_bot/main.py
# CORRECTED

import streamlit as st
import traceback
import sys
from datetime import datetime

# Import configuration and utility functions first
from fintech_ai_bot.config import settings
from fintech_ai_bot.utils import get_logger, generate_error_html

# Import core components and UI modules
from fintech_ai_bot.db.postgres_client import PostgresClient
from fintech_ai_bot.vector_store.faiss_client import FAISSClient
from fintech_ai_bot.core.orchestrator import AgentOrchestrator
from fintech_ai_bot.ui import sidebar, chat_interface # Import UI modules

# Initialize root logger for main application scope
logger = get_logger("StreamlitApp")

# --- UI Setup (Inject CSS) ---
def setup_ui():
    """Configure Streamlit page settings and inject custom CSS."""
    try:
        st.set_page_config(
            page_title=settings.app_title,
            layout="wide",
            page_icon=settings.app_icon, # Use icon from settings
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'mailto:support@example.com',
                'Report a bug': "mailto:support@example.com",
                'About': f"### {settings.app_title}\nYour AI-powered financial guidance."
            }
        )

        # --- Modern Dark Theme CSS ---
        # (Paste the full CSS string here as before)
        st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        /* --- Base & General Styling --- */
        body, .stApp, input, textarea, button, select, .stMarkdown {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }}

        :root {{
            --color-primary: #6366f1;
            --color-secondary: #8b5cf6;
            --color-accent: #ec4899;
            --color-success: #10b981;
            --color-warning: #f59e0b;
            --color-danger: #ef4444;
            --color-info: #3b82f6;
            --color-background: #0f172a;
            --color-surface: #1e293b;
            --color-surface-light: #334155;
            --color-border: #475569;
            --color-border-light: #64748b;
            --color-text-primary: #f8fafc;
            --color-text-secondary: #e2e8f0;
            --color-text-muted: #94a3b8;
            --color-text-dark: #1e293b;
            --gradient-primary: linear-gradient(135deg, var(--color-primary) 0%, var(--color-secondary) 50%, var(--color-accent) 100%);
            --gradient-secondary: linear-gradient(135deg, var(--color-surface) 0%, var(--color-background) 100%);
        }}

        .stApp {{
            background-color: var(--color-background);
            color: var(--color-text-secondary);
        }}

        /* Main content area */
        .main .block-container,
        section[data-testid="st.main"] > div:first-child {{
            background-color: var(--color-surface);
            padding: 2.5rem 3rem 4rem 3rem;
            border-radius: 16px;
            margin-top: 1.5rem;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.25);
            border: 1px solid var(--color-border);
        }}

        /* Typography */
        h1 {{
            color: var(--color-text-primary); font-weight: 700;
            background: var(--gradient-primary); -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            padding-bottom: 0.6em; margin-bottom: 0.6em; font-size: 2.2rem; position: relative;
        }}
        h1::after {{ content: ''; position: absolute; bottom: 0; left: 0; width: 100%; height: 2px; background: var(--gradient-primary); }}
        h1 + p {{ color: var(--color-text-secondary); font-size: 1.1rem; font-weight: 400; margin-top: -1.5rem; margin-bottom: 3rem; max-width: 65ch; }}
        h2 {{ color: var(--color-text-primary); font-weight: 600; margin-top: 2.5em; margin-bottom: 1.2em; padding-bottom: 0.5em; font-size: 1.6rem; position: relative; }}
        h2::after {{ content: ''; position: absolute; bottom: 0; left: 0; width: 60px; height: 3px; background: var(--gradient-primary); border-radius: 3px; }}
        h3 {{ color: var(--color-text-primary); font-weight: 600; margin-top: 2em; margin-bottom: 1em; font-size: 1.3rem; }}
        p, .stMarkdown p, .stText {{ color: var(--color-text-secondary); line-height: 1.7; font-size: 1rem; }}
        a {{ color: var(--color-primary); text-decoration: none; font-weight: 500; transition: all 0.2s ease; }}
        a:hover {{ color: var(--color-accent); text-decoration: underline; }}

        /* Sidebar */
        .stSidebar > div:first-child {{ background: var(--gradient-secondary); border-right: 1px solid var(--color-border); padding: 1.8rem 1.5rem; }}
        .stSidebar h2 {{ color: var(--color-text-primary); font-size: 1.2rem; font-weight: 600; margin-bottom: 1.5rem; padding-bottom: 0.8rem; position: relative; }}
        .stSidebar h2::after {{ content: ''; position: absolute; bottom: 0; left: 0; width: 40px; height: 2px; background: var(--gradient-primary); border-radius: 2px; }}
        .stSidebar .stTextInput label {{ color: var(--color-text-secondary); font-size: 0.85rem; font-weight: 500; margin-bottom: 0.5rem; display: block; }}
        .stSidebar .stTextInput input {{ border-radius: 10px; border: 1px solid var(--color-border); background-color: rgba(15, 23, 42, 0.7); color: var(--color-text-primary); padding: 0.8rem 1rem; font-size: 0.95rem; width: 100%; transition: all 0.2s ease; }}
        .stSidebar .stTextInput input:focus {{ border-color: var(--color-primary); box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.3); outline: none; background-color: rgba(15, 23, 42, 0.9); }}
        .stSidebar .stButton button {{ background: var(--gradient-primary); color: white; border: none; border-radius: 10px; padding: 0.8rem 1.2rem; width: 100%; font-weight: 600; font-size: 0.95rem; transition: all 0.2s ease; cursor: pointer; box-shadow: 0 2px 10px rgba(99, 102, 241, 0.3); margin-top: 1rem; }}
        .stSidebar .stButton button:hover {{ transform: translateY(-1px); box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4); }}
        .stSidebar .stButton button:active {{ transform: translateY(0); box-shadow: 0 2px 5px rgba(99, 102, 241, 0.3); }}

        /* Portfolio Summary */
        .portfolio-summary {{ padding: 1.5rem 1.2rem; margin-bottom: 1.5rem; border-radius: 12px; background: rgba(15, 23, 42, 0.5); border: 1px solid var(--color-border); backdrop-filter: blur(5px); }}
        .stSidebar h5 {{ color: var(--color-text-primary); font-size: 1.1rem; font-weight: 600; margin-bottom: 1rem; margin-top: 2.5rem; padding-bottom: 0.6rem; position: relative; }}
        .stSidebar h5::after {{ content: ''; position: absolute; bottom: 0; left: 0; width: 30px; height: 2px; background: var(--gradient-primary); border-radius: 2px; }}
        .portfolio-summary .stMetric {{ padding: 0.3rem 0; text-align: left; display: flex; flex-direction: column; align-items: flex-start; }}
        .portfolio-summary .stMetric > label {{ color: var(--color-text-muted); font-size: 0.75rem; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px; margin-bottom: 0.3rem; order: 1; }}
        .portfolio-summary .stMetric > div[data-testid="stMetricValue"] {{ font-size: 1.25rem; color: var(--color-text-primary); font-weight: 700; line-height: 1.3; order: 2; }}
        .portfolio-summary div[data-testid="stHorizontalBlock"] > div[data-testid="stVerticalBlock"] {{ gap: 0.5rem; }}

        /* Holdings List */
        .stSidebar .stExpander {{ background-color: transparent; border: 1px solid var(--color-border); border-radius: 12px; margin-top: 1.5rem; overflow: hidden; }}
        .stSidebar .stExpander header {{ font-size: 1rem; font-weight: 600; color: var(--color-text-primary); padding: 1rem 1.2rem; border-bottom: 1px solid var(--color-border); transition: all 0.2s ease; }}
        .stSidebar .stExpander:hover header {{ background-color: rgba(30, 41, 59, 0.5); }}
        .stSidebar .stExpander[aria-expanded="true"] header {{ border-bottom: 1px solid var(--color-border); }}
        .stSidebar .stExpander svg {{ fill: var(--color-text-secondary); }}
        .holding-list-container {{ max-height: 350px; overflow-y: auto; padding: 8px 0; scrollbar-width: thin; scrollbar-color: var(--color-border) var(--color-surface); margin: 0; }}
        .holding-item {{ font-size: 0.9rem; display: flex; justify-content: space-between; align-items: center; padding: 12px 16px; border-bottom: 1px solid var(--color-border); transition: all 0.15s ease; cursor: default; }}
        .holding-item:last-child {{ border-bottom: none; }}
        .holding-item:hover {{ background-color: rgba(30, 41, 59, 0.7); transform: translateX(2px); }}
        .holding-item span:first-child {{ font-weight: 600; color: var(--color-text-primary); flex-shrink: 0; margin-right: 12px; }}
        .holding-item .value-alloc {{ text-align: right; color: var(--color-text-secondary); font-weight: 500; font-size: 0.9rem; line-height: 1.3; flex-grow: 1; display: flex; flex-direction: column; align-items: flex-end; }}
        .holding-item .value-alloc span {{ min-width: 45px; text-align: right; color: var(--color-text-primary); font-weight: 600; font-size: 0.9rem; }}
        .holding-item small {{ color: var(--color-text-muted); font-size: 0.85rem; display: block; margin-top: 3px; }}

        /* Chat Area */
        .stChatMessage {{ background-color: transparent; border: 1px solid var(--color-border); border-radius: 14px; padding: 16px 22px; margin-bottom: 1.2rem; color: var(--color-text-secondary); box-shadow: 0 4px 10px rgba(0,0,0,0.1); max-width: 90%; line-height: 1.7; display: flex; align-items: flex-start; clear: both; float: left; }}
        .stChatMessage span[data-testid^="chatAvatarIcon"] {{ margin-right: 16px; margin-top: 4px; flex-shrink: 0; font-size: 1.2rem; }}
        .stChatMessage > div:not(:has(span[data-testid^="chatAvatarIcon"])) {{ flex-grow: 1; }}
        /* User Message */
        div[data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-user"]) {{ background: var(--gradient-primary); border-color: transparent; color: #e0e7ff; float: right; flex-direction: row-reverse; box-shadow: 0 4px 10px rgba(99, 102, 241, 0.2); }}
        div[data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-user"]) span[data-testid^="chatAvatarIcon"] {{ margin-right: 0; margin-left: 16px; }}
        div[data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-user"]) p,
        div[data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-user"]) div,
        div[data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-user"]) li {{ color: #e0e7ff !important; }}
        div[data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-user"]) a {{ color: #c7d2fe !important; text-decoration: underline; }}
        div[data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-user"]) code {{ background-color: rgba(255, 255, 255, 0.15) !important; color: #e0e7ff !important; }}
        div[data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-user"]) strong {{ color: white !important; }}
        /* Assistant Message */
        div[data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-assistant"]) {{ background-color: var(--color-surface-light); color: var(--color-text-primary); max-width: 85%; }} /* Slightly lighter background for assistant */
        div[data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-assistant"]) p,
        div[data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-assistant"]) li {{ color: var(--color-text-primary); }}
        div[data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-assistant"]) strong {{ color: var(--color-text-primary); font-weight: 700; }}
        div[data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-assistant"]) table {{ border-color: var(--color-border-light); box-shadow: none; }} /* Adjust table style if needed */
        div[data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-assistant"]) th {{ background-color: var(--color-surface); border-color: var(--color-border-light); }}
        div[data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-assistant"]) td {{ border-color: var(--color-border-light); }}
        div[data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-assistant"]) tr:nth-child(even) td {{ background-color: rgba(67, 85, 105, 0.3); }} /* Slightly lighter even row */


        /* Chat Input */
        div[data-testid="stChatInput"] {{ background-color: var(--color-background); border-top: 1px solid var(--color-border); padding: 1rem 1.8rem 1.2rem 1.8rem; position: sticky; bottom: 0; z-index: 10; }}
        div[data-testid="stChatInput"] textarea {{ font-family: 'Inter', sans-serif !important; background-color: var(--color-surface) !important; color: var(--color-text-primary) !important; border: 1px solid var(--color-border) !important; border-radius: 12px !important; padding: 1rem 1.3rem !important; font-size: 1rem !important; line-height: 1.6 !important; min-height: 60px; box-shadow: none; transition: all 0.2s ease; }}
        div[data-testid="stChatInput"] textarea:focus {{ border-color: var(--color-primary) !important; box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.3) !important; }}
        div[data-testid="stChatInput"] button {{ background: var(--gradient-primary) !important; border-radius: 10px !important; bottom: 18px; right: 12px; transition: all 0.2s ease !important; box-shadow: 0 2px 8px rgba(99, 102, 241, 0.3) !important; }}
        div[data-testid="stChatInput"] button:hover {{ transform: translateY(-1px) !important; box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4) !important; }}
        div[data-testid="stChatInput"] button svg {{ fill: white !important; }}

        /* Error & Warning Messages */
        .error-message, .token-warning {{ border-left-width: 5px; padding: 1.2rem 1.5rem; border-radius: 8px; margin: 1.2rem 0; font-size: 0.95rem; line-height: 1.7; backdrop-filter: blur(5px); }}
        .error-message {{ background-color: rgba(239, 68, 68, 0.15); border-left-color: var(--color-danger); color: #fca5a5; }}
        .error-message strong {{ color: #f87171; }}
        .token-warning {{ background-color: rgba(245, 158, 11, 0.15); border-left-color: var(--color-warning); color: #fcd34d; }}
        .token-warning strong {{ color: #fbbf24; }}

        /* Spinner */
        .stSpinner > div {{ border-top-color: var(--color-primary) !important; border-right-color: var(--color-primary) !important; border-bottom-color: rgba(99, 102, 241, 0.3) !important; border-left-color: rgba(99, 102, 241, 0.3) !important; width: 28px !important; height: 28px !important; }}

        /* Markdown Content Styling */
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{ border-bottom: 1px solid var(--color-border); padding-bottom: 6px; margin-top: 2em; margin-bottom: 1.3em;}}
        .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {{ color: var(--color-text-primary); font-weight: 600; margin-top: 1.7em; margin-bottom: 0.8em; }}
        .stMarkdown code {{ background-color: var(--color-border); padding: 0.2em 0.5em; border-radius: 5px; font-size: 0.9em; color: var(--color-text-primary); font-family: Consolas, Monaco, 'Andale Mono', 'Ubuntu Mono', monospace;}}
        .stMarkdown pre {{ background-color: var(--color-background); border: 1px solid var(--color-border); padding: 1.2rem; border-radius: 10px; color: var(--color-text-primary); font-family: Consolas, Monaco, 'Andale Mono', 'Ubuntu Mono', monospace; white-space: pre-wrap; word-wrap: break-word; font-size: 0.95em; margin: 1.5em 0; }}
        .stMarkdown pre code {{ background-color: transparent; padding: 0; font-size: inherit; }}
        .stMarkdown table {{ width: 100%; border-collapse: collapse; margin: 2em 0; font-size: 0.95rem; border: 1px solid var(--color-border); box-shadow: 0 3px 6px rgba(0,0,0,0.1); }}
        .stMarkdown th {{ background-color: var(--color-surface); border: 1px solid var(--color-border); padding: 14px 18px; text-align: left; font-weight: 600; color: var(--color-text-primary); }}
        .stMarkdown td {{ border: 1px solid var(--color-border); padding: 14px 18px; color: var(--color-text-secondary); vertical-align: top; }}
        .stMarkdown tr:nth-child(even) td {{ background-color: rgba(30, 41, 59, 0.5); }}
        .stMarkdown ul, .stMarkdown ol {{ margin-left: 1.5em; padding-left: 1em; margin-bottom: 1.5em; color: var(--color-text-secondary); }}
        .stMarkdown li {{ margin-bottom: 0.8em; line-height: 1.7; }}
        .stMarkdown li > p {{ margin-bottom: 0.5em; }}
        .stMarkdown li::marker {{ color: var(--color-text-secondary); }}
        .stMarkdown blockquote {{ border-left: 5px solid var(--color-primary); margin-left: 0; padding: 1rem 2rem; background-color: rgba(30, 41, 59, 0.5); color: var(--color-text-secondary); font-style: italic; border-radius: 0 8px 8px 0;}}
        .stMarkdown blockquote p {{ color: inherit; margin-bottom: 0; }}

        /* Custom scrollbar */
        ::-webkit-scrollbar {{ width: 10px; height: 10px; }}
        ::-webkit-scrollbar-track {{ background: var(--color-surface); border-radius: 10px; }}
        ::-webkit-scrollbar-thumb {{ background: #475569; border-radius: 10px; border: 2px solid var(--color-surface); }}
        ::-webkit-scrollbar-thumb:hover {{ background: #64748b; }}

        </style>
        """, unsafe_allow_html=True)

    except Exception as e:
        logger.error(f"UI setup failed: {e}", exc_info=True)
        st.error("Failed to initialize application UI styling.")

# --- Resource Initialization (Cached) ---
@st.cache_resource(show_spinner="Connecting to Database...")
def get_db_client() -> PostgresClient:
    logger.info("Initializing PostgresClient...")
    try:
        client = PostgresClient()
        return client
    except Exception as e:
        logger.critical(f"CRITICAL: Failed to initialize PostgresClient: {e}", exc_info=True)
        st.error(generate_error_html("Database Connection Failed!", f"Could not connect to the database. Please check configuration and network. Error: {e}"), icon="🚨")
        st.stop() # Stop the app if DB fails

@st.cache_resource(show_spinner="Loading Knowledge Base...")
def get_vector_store_client() -> FAISSClient:
    logger.info("Initializing FAISSClient...")
    try:
        client = FAISSClient()
        if client.index is None or client.index.ntotal == 0:
             logger.warning("FAISS index seems empty or failed to load properly.")
             # Displaying warning inside cached function might not always show, better handled in main flow
             # st.warning("Knowledge base (vector store) might be empty or unavailable.", icon="⚠️")
        return client
    except Exception as e:
        logger.critical(f"CRITICAL: Failed to initialize FAISSClient: {e}", exc_info=True)
        st.error(generate_error_html("Knowledge Base Failed!", f"Could not load the vector store. Contextual answers might be impaired. Error: {e}"), icon="🚨")
        return None # Return None to indicate failure

# --- CORRECTED FUNCTION DEFINITION ---
@st.cache_resource(show_spinner="Initializing AI Advisor...")
def get_agent_orchestrator(_db_client: PostgresClient, _vector_store_client: FAISSClient) -> AgentOrchestrator:
    """
    Initializes the AgentOrchestrator.
    Underscores added to arguments to prevent Streamlit from hashing them.
    """
    logger.info("Initializing AgentOrchestrator...")
    try:
        # Use the arguments with underscores inside the function
        orchestrator = AgentOrchestrator(_db_client, _vector_store_client)
        return orchestrator
    except Exception as e:
        logger.critical(f"CRITICAL: Failed to initialize AgentOrchestrator: {e}", exc_info=True)
        st.error(generate_error_html("AI System Failed!", f"The core AI components could not be initialized. Error: {e}"), icon="🚨")
        st.stop() # Stop the app if orchestrator fails

# --- Main Application ---
def main():
    """Main function to orchestrate the Streamlit application."""
    setup_ui() # Apply styling first

    st.title(settings.app_title)
    st.markdown(f"<p>Your AI partner for financial insights and portfolio analysis.</p>", unsafe_allow_html=True)

    # --- Initialize Core Components ---
    orchestrator = None # Initialize to None
    db_client = None
    try:
        db_client = get_db_client()
        vector_store_client = get_vector_store_client() # Might be None

        # Pass the retrieved clients to the orchestrator initializer
        # NOTE: The arguments passed here DO NOT have underscores
        orchestrator = get_agent_orchestrator(db_client, vector_store_client)

        # Display warning if vector store failed but we are continuing
        if vector_store_client is None:
             st.warning("Knowledge base failed to load. Contextual answers from documents may be unavailable.", icon="⚠️")

    except Exception as init_error:
         # Errors during get_db_client or get_agent_orchestrator should stop the app via st.stop()
         # This catch is for potential logic errors in the sequence itself
         logger.critical(f"Failed during component initialization sequence: {init_error}", exc_info=True)
         if not st.errors_displayed: # Check if error already shown by cached func
            st.error(generate_error_html("Application Initialization Error", f"A core component failed to start. Please check logs. Error: {init_error}"), icon="💥")
         st.stop()

    # --- Sidebar ---
    # Ensure db_client is valid before calling sidebar
    if db_client:
        sidebar.manage_sidebar(db_client)
    else:
        # Handle case where db_client failed init (though get_db_client should stop)
        st.sidebar.error("Database connection failed. Cannot load client data.", icon="🚨")


    # --- Main Chat Area ---
    if orchestrator and db_client:
        # Pass the initialized orchestrator and db_client to the chat interface
        chat_interface.display_chat_messages()
        chat_interface.handle_chat_input(orchestrator, db_client)
    else:
        # Display error if orchestrator or db_client failed to initialize
        st.markdown(f"<hr style='border-top: 1px solid var(--color-border); margin: 2rem 0;'>", unsafe_allow_html=True)
        st.warning("🔴 AI Advisor features are unavailable due to an initialization error.", icon="⚠️")


# --- Application Entry Point & Final Exception Handler ---
if __name__ == "__main__":
    # Add a global flag to track if errors were displayed by st.error
    st.errors_displayed = False
    try:
        main()
    except SystemExit:
        logger.info("Application stopped via st.stop() or SystemExit.")
        pass # Allow clean exit from st.stop()
    except Exception as e:
        critical_logger = get_logger("MainCriticalError")
        error_type = type(e).__name__
        error_message = str(e)
        full_traceback = traceback.format_exc()
        critical_logger.critical(f"Application encountered CRITICAL unhandled error: {error_type} - {error_message}", exc_info=True)
        critical_logger.critical(full_traceback)

        # Attempt to display error in Streamlit if possible and not already shown
        try:
            # Check the flag before displaying a generic error
            if not getattr(st, 'errors_displayed', False):
                st.error(generate_error_html("Critical Application Error!",
                         f"A critical error occurred: {error_type}. Please check the application logs or contact support."),
                         icon="💥")
                st.errors_displayed = True # Set flag
        except Exception as st_err:
            critical_logger.critical(f"!!! FAILED to display critical error via st.error: {st_err}", exc_info=True)
            # Fallback to printing to stderr
            print(f"\n--- CRITICAL UNHANDLED ERROR ---", file=sys.stderr)
            print(f"Timestamp: {datetime.now()}", file=sys.stderr)
            print(f"Original Error: {error_type} - {error_message}", file=sys.stderr)
            print(f"Traceback:\n{full_traceback}", file=sys.stderr)
            print(f"Error during st.error display: {type(st_err).__name__} - {st_err}", file=sys.stderr)
            print(f"--- END CRITICAL ERROR ---\n", file=sys.stderr)