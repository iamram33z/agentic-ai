# main.py
import streamlit as st
# Standard library imports
import os
import time
import json
import re
from datetime import datetime
import sys
import traceback
# Third-party imports
from dotenv import load_dotenv
# Local application imports
from agent import FinancialAgents # Assuming agent.py is in the same directory or accessible path
from azure_postgres import AzurePostgresClient # Assuming azure_postgres.py is accessible
from utils import (
    get_logger,
    generate_error_html,
    generate_warning_html, # Import warning html
    validate_query_text,
    validate_portfolio_data,
    validate_client_id,
    log_execution_time, # If used
    # process_raw_portfolio_data # Not directly used here, but could be
)
from typing import Optional
# --- Load Environment Variables & Initialize Logger ---
load_dotenv()
# Use the configured logger from utils.py
logger = get_logger("StreamlitApp")

# --- UI Setup ---
def setup_ui():
    """Configure Streamlit UI layout and styling."""
    try:
        st.set_page_config(
            page_title="FinTech AI Advisor",
            layout="wide",
            page_icon="💹",
            initial_sidebar_state="expanded"
        )
        # Define CSS (Consider moving to a separate CSS file for maintainability)
        st.markdown("""
        <style>
            /* General Card Style */
            .card { padding: 20px; border-radius: 10px; background-color: #1e293b; color: #f8fafc; margin-bottom: 20px; border-left: 4px solid #3b82f6; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }
            /* Stock Card specific style */
            .stock-card { padding: 15px; border-radius: 8px; background-color: #334155; margin: 10px 0; border: 1px solid #475569; color: #f8fafc; }
            /* Error message styling */
            .error-message { color: #fecaca; background-color: #7f1d1d; padding: 15px; border-radius: 8px; border: 1px solid #ef4444; border-left-width: 4px; margin: 10px 0; font-family: sans-serif; }
            .error-message strong { color: #ffffff; }
            .error-message p { font-family: monospace; color: #fca5a5; white-space: pre-wrap; margin-top: 8px; font-size: 0.85rem; }
            /* Warning message styling */
            .token-warning { color: #fef3c7; background-color: #92400e; padding: 15px; border-radius: 8px; border: 1px solid #f59e0b; border-left-width: 4px; margin: 10px 0; font-family: sans-serif; }
            .token-warning strong { color: #ffffff; }
            .token-warning p { color: #fde68a; margin-top: 8px; font-size: 0.9rem; }
            /* Chat message styling */
            .stChatMessage { background-color: #334155; border-radius: 10px; padding: 10px 15px; margin-bottom: 10px; border: 1px solid #475569; color: #e2e8f0; }
            /* Card headers */
            .card h3 { margin-top: 0; color: #e2e8f0; border-bottom: 1px solid #475569; padding-bottom: 8px; margin-bottom: 15px; font-size: 1.15rem; }
            /* Text labels/values within cards */
            .card p, .card span, .card div { color: #cbd5e1; }
            .card p.label { font-size: 0.85rem; color: #94a3b8; margin-bottom: 2px; text-transform: uppercase; }
            .card p.value-large { font-size: 1.5rem; font-weight: 600; color: #f8fafc; margin: 0; line-height: 1.2; }
            .card p.value-medium { font-size: 1.1rem; font-weight: 600; color: #f8fafc; margin: 0; }
            .card p.footnote { font-size: 0.8rem; color: #94a3b8; margin: 0; }
            /* Allocation bar */
            .allocation-bar-bg { height: 8px; background-color: #475569; border-radius: 4px; overflow: hidden; margin-top: 5px; }
            .allocation-bar-fg { height: 100%; background-color: #3b82f6; border-radius: 4px; }
            /* Sidebar styling */
            .stSidebar > div:first-child { background-color: #0f172a; /* slate-900 darker */ padding-top: 1rem; border-right: 1px solid #334155; }
            .stSidebar .stTextInput label { color: #94a3b8; font-size: 0.9rem; }
            .stSidebar .stButton button { border: 1px solid #3b82f6; background-color: transparent; color: #cbd5e1; width: 100%; margin-top: 10px; transition: background-color 0.2s ease, color 0.2s ease; }
            .stSidebar .stButton button:hover { background-color: #3b82f6; color: white; border-color: #3b82f6; }
            .stSidebar .stButton button:active { background-color: #2563eb; /* Slightly darker blue on click */ }
            /* Spinner color */
            .stSpinner > div { border-top-color: #3b82f6 !important; border-right-color: #3b82f6 !important; }
            /* Markdown adjustments */
            .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { color: #e2e8f0; border-bottom: 1px solid #475569; padding-bottom: 5px; margin-top: 1.5em; }
            .stMarkdown code { background-color: #475569; padding: 0.2em 0.4em; border-radius: 3px; font-size: 0.9em; }
            .stMarkdown pre > code { background-color: #1e293b; border: 1px solid #475569; display: block; padding: 10px; border-radius: 5px; }
        </style>
        """, unsafe_allow_html=True)
    except Exception as e:
        logger.error(f"UI setup failed: {str(e)}", exc_info=True)
        st.error("Failed to initialize application UI styling.")

# --- Database Interaction & Caching ---
@st.cache_data(ttl=300, show_spinner="Fetching client portfolio...")
def get_client_portfolio_cached(client_id: str) -> Optional[dict]:
    """Cached wrapper for fetching and processing portfolio data."""
    local_logger = get_logger("PortfolioCache")
    if not client_id or not isinstance(client_id, str):
        local_logger.warning("Invalid client ID format provided for caching.")
        return None
    local_logger.info(f"Cache check/fetch for client ID: {client_id}")
    try:
        db_client = AzurePostgresClient()
        raw_portfolio = db_client.get_client_portfolio(client_id)
        if not raw_portfolio or not isinstance(raw_portfolio, dict):
            local_logger.warning(f"No portfolio data found or invalid format from DB for client ID: {client_id}")
            return None # Cache None result for non-existent data
        local_logger.info(f"Raw portfolio data retrieved from DB for {client_id}, processing...")
        # Process raw data into standardized format (moved logic outside cache function)
        processed_portfolio = process_raw_portfolio_data(client_id, raw_portfolio)
        if processed_portfolio and validate_portfolio_data(processed_portfolio):
            local_logger.info(f"Successfully processed/validated portfolio for {client_id}.")
            return processed_portfolio
        else:
            local_logger.error(f"Processed DB portfolio failed validation for client {client_id}.")
            return None # Cache None if processing/validation fails
    except Exception as e:
        local_logger.error(f"Database error during cache fetch/processing for client {client_id}: {e}", exc_info=True)
        # Do not cache DB errors, allow retries. Return None to signal failure.
        return None

# Separate function for processing logic (makes cache function cleaner)
# @log_execution_time # Optional performance logging
def process_raw_portfolio_data(client_id: str, raw_portfolio: dict) -> Optional[dict]:
    """Processes raw portfolio data from DB into the standard context format."""
    local_logger = get_logger("PortfolioProcessing")
    try:
        portfolio = {
            "id": client_id,
            "name": f"Client {client_id[-4:]}" if len(client_id) >= 4 else f"Client {client_id}",
            "risk_profile": raw_portfolio.get('risk_profile', 'Not specified') or 'Not specified',
            "portfolio_value": 0.0,
            "holdings": [],
            "last_update": raw_portfolio.get('last_updated', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        }
        db_total_value = raw_portfolio.get('total_value')
        holdings_data = raw_portfolio.get('holdings', [])
        calculated_total_value = 0.0
        if isinstance(holdings_data, list):
            for h in holdings_data:
                 if isinstance(h, dict):
                    value = h.get('current_value', 0)
                    if isinstance(value, (int, float)): calculated_total_value += float(value)
        # Determine final portfolio value
        if isinstance(db_total_value, (int, float)) and db_total_value > 0:
             portfolio['portfolio_value'] = float(db_total_value)
        elif calculated_total_value >= 0: # Allow zero value
             portfolio['portfolio_value'] = calculated_total_value
        # Process holdings
        if isinstance(holdings_data, list):
            total_val = portfolio['portfolio_value']
            for holding in holdings_data:
                if not isinstance(holding, dict) or not holding.get('symbol'): continue
                symbol = str(holding['symbol']).strip().upper()
                if not symbol: continue
                value = holding.get('current_value', 0)
                if not isinstance(value, (int, float)): value = 0.0
                else: value = float(value)
                allocation = (value / total_val * 100) if total_val > 0 else 0.0
                portfolio['holdings'].append({"symbol": symbol, "value": value, "allocation": allocation})
        # Final validation before returning
        if validate_portfolio_data(portfolio):
            return portfolio
        else:
            local_logger.error(f"Final processed portfolio failed validation for client {client_id}.")
            return None
    except Exception as e:
        local_logger.error(f"Error processing raw portfolio for {client_id}: {e}", exc_info=True)
        return None

# --- UI Components ---
def display_portfolio(portfolio: dict):
    """Displays portfolio summary and holdings in the sidebar."""
    if not portfolio or not isinstance(portfolio, dict) or not validate_portfolio_data(portfolio):
        st.sidebar.error("Invalid portfolio data for display.")
        logger.warning(f"display_portfolio called with invalid data: {portfolio}")
        return
    try:
        client_id = portfolio.get('id', 'N/A')
        total_value = portfolio.get('portfolio_value', 0.0)
        risk_profile = portfolio.get('risk_profile', 'N/A')
        holdings = portfolio.get('holdings', [])
        # --- Summary Section ---
        st.sidebar.markdown(f"""
        <div style="margin-bottom: 15px; padding: 10px; background-color: #1e293b; border-radius: 8px; border: 1px solid #334155;">
            <p class="label" style="margin-bottom: 5px;">Client: {client_id}</p>
            <p class="value-medium" style="margin-bottom: 8px;">${total_value:,.2f}</p>
            <p class="label" style="margin-top: 8px;">Risk: {risk_profile.capitalize()}</p>
        </div>
        """, unsafe_allow_html=True)
        # --- Holdings Expander ---
        if holdings:
            with st.sidebar.expander(f"Holdings ({len(holdings)})", expanded=False): # Start collapsed
                 # Scrollable div for holdings list
                 st.markdown('<div style="max-height: 300px; overflow-y: auto; padding-right: 5px;">', unsafe_allow_html=True)
                 sorted_holdings = sorted(holdings, key=lambda x: x.get('value', 0), reverse=True)
                 for holding in sorted_holdings:
                     symbol = holding.get('symbol', 'ERR')
                     value = holding.get('value', 0.0)
                     allocation = holding.get('allocation', 0.0)
                     # Compact display for sidebar
                     st.markdown(f"""
                     <div style="font-size: 0.9rem; display: flex; justify-content: space-between; margin-bottom: 5px; padding: 5px 0; border-bottom: 1px solid #334155;">
                         <span>{symbol}</span>
                         <span style="text-align: right;">{allocation:.1f}%<br><small style='color:#94a3b8'>${value:,.0f}</small></span>
                     </div>
                     """, unsafe_allow_html=True)
                 st.markdown('</div>', unsafe_allow_html=True) # Close scrollable div
        else:
            st.sidebar.info("No holdings data found.")
    except Exception as e:
        logger.error(f"Sidebar portfolio display error for client {client_id}: {e}", exc_info=True)
        st.sidebar.error("Error displaying portfolio.")

def client_sidebar():
    """Manages client ID input, loading, and portfolio display in the sidebar."""
    with st.sidebar:
        st.markdown("## Client Access")
        if 'client_id_input' not in st.session_state: st.session_state.client_id_input = ""
        client_id_input = st.text_input(
            "Client ID", value=st.session_state.client_id_input,
            placeholder="e.g., CLIENT123", help="Enter client ID to load portfolio.",
            key="client_id_input_widget"
        )
        st.session_state.client_id_input = client_id_input
        load_button = st.button("Load Client Data", key="load_client_button", use_container_width=True)
        # --- Logic for Loading ---
        if load_button and client_id_input:
            validated_client_id = validate_client_id(client_id_input)
            if validated_client_id:
                st.session_state.client_id = validated_client_id
                st.session_state.client_context = None # Reset state
                st.session_state.portfolio_loaded = False
                st.session_state.load_error = None
                # Call cached function
                portfolio_data = get_client_portfolio_cached(validated_client_id)
                if portfolio_data: # Success
                    st.session_state.client_context = portfolio_data
                    st.session_state.portfolio_loaded = True
                    logger.info(f"Context loaded for {validated_client_id}")
                else: # Failure (no data or DB error)
                    st.session_state.load_error = f"Could not load portfolio for {validated_client_id}. Check ID or logs."
                    logger.warning(st.session_state.load_error)
                st.rerun() # Rerun to update display
            else: # Validation failed
                 st.session_state.load_error = f"Invalid Client ID format: '{client_id_input}'."
                 st.session_state.portfolio_loaded = False
                 st.session_state.client_context = None
                 logger.warning(st.session_state.load_error)
                 st.rerun()
        # --- Display Logic ---
        if st.session_state.get('portfolio_loaded') and st.session_state.get('client_context'):
            st.success(f"Client {st.session_state.client_id} Loaded", icon="✅")
            display_portfolio(st.session_state.client_context)
        elif st.session_state.get('load_error'):
            st.error(st.session_state.load_error, icon="⚠️")
        elif st.session_state.get('client_id_input') and not st.session_state.get('client_id'):
            # ID entered but not loaded (e.g., initial state or after clearing error)
            st.info("Click 'Load Client Data'.")

# --- Main Chat Interface ---
# @log_execution_time # Optional performance logging
def main_chat_interface(agents: FinancialAgents):
    """Handles the main chat interaction area."""
    st.title("💹 AI Financial Advisor")
    st.markdown("""<p style="font-size: 1.1rem; color: #cbd5e1; margin-top:-10px; margin-bottom: 20px;">
                Ask about market trends, specific investments, or portfolio analysis (if client data is loaded).</p>
                """, unsafe_allow_html=True)
    # Init chat history
    if 'conversation' not in st.session_state:
        st.session_state.conversation = [{"role": "assistant", "content": "Hello! How can I assist you today?"}]
    # Display past messages
    for message in st.session_state.conversation:
        with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
            st.markdown(message["content"], unsafe_allow_html=True)
    # Handle new user input
    if prompt := st.chat_input("Ask a financial question..."):
        if not validate_query_text(prompt):
             st.warning("Please enter a more specific question (3-1500 characters).")
        else:
            # Add user message to UI and history
            st.chat_message("user", avatar="👤").markdown(prompt)
            st.session_state.conversation.append({"role": "user", "content": prompt})
            # Get agent response
            with st.chat_message("assistant", avatar="🤖"):
                message_placeholder = st.empty()
                message_placeholder.markdown("Thinking... ⏳")
                try:
                    client_context = st.session_state.get('client_context')
                    client_id = st.session_state.get('client_id') # Loaded & validated ID
                    log_prefix = f"Client {client_id}" if client_id else "Generic"
                    logger.info(f"{log_prefix}: Calling agent.get_response for query: '{prompt[:50]}...'")
                    # Call the main agent function
                    response = agents.get_response(query=prompt, client_id=client_id, client_context=client_context)
                    message_placeholder.markdown(response, unsafe_allow_html=True) # Display result/error
                    st.session_state.conversation.append({"role": "assistant", "content": response}) # Add to history
                    # Log successful interaction to DB
                    is_error_response = '<div class="error-message">' in response
                    if client_id and not is_error_response:
                        try:
                            db_client = AzurePostgresClient()
                            db_client.log_client_query(client_id=client_id, query=prompt[:1000], response=response[:2000])
                            logger.info(f"Logged query for client {client_id}")
                        except Exception as db_error: logger.error(f"DB log failed for {client_id}: {db_error}")
                except Exception as agent_call_error:
                    logger.error(f"Error calling FinancialAgents.get_response: {agent_call_error}", exc_info=True)
                    error_html = generate_error_html("Failed to Process Request", f"Error: {agent_call_error}")
                    message_placeholder.markdown(error_html, unsafe_allow_html=True)
                    st.session_state.conversation.append({"role": "assistant", "content": error_html})

# --- Application Entry Point ---
def main():
    """Main function to run the Streamlit application."""
    setup_ui() # Setup UI first

    # Initialize agents (cached)
    @st.cache_resource(show_spinner="Initializing AI Advisor...")
    def initialize_financial_agents():
        """Initializes the FinancialAgents system once per session."""
        local_logger = get_logger("AgentInitialization")
        local_logger.info("Attempting to initialize FinancialAgents...")
        try:
            return FinancialAgents()
        except Exception as e:
            local_logger.critical(f"CRITICAL: FinancialAgents initialization failed: {e}", exc_info=True)
            error_html = generate_error_html("System Initialization Failed!", f"AI agents could not load. Error: {e}")
            st.error(error_html, icon="🚨")
            return None # Indicate failure

    financial_agents = initialize_financial_agents()

    # Manage client loading/display in sidebar
    client_sidebar()

    # Display main chat interface if agents are ready
    if financial_agents:
        main_chat_interface(financial_agents)
    else:
         # Error already shown by initialize_financial_agents
         st.warning("🔴 AI Advisor functionality is offline due to system initialization issues.")

# --- Final Application Exception Handler ---
if __name__ == "__main__":
    try:
        main()
    except SystemExit: pass # Allow st.stop() to exit cleanly
    except Exception as e:
        critical_logger = get_logger("MainCriticalError")
        error_type = type(e).__name__
        error_message = str(e)
        full_traceback = traceback.format_exc()
        critical_logger.critical(f"Application encountered critical unhandled error: {error_type} - {error_message}", exc_info=True)
        critical_logger.critical(full_traceback)
        # Try to display error in Streamlit UI as last resort
        try:
            st.error(generate_error_html("Critical Application Error!", f"Error Type: {error_type}. Check logs."), icon="💥")
        except Exception as st_err:
            critical_logger.critical(f"!!! FAILED to display critical error via st.error: {st_err}", exc_info=True)
            # Print to console if UI fails
            print(f"\n--- CRITICAL UNHANDLED ERROR ---", file=sys.stderr)
            print(f"Error: {error_type} - {error_message}", file=sys.stderr)
            print(f"Traceback:\n{full_traceback}", file=sys.stderr)
            print(f"--- END CRITICAL ERROR ---\n", file=sys.stderr)