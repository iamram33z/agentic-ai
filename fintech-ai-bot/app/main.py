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
from typing import Optional, Dict # Added Dict type hint

# Third-party imports
from dotenv import load_dotenv

# Local application imports
# Ensure these modules are accessible (e.g., in the same directory or added to PYTHONPATH)
try:
    from agent import FinancialAgents
    from azure_postgres import AzurePostgresClient # Assuming class exists
    # process_raw_portfolio_data moved into main.py
    from utils import (
        get_logger,
        generate_error_html,
        generate_warning_html,
        validate_query_text,
        validate_portfolio_data, # Keep validation util
        validate_client_id,
        log_execution_time, # Keep if used elsewhere or in utils
        # Add any other utils ACTUALLY used by main
    )
except ImportError as e:
    # If imports fail, show an error message and stop
    st.error(f"Failed to import necessary modules: {e}. Please ensure agent.py, utils.py, and azure_postgres.py are available.", icon="🚨")
    st.stop()


# --- Load Environment Variables & Initialize Logger ---
load_dotenv()
logger = get_logger("StreamlitApp") # Use the configured logger from utils

# --- Constants ---
APP_TITLE = "FinTech AI Advisor"
APP_ICON = "💹"

# --- UI Setup & Styling ---
def setup_ui():
    """Configure Streamlit page settings and inject custom CSS."""
    try:
        st.set_page_config(
            page_title=APP_TITLE,
            layout="wide",
            page_icon=APP_ICON,
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'mailto:support@example.com', # Replace with actual help link
                'Report a bug': "mailto:support@example.com", # Replace
                'About': f"### {APP_TITLE}\nYour AI-powered financial guidance."
            }
        )
        # Comprehensive CSS (same as provided previously)
        st.markdown("""
        <style>
            /* Base & General */
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
            .stApp { background-color: #0f172a; } /* Main app background */
            h1 { color: #e2e8f0; font-weight: 600; border-bottom: 2px solid #3b82f6; padding-bottom: 0.3em; }
            h2, h3 { color: #cbd5e1; font-weight: 500; }

            /* Sidebar Enhancements */
            .stSidebar > div:first-child { background-color: #1e293b; border-right: 1px solid #334155; }
            .stSidebar h2 { color: #94a3b8; font-size: 1.1rem; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 1rem; }
            .stSidebar h5 { color: #a1a1aa; font-size: 1.0rem; font-weight: 500; margin-bottom: 0.5rem; } /* Styling for 'Portfolio Snapshot' */
            .stSidebar .stTextInput label { color: #94a3b8; font-size: 0.9rem; font-weight: 500; }
            .stSidebar .stTextInput input { border-radius: 6px; border: 1px solid #475569; background-color: #334155; color: #e2e8f0; }
            .stSidebar .stButton button { background-color: #3b82f6; color: white; border: none; border-radius: 6px; padding: 0.6em 1em; width: 100%; margin-top: 10px; font-weight: 500; transition: background-color 0.2s ease; }
            .stSidebar .stButton button:hover { background-color: #2563eb; }
            .stSidebar .stButton button:active { background-color: #1d4ed8; }
            .stSidebar .stExpander { background-color: #1e293b; border: 1px solid #334155; border-radius: 8px; margin-top: 10px; }
            .stSidebar .stExpander header { font-size: 0.95rem; font-weight: 500; color: #cbd5e1; }

            /* Portfolio Display in Sidebar */
            .portfolio-summary { padding: 15px 0px 10px 0px; /* Adjusted padding */ margin-bottom: 15px; }
            .portfolio-summary .stMetric { background-color: transparent !important; border: none !important; padding: 5px 0; }
            .portfolio-summary .stMetric > label { color: #94a3b8; font-size: 0.85rem; text-transform: uppercase; }
            .portfolio-summary .stMetric > div[data-testid="stMetricValue"] { font-size: 1.4rem; color: #f8fafc; font-weight: 600; }
            .portfolio-summary .stMetric > div[data-testid="stMetricDelta"] { display: none; } /* Hide delta if not needed */

            /* Holding list specific styling */
            .holding-list-container { max-height: 350px; overflow-y: auto; padding: 5px 10px 5px 5px; /* Adjusted padding */ scrollbar-width: thin; scrollbar-color: #475569 #1e293b; }
            .holding-item { font-size: 0.9rem; display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; padding: 6px 3px; border-bottom: 1px solid #475569; }
            .holding-item:last-child { border-bottom: none; }
            .holding-item span:first-child { font-weight: 500; color: #e2e8f0; }
            .holding-item .value-alloc { text-align: right; color: #cbd5e1; }
            .holding-item small { color:#94a3b8; font-size: 0.8rem; display: block; line-height: 1.1; margin-top: 1px; }

            /* Main Chat Area */
            .stChatMessage { background-color: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 12px 18px; margin-bottom: 12px; color: #e2e8f0; box-shadow: 0 2px 4px rgba(0,0,0,0.2); max-width: 85%; /* Limit width */ float: left; /* Default align left */ clear: both; }
            /* User message alignment */
            div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) { background-color: #2563eb; border-color: #3b82f6; color: white; float: right; /* Align user messages right */ }
            .stChatMessage strong { color: #f8fafc; font-weight: 600; } /* Bold text color */
            .stChatInput > div { background-color: #1e293b; border-top: 1px solid #334155; }

            /* Error & Warning Messages (inline styles added in utils for robustness) */
            .error-message { /* Base styles handled by utils function */ }
            .token-warning { /* Base styles handled by utils function */ }

            /* Spinner color */
            .stSpinner > div { border-top-color: #3b82f6 !important; border-right-color: #3b82f6 !important; }
            /* Markdown adjustments */
            .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { color: #e2e8f0; border-bottom: 1px solid #475569; padding-bottom: 5px; margin-top: 1.5em; }
            .stMarkdown code { background-color: #475569; padding: 0.2em 0.4em; border-radius: 3px; font-size: 0.9em; color: #e2e8f0; }
            .stMarkdown pre { background-color: #0f172a; border: 1px solid #475569; display: block; padding: 10px; border-radius: 5px; color: #cbd5e1; font-family: 'Courier New', Courier, monospace; white-space: pre-wrap; word-wrap: break-word; }
            .stMarkdown table { width: auto; border-collapse: collapse; margin: 1em 0; font-size: 0.9rem; } /* Slightly smaller table font */
            .stMarkdown th { background-color: #334155; border: 1px solid #475569; padding: 6px 10px; text-align: left; font-weight: 600; }
            .stMarkdown td { border: 1px solid #475569; padding: 6px 10px; }
            .stMarkdown li { margin-bottom: 0.5em; } /* Spacing for list items */

        </style>
        """, unsafe_allow_html=True)
    except Exception as e:
        logger.error(f"UI setup failed: {e}", exc_info=True)
        st.error("Failed to initialize application UI styling.")

# --- Portfolio Processing Logic (Defined within main.py) ---
# @log_execution_time # Optional performance logging
def process_raw_portfolio_data(client_id: str, raw_portfolio: dict) -> Optional[Dict]:
    """Processes raw portfolio data from DB into the standard context format."""
    local_logger = get_logger("PortfolioProcessing") # Use specific logger
    if not isinstance(raw_portfolio, dict):
        local_logger.warning(f"Invalid raw portfolio data type for {client_id}: {type(raw_portfolio)}")
        return None
    try:
        # Initialize portfolio structure
        portfolio = {
            "id": client_id,
            "name": f"Client {client_id[-4:]}" if len(client_id) >= 4 else f"Client {client_id}",
            "risk_profile": raw_portfolio.get('risk_profile', 'Not specified') or 'Not specified', # Ensure not None
            "portfolio_value": 0.0,
            "holdings": [],
            "last_update": raw_portfolio.get('last_updated', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        }
        db_total_value = raw_portfolio.get('total_value')
        holdings_data = raw_portfolio.get('holdings', [])
        calculated_total_value = 0.0

        # Calculate total value from holdings robustly
        if isinstance(holdings_data, list):
            for h in holdings_data:
                 if isinstance(h, dict):
                    value = h.get('current_value') # Get value first
                    # Check if value is valid number before adding
                    if isinstance(value, (int, float)):
                        calculated_total_value += float(value)
                    else:
                        local_logger.warning(f"Invalid 'current_value' type ({type(value)}) for holding {h.get('symbol')} in client {client_id}")
        else:
             local_logger.warning(f"Holdings data for client {client_id} is not a list: {type(holdings_data)}")

        # Determine final portfolio value (prefer explicit total_value if valid)
        if isinstance(db_total_value, (int, float)) and db_total_value > 0:
             portfolio['portfolio_value'] = float(db_total_value)
        elif calculated_total_value >= 0: # Use calculated value if db total is invalid/missing
             portfolio['portfolio_value'] = calculated_total_value
        else:
             local_logger.warning(f"Could not determine valid portfolio value for client {client_id}")
             # Keep portfolio_value as 0.0

        # Process holdings list
        if isinstance(holdings_data, list):
            total_val = portfolio['portfolio_value']
            for holding in holdings_data:
                if not isinstance(holding, dict) or not holding.get('symbol'):
                     local_logger.debug(f"Skipping invalid holding item: {holding}")
                     continue # Skip malformed holding entries
                symbol = str(holding['symbol']).strip().upper()
                if not symbol:
                     local_logger.debug(f"Skipping holding item with empty symbol.")
                     continue

                value = holding.get('current_value', 0) # Default to 0 if missing
                if not isinstance(value, (int, float)):
                     local_logger.warning(f"Holding '{symbol}' for client {client_id} has invalid value type ({type(value)}), using 0.")
                     value = 0.0
                else: value = float(value)

                allocation = (value / total_val * 100) if total_val > 0 else 0.0
                portfolio['holdings'].append({"symbol": symbol, "value": value, "allocation": allocation})

        # Final validation before returning
        if validate_portfolio_data(portfolio): # Use validation util
            local_logger.debug(f"Processed portfolio validated successfully for {client_id}.")
            return portfolio
        else:
            # Log the portfolio that failed validation for debugging
            local_logger.error(f"Final processed portfolio FAILED validation for client {client_id}. Data: {portfolio}")
            return None # Return None if validation fails

    except Exception as e:
        local_logger.error(f"Critical error processing raw portfolio for {client_id}: {e}", exc_info=True)
        return None # Return None on unexpected errors

# --- Database Interaction & Caching ---
@st.cache_data(ttl=300, show_spinner="Fetching client portfolio...")
def get_client_portfolio_cached(client_id: str) -> Optional[Dict]:
    """Cached wrapper for fetching and processing portfolio data."""
    local_logger = get_logger("PortfolioCache")
    if not client_id or not isinstance(client_id, str):
        local_logger.warning("Invalid client ID format provided for caching.")
        return None

    # Validate ID format using util function before hitting DB/cache potentially
    validated_id = validate_client_id(client_id)
    if not validated_id:
         local_logger.warning(f"Invalid client ID format '{client_id}' blocked before cache fetch.")
         # Optionally return an error structure or specific message
         # For now, return None, the calling function should handle it
         return None

    local_logger.info(f"Cache check/fetch for client ID: {validated_id}")
    try:
        db_client = AzurePostgresClient() # Instantiate DB client
        raw_portfolio = db_client.get_client_portfolio(validated_id)

        if not raw_portfolio: # Handles None or empty dict/list
            local_logger.warning(f"No portfolio data returned from DB for {validated_id}")
            return None # No data found is a valid cacheable result (None)

        if not isinstance(raw_portfolio, dict):
            local_logger.error(f"Invalid data type received from DB for {validated_id}: {type(raw_portfolio)}")
            return None # Invalid data structure

        local_logger.info(f"Raw portfolio data retrieved from DB for {validated_id}, processing...")
        # Call the processing function defined locally in main.py
        # Pass the validated_id to ensure consistency
        processed_portfolio = process_raw_portfolio_data(validated_id, raw_portfolio)

        if processed_portfolio: # process_raw_portfolio_data includes validation
            local_logger.info(f"Successfully processed/validated portfolio for {validated_id}. Caching result.")
            return processed_portfolio
        else:
            # Processing or validation failed, log occurred in process_raw_portfolio_data
            local_logger.error(f"Processed DB portfolio failed validation/processing for {validated_id}. Caching None.")
            return None # Cache failure as None
    except Exception as e:
        local_logger.error(f"Database or processing error during cache fetch for {validated_id}: {e}", exc_info=True)
        # Decide if you want to cache None on error or raise/return specific error
        return None # Caching None on error

# --- UI Components ---
# @log_execution_time # Optional
def display_portfolio_summary(portfolio_data: Dict):
    """Displays portfolio summary using st.metric in the sidebar."""
    local_logger = get_logger("PortfolioDisplay")
    try:
        total_value = portfolio_data.get('portfolio_value', 0.0)
        # Capitalize risk profile, handle None/empty
        risk_profile_raw = portfolio_data.get('risk_profile')
        risk_profile = risk_profile_raw.capitalize() if risk_profile_raw else 'N/A'

        with st.sidebar.container():
            # Added CSS class for easier targeting if needed
            st.markdown('<div class="portfolio-summary">', unsafe_allow_html=True)
            st.markdown("##### Portfolio Snapshot") # Smaller heading
            # Use columns for better alignment of metric + risk
            col1, col2 = st.columns(2)
            with col1:
                 # Format value with commas and 2 decimal places
                 st.metric(label="Total Value", value=f"${total_value:,.2f}")
            with col2:
                 # Use 'N/A' if risk profile is missing/empty
                 st.metric(label="Risk Profile", value=risk_profile)
            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        local_logger.error(f"Error displaying portfolio summary: {e}", exc_info=True)
        st.sidebar.error("Error displaying summary.", icon="⚠️")

# @log_execution_time # Optional
def display_holdings_list(holdings_data: list):
    """Displays scrollable holdings list in the sidebar expander."""
    local_logger = get_logger("PortfolioDisplay")
    if not holdings_data:
        # Don't display expander if no holdings
        # st.sidebar.info("No holdings data available.") # Removed for cleaner look
        return
    try:
        with st.sidebar.expander(f"Holdings ({len(holdings_data)})", expanded=False):
            # Use the CSS class for the scrollable container
            st.markdown('<div class="holding-list-container">', unsafe_allow_html=True)
            # Sort holdings by value descending
            sorted_holdings = sorted(holdings_data, key=lambda x: x.get('value', 0), reverse=True)
            for holding in sorted_holdings:
                symbol = holding.get('symbol', 'ERR')
                value = holding.get('value', 0.0)
                allocation = holding.get('allocation', 0.0)
                # Use the holding-item class and structure defined in CSS
                st.markdown(f"""
                <div class="holding-item">
                    <span>{symbol}</span>
                    <span class="value-alloc">{allocation:.1f}%<small>${value:,.0f}</small></span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    except Exception as e:
        local_logger.error(f"Error displaying holdings list: {e}", exc_info=True)
        st.sidebar.error("Error displaying holdings.", icon="⚠️")

def client_sidebar_manager():
    """Manages client ID input, loading, and portfolio display calls in the sidebar."""
    with st.sidebar:
        st.markdown("## Client Access")
        # Initialize session state variables safely
        st.session_state.setdefault('client_id_input', "")
        st.session_state.setdefault('client_id', None) # The currently validated and loaded client ID
        st.session_state.setdefault('client_context', None) # The loaded portfolio data
        st.session_state.setdefault('portfolio_loaded', False)
        st.session_state.setdefault('load_error', None) # Store error messages

        # Use a key for the text input widget
        client_id_input = st.text_input(
            "Client ID",
            value=st.session_state.client_id_input,
            placeholder="e.g., CLIENT123",
            help="Enter client ID and click Load.",
            key="client_id_input_widget" # Assign a key
        )

        # Sync input value back to session state immediately (on_change might be better but this works)
        st.session_state.client_id_input = client_id_input

        load_button = st.button("Load Client Data", key="load_client_button", use_container_width=True)

        # --- Loading Logic ---
        if load_button:
            st.session_state.load_error = None # Clear previous error on new attempt
            if not client_id_input:
                 st.session_state.load_error = "Please enter a Client ID."
                 st.session_state.portfolio_loaded = False
                 st.session_state.client_context = None
                 st.session_state.client_id = None
            else:
                 # Validate ID format FIRST
                 validated_id = validate_client_id(client_id_input)
                 if validated_id:
                     # Check if we are trying to load the *same* ID that's already loaded
                     if validated_id == st.session_state.get('client_id') and st.session_state.get('portfolio_loaded'):
                          logger.debug(f"Client {validated_id} data already loaded. Skipping reload.")
                          st.toast(f"Client {validated_id} already loaded.", icon="✅")
                     else:
                          # Reset state for new load attempt
                          st.session_state.client_id = validated_id
                          st.session_state.client_context = None
                          st.session_state.portfolio_loaded = False
                          logger.info(f"Load button clicked for Client ID: {validated_id}")
                          # Call cached function (handles DB call + processing)
                          portfolio = get_client_portfolio_cached(validated_id)
                          if portfolio:
                              st.session_state.client_context = portfolio
                              st.session_state.portfolio_loaded = True
                              st.session_state.load_error = None # Clear error on success
                              logger.info(f"Portfolio context loaded successfully via button for {validated_id}")
                              st.toast(f"Loaded data for {validated_id}", icon="✅")
                          else:
                              # Handle case where cache function returned None (error or no data)
                              st.session_state.load_error = f"Failed to load or process data for '{validated_id}'. Check ID or logs."
                              st.session_state.portfolio_loaded = False
                              st.session_state.client_context = None
                              st.session_state.client_id = None # Reset client ID if load failed
                              logger.warning(st.session_state.load_error)
                          st.rerun() # Rerun to update sidebar display
                 else: # Invalid ID format from validate_client_id
                      st.session_state.load_error = f"Invalid Client ID format: '{client_id_input}'. Please use expected format (e.g., CLIENT123)."
                      st.session_state.portfolio_loaded = False
                      st.session_state.client_context = None
                      st.session_state.client_id = None # Reset client ID on invalid format
                      logger.warning(st.session_state.load_error)
                      st.rerun() # Rerun to show error

        # --- Display Logic ---
        # Display error messages if they exist
        if st.session_state.load_error:
            st.sidebar.error(st.session_state.load_error, icon="⚠️")

        # Display portfolio only if loaded successfully
        if st.session_state.portfolio_loaded and st.session_state.client_context:
            # Display summary and holdings
            display_portfolio_summary(st.session_state.client_context)
            display_holdings_list(st.session_state.client_context.get('holdings', []))
        elif st.session_state.client_id_input and not st.session_state.portfolio_loaded and not st.session_state.load_error:
             # If ID is entered, but not loaded and no error shown yet, prompt to load
             # This state might be brief due to reruns, but can be helpful
             st.sidebar.info("Click 'Load Client Data' to fetch portfolio.", icon="⬆️")


# --- Main Chat Interface ---
def main_chat_interface(agents: FinancialAgents):
    """Handles the main chat interaction area."""
    # Init chat history in session state if not present
    st.session_state.setdefault('conversation', [{"role": "assistant", "content": "Hello! How can I assist you today?"}])

    # Display past messages
    for message in st.session_state.conversation:
        with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
            # Use unsafe_allow_html=True carefully, ensure agent output is sanitized
            # or markdown rendering handles potential issues.
            st.markdown(message["content"], unsafe_allow_html=True)

    # Handle new user input via chat_input
    if prompt := st.chat_input("Ask a financial question..."):
        # Add user message to chat history immediately
        st.session_state.conversation.append({"role": "user", "content": prompt})
        # Rerun to display the user message right away
        st.rerun()

    # Check if the last message was from the user, if so, get agent response
    if st.session_state.conversation[-1]["role"] == "user":
        last_user_prompt = st.session_state.conversation[-1]["content"]

        # Validate the user's prompt before sending to agent
        if not validate_query_text(last_user_prompt):
            error_msg = "Your question seems too short, too long, or invalid. Please provide a more specific financial question (3-1500 characters)."
            st.warning(error_msg, icon="⚠️")
            # Add a warning message to the chat history
            st.session_state.conversation.append({"role": "assistant", "content": generate_warning_html("Invalid Query", error_msg)})
            # No rerun here, warning appears on the next interaction or manual refresh
        else:
            # Display thinking indicator and call agent
            with st.chat_message("assistant", avatar="🤖"):
                message_placeholder = st.empty()
                message_placeholder.markdown("Thinking... <span style='opacity: 0.6;'>🧠</span>", unsafe_allow_html=True)
                try:
                    client_context = st.session_state.get('client_context')
                    client_id = st.session_state.get('client_id') # Get loaded client ID
                    log_prefix = f"Client {client_id}" if client_id else "Generic"
                    logger.info(f"{log_prefix}: Calling agent.get_response for query: '{last_user_prompt[:50]}...'")

                    # Call agent function
                    response = agents.get_response(query=last_user_prompt, client_id=client_id, client_context=client_context)

                    # Display response (could be success or formatted error HTML)
                    message_placeholder.markdown(response, unsafe_allow_html=True)
                    # Add agent's response to chat history
                    st.session_state.conversation.append({"role": "assistant", "content": response})

                    # Log successful interaction to DB (if client_id exists and response wasn't an error)
                    is_error_response = '<div class="error-message">' in response or '<div class="token-warning">' in response # Check for error/warning HTML
                    if client_id and not is_error_response:
                        try:
                            # *** FIXED: Remove 'with' statement ***
                            db_client = AzurePostgresClient() # Instantiate
                            db_client.log_client_query(client_id=client_id, query=last_user_prompt[:1000], response=response[:2000])
                            # If db_client requires manual closing:
                            # if hasattr(db_client, 'close'): db_client.close()
                            logger.info(f"Logged query for client {client_id}")
                        except Exception as db_error:
                            # Log DB error but don't crash the app
                            logger.error(f"DB log FAILED for {client_id}: {db_error}", exc_info=True)
                            # Optionally inform user via toast or small note if critical
                            # st.toast("Warning: Could not log interaction.", icon="💾")
                    elif is_error_response:
                         logger.warning(f"Agent returned an error/warning response for {log_prefix}, not logging to DB.")

                except Exception as agent_call_error:
                    # Catch errors during the agent call itself
                    logger.error(f"Critical error calling FinancialAgents.get_response: {agent_call_error}", exc_info=True)
                    error_html = generate_error_html("Failed to Process Request", f"An unexpected error occurred while contacting the AI advisor: {agent_call_error}")
                    message_placeholder.markdown(error_html, unsafe_allow_html=True)
                    # Add the error message to history as well
                    st.session_state.conversation.append({"role": "assistant", "content": error_html})

            # Rerun *after* processing the agent response to update the chat display fully
            st.rerun()


# --- Application Entry Point ---
def main():
    """Main function to orchestrate the Streamlit application."""
    setup_ui()

    # --- Header ---
    st.title(APP_TITLE)
    st.markdown("""<p style="font-size: 1.1rem; color: #cbd5e1; margin-top:-10px; margin-bottom: 20px;">
                Your AI partner for financial insights and portfolio analysis.</p>
                """, unsafe_allow_html=True)

    # --- Agent Initialization (Cached Resource) ---
    @st.cache_resource(show_spinner="Initializing AI Advisor...")
    def initialize_financial_agents() -> Optional[FinancialAgents]:
        """Initializes the FinancialAgents system once per session."""
        local_logger = get_logger("AgentInitialization")
        local_logger.info("Attempting to initialize FinancialAgents...")
        try:
            agents_instance = FinancialAgents()
            local_logger.info("FinancialAgents initialized successfully.")
            return agents_instance
        except Exception as e:
            local_logger.critical(f"CRITICAL: FinancialAgents initialization failed: {e}", exc_info=True)
            # Display error directly in main area if init fails
            # Use the generate_error_html util for consistency
            st.error(generate_error_html("System Initialization Failed!", f"AI agents could not load. Please check logs or contact support. Error: {e}"), icon="🚨")
            return None # Return None to indicate failure

    financial_agents = initialize_financial_agents()

    # --- Sidebar Manager ---
    client_sidebar_manager() # Manages client loading and portfolio display in sidebar

    # --- Main Content Area (Chat Interface) ---
    if financial_agents:
        main_chat_interface(financial_agents) # Display chat interface if agents loaded
    else:
         # Error message already shown by initialize_financial_agents via st.error
         # Optionally add a less intrusive warning in the main area too
         st.warning("🔴 AI Advisor functionality is currently offline due to initialization failure.", icon="⚠️")

# --- Final Application Exception Handler ---
if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass # Allow clean exit from st.stop()
    except Exception as e:
        # Log critical unhandled errors thoroughly
        critical_logger = get_logger("MainCriticalError")
        error_type = type(e).__name__
        error_message = str(e)
        full_traceback = traceback.format_exc()
        critical_logger.critical(f"Application encountered CRITICAL unhandled error: {error_type} - {error_message}", exc_info=True)
        critical_logger.critical(full_traceback)

        # Display a user-friendly error in the UI as a last resort
        try:
            # Use the util function for consistent error display
            st.error(generate_error_html("Critical Application Error!", f"A critical error occurred: {error_type}. Please check the application logs or contact support."), icon="💥")
        except Exception as st_err:
            # If even displaying the Streamlit error fails, print to console/stderr
            critical_logger.critical(f"!!! FAILED to display critical error via st.error: {st_err}", exc_info=True)
            # Print crucial info to stderr as a final fallback
            print(f"\n--- CRITICAL UNHANDLED ERROR ---", file=sys.stderr)
            print(f"Timestamp: {datetime.now()}", file=sys.stderr)
            print(f"Original Error: {error_type} - {error_message}", file=sys.stderr)
            print(f"Traceback:\n{full_traceback}", file=sys.stderr)
            print(f"Error during st.error display: {type(st_err).__name__} - {st_err}", file=sys.stderr)
            print(f"--- END CRITICAL ERROR ---\n", file=sys.stderr)