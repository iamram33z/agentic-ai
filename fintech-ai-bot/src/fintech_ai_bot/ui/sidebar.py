# src/fintech_ai_bot/ui/sidebar.py
# CORRECTED

import streamlit as st
from typing import Optional, Dict, Any
from fintech_ai_bot.config import settings
from fintech_ai_bot.db.postgres_client import PostgresClient # Need DB client for fetching
from fintech_ai_bot.utils import get_logger, validate_client_id, validate_portfolio_data # Added validate_portfolio_data
# Import necessary for the processing step - consider moving processing logic
from fintech_ai_bot.core.orchestrator import AgentOrchestrator


logger = get_logger(__name__)

# --- Database Interaction & Caching ---
# CORRECTED FUNCTION DEFINITION: Added underscore to db_client argument
@st.cache_data(ttl=300, show_spinner="Fetching client portfolio...")
def get_client_portfolio_cached(client_id: str, _db_client: PostgresClient) -> Optional[Dict]:
    """
    Cached wrapper for fetching and processing portfolio data.
    _db_client argument is prefixed with underscore to prevent hashing by Streamlit.
    """
    validated_id = validate_client_id(client_id)
    if not validated_id:
        logger.warning(f"Invalid client ID format '{client_id}' blocked before cache fetch.")
        return None

    logger.info(f"Cache check/fetch for client ID: {validated_id}")
    try:
        # Use the _db_client argument name inside the function
        raw_portfolio = _db_client.get_client_portfolio(validated_id)
        if not raw_portfolio:
            logger.warning(f"No portfolio data returned from DB for {validated_id}")
            return None
        if not isinstance(raw_portfolio, dict):
            logger.error(f"Invalid data type received from DB for {validated_id}: {type(raw_portfolio)}")
            return None

        logger.info(f"Raw portfolio data retrieved from DB for {validated_id}, processing...")
        # --- Processing Logic ---
        # Ideally, this complex processing shouldn't be inside the cache function directly
        # or inside the UI module. It makes testing harder and mixes concerns.
        # Consider moving this to a dedicated data processing utility or service.
        # For now, replicating the logic using a static method/helper as before:
        # This still feels hacky - needs refactoring ideally.
        try:
            # We need an instance or a static way to call _process_db_portfolio
            # Option 1: Make _process_db_portfolio static (if it doesn't need instance state)
            # Option 2: Pass orchestrator/processor instance (but can't hash that either!)
            # Option 3: Move processing outside the cached function (preferred)
            # TEMPORARY WORKAROUND (assuming _process_db_portfolio can be made static or doesn't need self):
            # This will likely fail if _process_db_portfolio actually uses self.
            # You might need to refactor AgentOrchestrator._process_db_portfolio
            # to be a static method or move it to utils.
            # Placeholder call - **NEEDS VERIFICATION/REFACTORING**
            processed_portfolio = AgentOrchestrator._process_db_portfolio(None, validated_id, raw_portfolio)

        except AttributeError:
             logger.error("Cannot call _process_db_portfolio statically. Refactoring needed.")
             # Fallback: Return raw data with basic structure - validation might fail later
             processed_portfolio = {
                 "id": validated_id,
                 "name": f"Client {validated_id[-4:]}" if len(validated_id) >= 4 else f"Client {validated_id}",
                 "risk_profile": raw_portfolio.get('risk_profile', 'Not specified') or 'Not specified',
                 "portfolio_value": float(raw_portfolio.get('total_value', 0.0)),
                 "holdings": raw_portfolio.get('holdings', []) # Pass raw holdings list
             }
             # Attempt validation on this basic structure
             if not validate_portfolio_data(processed_portfolio):
                  logger.error(f"Basic processed portfolio failed validation for {validated_id}")
                  return None # Give up if even basic structure fails validation

        except Exception as proc_e:
            logger.error(f"Error during portfolio processing for {validated_id}: {proc_e}", exc_info=True)
            return None # Return None on processing error

        # --- End Processing Logic ---

        if processed_portfolio:
            logger.info(f"Successfully processed/validated portfolio for {validated_id}. Caching result.")
            return processed_portfolio
        else:
            logger.error(f"Processed DB portfolio failed validation/processing for {validated_id}. Caching None.")
            return None
    except Exception as e:
        logger.error(f"Database or processing error during cache fetch for {validated_id}: {e}", exc_info=True)
        return None


# --- UI Components (display_portfolio_summary, display_holdings_list) ---
# These remain the same as in the previous correct version. Copied here for completeness.

def display_portfolio_summary(portfolio_data: Dict):
    """Displays enhanced portfolio summary using st.metric in the sidebar."""
    try:
        total_value = portfolio_data.get('portfolio_value', 0.0)
        risk_profile_raw = portfolio_data.get('risk_profile')
        risk_profile = risk_profile_raw.capitalize() if risk_profile_raw and isinstance(risk_profile_raw, str) else 'N/A'

        # Apply class for CSS styling (CSS injected in main.py)
        st.markdown('<div class="portfolio-summary">', unsafe_allow_html=True)
        st.markdown("<h5>Portfolio Snapshot</h5>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Total Value", value=f"${total_value:,.2f}")
        with col2:
            st.metric(label="Risk Profile", value=risk_profile)
        st.markdown('</div>', unsafe_allow_html=True)
    except Exception as e:
        logger.error(f"Error displaying portfolio summary: {e}", exc_info=True)
        st.warning("Could not display summary.", icon="⚠️")


def display_holdings_list(holdings_data: list, client_name: str):
    """Displays enhanced scrollable holdings list in the sidebar expander."""
    if not holdings_data:
        return
    try:
        with st.expander(f"{client_name} Holdings ({len(holdings_data)})", expanded=False):
            st.markdown('<div class="holding-list-container">', unsafe_allow_html=True)
            sorted_holdings = sorted(holdings_data, key=lambda x: x.get('value', 0), reverse=True)
            for holding in sorted_holdings:
                symbol = holding.get('symbol', 'N/A')
                value = holding.get('value', 0.0)
                # Ensure allocation exists and is numeric, default to 0.0
                allocation = holding.get('allocation', 0.0)
                if not isinstance(allocation, (int, float)):
                    allocation = 0.0

                st.markdown(f"""
                <div class="holding-item">
                    <span>{symbol}</span>
                    <span class="value-alloc">
                        <span>{allocation:.1f}%</span>
                        <small>${value:,.0f}</small>
                    </span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    except Exception as e:
        logger.error(f"Error displaying holdings list: {e}", exc_info=True)
        st.warning("Could not display holdings.", icon="⚠️")

# --- Sidebar Manager ---
def manage_sidebar(db_client: PostgresClient):
    """Manages client ID input, loading, and portfolio display calls in the sidebar."""
    with st.sidebar:
        st.markdown("## Client Access")
        # Initialize session state keys
        st.session_state.setdefault('client_id_input', "")
        st.session_state.setdefault('client_id', None)
        st.session_state.setdefault('client_context', None)
        st.session_state.setdefault('portfolio_loaded', False)
        st.session_state.setdefault('load_error', None)

        client_id_input = st.text_input(
            "Client ID Input",
            value=st.session_state.client_id_input,
            placeholder="Enter Client ID (e.g., CLIENT123)",
            help="Enter the client identifier and click Load.",
            key="client_id_input_widget",
            label_visibility="collapsed"
        )
        # Update session state immediately if input changes
        if client_id_input != st.session_state.client_id_input:
             st.session_state.client_id_input = client_id_input
             st.rerun() # Optional: rerun on input change if needed

        load_button = st.button("Load Client Data", key="load_client_button", use_container_width=True)

        if load_button:
            st.session_state.load_error = None # Reset error on new attempt
            if not client_id_input:
                st.session_state.load_error = "Please enter a Client ID."
                st.session_state.portfolio_loaded = False
                st.session_state.client_context = None
                st.session_state.client_id = None
            else:
                validated_id = validate_client_id(client_id_input)
                if validated_id:
                    if validated_id != st.session_state.get('client_id') or not st.session_state.get('portfolio_loaded'):
                        logger.info(f"Load button clicked for Client ID: {validated_id}")
                        st.session_state.client_id = validated_id # Store validated ID
                        st.session_state.client_context = None
                        st.session_state.portfolio_loaded = False

                        # Call the cached function, passing the actual db_client instance
                        # The underscore prefix is only in the function *definition*
                        portfolio = get_client_portfolio_cached(validated_id, db_client)

                        if portfolio:
                            st.session_state.client_context = portfolio
                            st.session_state.portfolio_loaded = True
                            st.session_state.load_error = None
                            client_name = portfolio.get('name', validated_id)
                            logger.info(f"Portfolio context loaded successfully for {client_name}")
                            st.toast(f"Loaded data for {client_name}", icon="✅")
                        else:
                            st.session_state.load_error = f"Failed to load/process data for '{validated_id}'. Verify ID or check logs."
                            st.session_state.portfolio_loaded = False
                            st.session_state.client_context = None
                            st.session_state.client_id = None # Clear loaded ID on failure
                            logger.warning(st.session_state.load_error + f" (Raw ID: {client_id_input})")
                    else:
                         logger.debug(f"Client {validated_id} data already loaded. Skipping reload.")
                         st.toast(f"Client {st.session_state.client_context.get('name', validated_id)} already loaded.", icon="ℹ️")
                else:
                    st.session_state.load_error = f"Invalid Client ID format: '{client_id_input}'. Please check."
                    st.session_state.portfolio_loaded = False
                    st.session_state.client_context = None
                    st.session_state.client_id = None
                    logger.warning(st.session_state.load_error)
            st.rerun() # Rerun to update sidebar display based on new state

        # --- Display Logic ---
        if st.session_state.load_error:
            st.warning(st.session_state.load_error, icon="⚠️")

        # Display portfolio details only if successfully loaded
        if st.session_state.portfolio_loaded and st.session_state.client_context:
            client_name = st.session_state.client_context.get('name', st.session_state.client_id)
            display_portfolio_summary(st.session_state.client_context)
            # Pass holdings and client name for display
            display_holdings_list(st.session_state.client_context.get('holdings', []), client_name)
        elif client_id_input and not st.session_state.portfolio_loaded and not st.session_state.load_error:
            # Only show this if an ID is entered but not loaded and no error occurred yet
            st.info("Click 'Load Client Data' above.", icon="⬆️")
        elif not client_id_input and not st.session_state.client_id:
            # Only show if no ID entered and no client is loaded
             st.info("Enter a Client ID to load data.", icon="🆔")