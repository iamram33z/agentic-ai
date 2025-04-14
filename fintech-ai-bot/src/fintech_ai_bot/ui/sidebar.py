# src/fintech_ai_bot/ui/sidebar.py
# ENHANCED UI/UX: Added st.form, dividers, theme-adaptive chart
# + TEMPORARY DEBUGGING for client name issue

import streamlit as st
from typing import Optional, Dict, Any
import pandas as pd
import plotly.express as px
from fintech_ai_bot.config import settings
from fintech_ai_bot.db.postgres_client import PostgresClient
from fintech_ai_bot.utils import get_logger, validate_client_id, validate_portfolio_data
from fintech_ai_bot.core.orchestrator import AgentOrchestrator # Still needed for processing logic temporarily

logger = get_logger(__name__)

# --- Database Interaction & Caching ---
# --- Includes DEBUG steps ---
@st.cache_data(ttl=settings.cache_ttl_seconds, show_spinner="Fetching client portfolio...") # Use TTL from settings
def get_client_portfolio_cached(client_id: str, _db_client: PostgresClient) -> Optional[Dict]:
    """ Cached wrapper for fetching and processing portfolio data. """
    validated_id = validate_client_id(client_id)
    if not validated_id:
        logger.warning(f"Invalid client ID format '{client_id}' blocked before cache fetch.")
        return None

    logger.info(f"Cache check/fetch for client ID: {validated_id}")
    try:
        # Fetch raw data from DB (should include 'name' now)
        raw_portfolio = _db_client.get_client_portfolio(validated_id)

        # --- DEBUG STEP 1: Check raw_portfolio ---
        st.sidebar.write("--- DEBUG: Raw Portfolio from DB ---")
        st.sidebar.json(raw_portfolio or {"error": "No raw data returned"})
        # --- END DEBUG STEP 1 ---

        if not raw_portfolio:
            logger.warning(f"No portfolio data returned from DB for {validated_id}")
            return None # Explicitly return None if no data
        if not isinstance(raw_portfolio, dict):
            logger.error(f"Invalid data type received from DB for {validated_id}: {type(raw_portfolio)}")
            return None

        logger.info(f"Raw portfolio data retrieved from DB for {validated_id}, processing...")
        processed_portfolio = None # Initialize before try block
        try:
            # !! This is the likely suspect !!
            # !! IMPORTANT: Calling a private method of another class like this is poor practice.
            # This processing logic should ideally be moved to a shared utility or service layer.
            processed_portfolio = AgentOrchestrator._process_db_portfolio(None, validated_id, raw_portfolio)

        except AttributeError:
             logger.error("Cannot call _process_db_portfolio statically. Refactoring needed.")
             # Basic fallback structure if processing fails
             processed_portfolio = {
                 "id": validated_id,
                 "name": f"Client {validated_id[-4:]}" if len(validated_id) >= 4 else f"Client {validated_id}", # Fallback name
                 "risk_profile": raw_portfolio.get('risk_profile', 'Not specified') or 'Not specified',
                 "portfolio_value": float(raw_portfolio.get('total_value', 0.0)),
                 "holdings": raw_portfolio.get('holdings', [])
             }
             # Even if fallback works, validate it
             if not validate_portfolio_data(processed_portfolio):
                  logger.error(f"Basic processed portfolio fallback failed validation for {validated_id}")
                  return None # Return None if fallback is invalid

        except Exception as proc_e:
            logger.error(f"Error during portfolio processing step for {validated_id}: {proc_e}", exc_info=True)
            return None # Return None if processing fails

        # --- DEBUG STEP 2: Check processed_portfolio ---
        st.sidebar.write("--- DEBUG: Processed Portfolio ---")
        st.sidebar.json(processed_portfolio or {"error": "Processing failed or returned None"})
        # --- END DEBUG STEP 2 ---

        # Final validation after processing
        if processed_portfolio and validate_portfolio_data(processed_portfolio):
            logger.info(f"Successfully processed/validated portfolio for {validated_id}. Caching result.")
            return processed_portfolio
        else:
            # Log if validation fails *after* processing seemed okay
            if processed_portfolio:
                 logger.error(f"Processed portfolio failed validation for {validated_id}. Processed data: {processed_portfolio}")
            else:
                 logger.error(f"Processing resulted in None or empty data for {validated_id}, cannot validate.")
            return None # Cache None if validation fails

    except Exception as e:
        logger.error(f"Database error during cache fetch/processing for {validated_id}: {e}", exc_info=True)
        # Don't display UI error here, let the calling function handle it based on return value
        return None

# --- UI Component: Portfolio Summary ---
def display_portfolio_summary(portfolio_data: Dict):
    """ Displays portfolio summary using st.metric in the sidebar. """
    try:
        total_value = portfolio_data.get('portfolio_value', 0.0)
        risk_profile_raw = portfolio_data.get('risk_profile')
        risk_profile = risk_profile_raw.capitalize() if risk_profile_raw and isinstance(risk_profile_raw, str) else 'N/A'
        client_name = portfolio_data.get('name', 'Client') # Get name for context

        st.subheader(f"Portfolio Summary of {client_name}") # More specific title
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Total Value", value=f"${total_value:,.2f}")
        with col2:
            st.metric(label="Risk Profile", value=risk_profile)
    except Exception as e:
        logger.error(f"Error displaying portfolio summary: {e}", exc_info=True)
        st.warning("Could not display summary.", icon="⚠️")


# --- UI Component: Allocation Bars (Theme Adaptive) ---
def display_allocation_bars(holdings_data: list, client_name: str):
    """ Displays asset allocation as horizontal bars, adapting to Streamlit theme. """
    if not holdings_data:
        st.info("No holdings data available to display chart.")
        return
    try:
        st.subheader("Asset Allocation (%)")

        df_holdings = pd.DataFrame(holdings_data)

        # --- Allocation Data Validation/Calculation ---
        if 'allocation' not in df_holdings.columns:
            logger.warning("Holdings data missing 'allocation' column. Attempting calculation from value.")
            # Ensure client_context exists before accessing it
            if 'value' in df_holdings.columns and 'client_context' in st.session_state and st.session_state.client_context:
                total_value = st.session_state.client_context.get('portfolio_value', 0.0)
                if total_value > 0:
                    df_holdings['value'] = pd.to_numeric(df_holdings['value'], errors='coerce')
                    df_holdings.dropna(subset=['value'], inplace=True) # Drop rows where value couldn't be converted
                    df_holdings['allocation'] = (df_holdings['value'] / total_value) * 100
                    logger.info("Calculated allocation based on holding values.")
                else:
                    logger.warning("Cannot calculate allocation: Total portfolio value is zero or missing.")
                    st.warning("Cannot calculate allocation: Total portfolio value is zero.", icon="⚠️")
                    return
            else:
                logger.error("Cannot calculate or find allocation: Missing 'value' column or client context.")
                st.warning("Cannot display allocation chart: Missing required data.", icon="⚠️")
                return
        else:
            df_holdings['allocation'] = pd.to_numeric(df_holdings['allocation'], errors='coerce')

        # --- Symbol Validation and Filtering ---
        if 'symbol' not in df_holdings.columns:
            logger.warning("Holdings data missing 'symbol' column for chart labels.")
            st.warning("Cannot display chart: Missing 'symbol' data.", icon="⚠️")
            return

        df_holdings.dropna(subset=['allocation', 'symbol'], inplace=True)
        df_holdings = df_holdings[df_holdings['allocation'] > 0.01] # Filter small/zero allocations for clarity

        if df_holdings.empty:
            st.info(f"No holdings with significant allocation to display for {client_name}.")
            return

        # Sort by allocation for better visualization (ascending for horizontal bars)
        df_holdings = df_holdings.sort_values(by='allocation', ascending=True)

        # --- Create Horizontal Bar Chart (Theme Adaptive) ---
        fig = px.bar(
            df_holdings,
            x='allocation',
            y='symbol',
            orientation='h',
            text='allocation', # Show value on bars
            labels={'allocation': '% Allocation', 'symbol': 'Holding'}, # Clearer axis labels
            # No explicit template - let Plotly adapt to Streamlit theme
        )

        # --- Customize Layout ---
        fig.update_layout(
            margin=dict(l=5, r=20, t=10, b=10), # Fine-tune margins for sidebar
            yaxis={'categoryorder': 'total ascending', 'title': None}, # Remove y-axis title
            xaxis_title="% Allocation",
            # Remove explicit bg/paper colors - let theme handle it
            # plot_bgcolor='rgba(0,0,0,0)',
            # paper_bgcolor='rgba(0,0,0,0)',
            # Remove explicit font color - let theme handle it
            # font=dict(color='white'),
            height=min(400, 30 + len(df_holdings) * 35), # Dynamic height based on number of bars
            xaxis=dict(showgrid=True) # Ensure grid is shown for readability
        )
        fig.update_traces(
            texttemplate='%{text:.1f}%', # Format text display
            textposition='outside', # Position text outside bars
            # Use Plotly's default color sequence, adaptable to theme
            # marker_color='#636EFA' # Removed fixed color
        )
        # No need to update xaxes gridcolor explicitly, theme should handle contrast

        st.plotly_chart(fig, use_container_width=True) # Display chart

    except ValueError as ve:
        logger.error(f"Data conversion error displaying allocation bars: {ve}", exc_info=True)
        st.warning("Could not display allocation chart due to data format issues.", icon="⚠️")
    except Exception as e:
        logger.error(f"Error displaying allocation bars: {e}", exc_info=True)
        st.warning("Could not display allocation chart.", icon="⚠️")


# --- Sidebar Manager ---
def manage_sidebar(db_client: PostgresClient):
    """ Manages client ID input, loading, and portfolio display in the sidebar. """
    with st.sidebar:
        st.markdown("## Client Access")

        # Initialize session state keys if they don't exist
        st.session_state.setdefault('client_id_input', "")
        st.session_state.setdefault('client_id', None) # The validated ID currently loaded
        st.session_state.setdefault('client_context', None) # The loaded portfolio data
        st.session_state.setdefault('portfolio_loaded', False)
        st.session_state.setdefault('load_error', None) # Store specific load errors

        # Use a form for better grouping of input and button
        with st.form("client_form"):
            client_id_input = st.text_input(
                "Client ID",
                value=st.session_state.client_id_input,
                placeholder="Enter Client ID (e.g., CLIENT123)",
                help="Enter the client identifier to load their portfolio.",
                key="client_id_form_input", # Use different key if needed inside form
                label_visibility="collapsed" # Hide label if placeholder is clear
            )
            submitted = st.form_submit_button("Load Client Data", use_container_width=True)

            if submitted:
                # Update session state immediately with the input value for persistence
                st.session_state.client_id_input = client_id_input
                st.session_state.load_error = None # Reset error on new submission
                st.session_state.portfolio_loaded = False # Reset load status
                st.session_state.client_context = None # Clear previous context
                st.session_state.client_id = None # Clear previous validated ID

                if not client_id_input:
                    st.session_state.load_error = "Please enter a Client ID."
                else:
                    validated_id = validate_client_id(client_id_input)
                    if validated_id:
                        logger.info(f"Form submitted for Client ID: {validated_id}. Triggering data fetch.")
                        st.session_state.client_id = validated_id # Store validated ID

                        # Call the cached function (spinner shown automatically)
                        # This now includes the DEBUG prints inside it
                        portfolio = get_client_portfolio_cached(validated_id, db_client)

                        if portfolio:
                            # Successfully loaded and processed
                            st.session_state.client_context = portfolio
                            st.session_state.portfolio_loaded = True
                            client_name = portfolio.get('name', validated_id) # Use name from processed portfolio
                            logger.info(f"Portfolio context loaded successfully for {client_name} ({validated_id})")
                            st.toast(f"Loaded data for {client_name}", icon="✅")
                            # No rerun needed here, form submission handles it
                        else:
                            # Failed to load or process (error logged in cached function)
                            st.session_state.load_error = f"Failed to load/process data for '{validated_id}'. Verify ID or check system logs."
                            logger.warning(st.session_state.load_error + f" (Raw Input: {client_id_input})")

                    else:
                        # Invalid ID format
                        st.session_state.load_error = f"Invalid Client ID format: '{client_id_input}'. Please check and try again."
                        logger.warning(st.session_state.load_error)

                # Rerun needed after form submission logic to reflect state changes outside the form
                st.rerun()

        # --- Display Logic (outside the form) ---
        if st.session_state.load_error:
            st.warning(st.session_state.load_error, icon="⚠️") # Show load errors prominently

        if st.session_state.portfolio_loaded and st.session_state.client_context:
            # Display portfolio details if loaded successfully
            # Retrieves name from the potentially modified client_context
            client_name = st.session_state.client_context.get('name', st.session_state.client_id)
            st.divider() # Separator before summary
            display_portfolio_summary(st.session_state.client_context)
            st.divider() # Separator before allocation
            display_allocation_bars(st.session_state.client_context.get('holdings', []), client_name)
        elif st.session_state.client_id and not st.session_state.portfolio_loaded and st.session_state.load_error:
            # Case: Tried loading but failed
            st.info(f"Could not display portfolio for {st.session_state.client_id}. See warning above.", icon="❌")
        elif not st.session_state.client_id:
             # Case: Nothing entered or loaded yet
             st.info("Enter a Client ID above and click 'Load Client Data' to view portfolio details.", icon="🆔")