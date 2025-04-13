# src/fintech_ai_bot/ui/sidebar.py
# UPDATED TO SHOW HORIZONTAL ALLOCATION BARS (DARK THEME)

import streamlit as st
from typing import Optional, Dict, Any
import pandas as pd # Added import
import plotly.express as px # Added import
from fintech_ai_bot.config import settings
from fintech_ai_bot.db.postgres_client import PostgresClient # Need DB client for fetching
from fintech_ai_bot.utils import get_logger, validate_client_id, validate_portfolio_data # Added validate_portfolio_data
# Import necessary for the processing step - consider moving processing logic
from fintech_ai_bot.core.orchestrator import AgentOrchestrator


logger = get_logger(__name__)

# --- Database Interaction & Caching ---
# get_client_portfolio_cached function remains the same as before
@st.cache_data(ttl=300, show_spinner="Fetching client portfolio...")
def get_client_portfolio_cached(client_id: str, _db_client: PostgresClient) -> Optional[Dict]:
    """ Cached wrapper for fetching and processing portfolio data. """
    # ... (keep the existing implementation of this function) ...
    validated_id = validate_client_id(client_id)
    if not validated_id:
        logger.warning(f"Invalid client ID format '{client_id}' blocked before cache fetch.")
        return None

    logger.info(f"Cache check/fetch for client ID: {validated_id}")
    try:
        raw_portfolio = _db_client.get_client_portfolio(validated_id)
        if not raw_portfolio:
            logger.warning(f"No portfolio data returned from DB for {validated_id}")
            return None
        if not isinstance(raw_portfolio, dict):
            logger.error(f"Invalid data type received from DB for {validated_id}: {type(raw_portfolio)}")
            return None

        logger.info(f"Raw portfolio data retrieved from DB for {validated_id}, processing...")
        try:
            # TEMPORARY WORKAROUND (from original code) - **NEEDS VERIFICATION/REFACTORING**
            processed_portfolio = AgentOrchestrator._process_db_portfolio(None, validated_id, raw_portfolio)

        except AttributeError:
             logger.error("Cannot call _process_db_portfolio statically. Refactoring needed.")
             # Fallback: Return raw data with basic structure
             processed_portfolio = {
                 "id": validated_id,
                 "name": f"Client {validated_id[-4:]}" if len(validated_id) >= 4 else f"Client {validated_id}",
                 "risk_profile": raw_portfolio.get('risk_profile', 'Not specified') or 'Not specified',
                 "portfolio_value": float(raw_portfolio.get('total_value', 0.0)),
                 "holdings": raw_portfolio.get('holdings', [])
             }
             if not validate_portfolio_data(processed_portfolio):
                  logger.error(f"Basic processed portfolio failed validation for {validated_id}")
                  return None

        except Exception as proc_e:
            logger.error(f"Error during portfolio processing for {validated_id}: {proc_e}", exc_info=True)
            return None

        if processed_portfolio:
            logger.info(f"Successfully processed/validated portfolio for {validated_id}. Caching result.")
            return processed_portfolio
        else:
            logger.error(f"Processed DB portfolio failed validation/processing for {validated_id}. Caching None.")
            return None
    except Exception as e:
        logger.error(f"Database or processing error during cache fetch for {validated_id}: {e}", exc_info=True)
        return None


# --- UI Component: Portfolio Summary ---
# (Remains the same as the previous version - standard components)
def display_portfolio_summary(portfolio_data: Dict):
    """ Displays portfolio summary using standard st.metric in the sidebar. """
    try:
        total_value = portfolio_data.get('portfolio_value', 0.0)
        risk_profile_raw = portfolio_data.get('risk_profile')
        risk_profile = risk_profile_raw.capitalize() if risk_profile_raw and isinstance(risk_profile_raw, str) else 'N/A'

        st.subheader("Portfolio Snapshot")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Total Value", value=f"${total_value:,.2f}")
        with col2:
            st.metric(label="Risk Profile", value=risk_profile)
    except Exception as e:
        logger.error(f"Error displaying portfolio summary: {e}", exc_info=True)
        st.warning("Could not display summary.", icon="⚠️")


# --- UI Component: Allocation Bars ---
# (MODIFIED function replacing display_allocation_chart)
def display_allocation_bars(holdings_data: list, client_name: str):
    """ Displays asset allocation as horizontal bars in the sidebar (dark theme). """
    if not holdings_data:
        st.info("No holdings data available to display chart.")
        return
    try:
        st.subheader("Asset Allocation (%)") # Update title

        # Prepare data for Plotly
        df_holdings = pd.DataFrame(holdings_data)

        # --- Validate Allocation Data ---
        if 'allocation' not in df_holdings.columns:
            # Fallback: Attempt to calculate if 'value' and total_value exist
            logger.warning("Holdings data missing 'allocation' column. Attempting calculation.")
            if 'value' in df_holdings.columns and 'portfolio_value' in st.session_state.get('client_context', {}):
                total_value = st.session_state.client_context['portfolio_value']
                if total_value > 0:
                     df_holdings['value'] = pd.to_numeric(df_holdings['value'], errors='coerce')
                     df_holdings.dropna(subset=['value'], inplace=True)
                     df_holdings['allocation'] = (df_holdings['value'] / total_value) * 100
                     logger.info("Calculated allocation based on holding values and total portfolio value.")
                else:
                    logger.error("Cannot calculate allocation: Total portfolio value is zero or missing.")
                    st.warning("Cannot calculate allocation: Total portfolio value is zero.", icon="⚠️")
                    return
            else:
                logger.error("Cannot calculate or find allocation: Missing 'value' or total portfolio value.")
                st.warning("Cannot display allocation chart: Missing data.", icon="⚠️")
                return
        else:
            # Ensure 'allocation' is numeric if it exists
            df_holdings['allocation'] = pd.to_numeric(df_holdings['allocation'], errors='coerce')

        # --- Validate Symbol and Filter ---
        if 'symbol' not in df_holdings.columns:
            logger.warning("Holdings data missing 'symbol' column for chart.")
            st.warning("Cannot display chart: Missing 'symbol' data.", icon="⚠️")
            return

        df_holdings.dropna(subset=['allocation', 'symbol'], inplace=True)
        df_holdings = df_holdings[df_holdings['allocation'] > 0] # Keep only positive allocations

        if df_holdings.empty:
            st.info(f"No holdings with positive allocation to display in chart for {client_name}.")
            return

        # Sort by allocation for better visualization
        df_holdings = df_holdings.sort_values(by='allocation', ascending=True) # Ascending for horizontal bar chart y-axis

        # Create Horizontal Bar Chart
        fig = px.bar(
            df_holdings,
            x='allocation',
            y='symbol',
            orientation='h', # Horizontal bars
            template='plotly_dark', # Explicitly set dark theme
            text='allocation', # Show allocation value on bars
            # title="Allocation by Holding (%)", # Title is now subheader
            labels={'allocation': '% Allocation', 'symbol': 'Holding Symbol'} # Axis labels
        )

        # Customize layout and text for dark theme & sidebar
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10), # Adjust margins
            yaxis={'categoryorder': 'total ascending', 'title': None}, # Ensure correct order, remove y-axis title
            xaxis_title="% Allocation",
            plot_bgcolor='rgba(0,0,0,0)', # Transparent background
            paper_bgcolor='rgba(0,0,0,0)', # Transparent background
            font=dict(color='white') # Ensure font is visible
        )
        fig.update_traces(
            texttemplate='%{text:.1f}%', # Format text as percentage
            textposition='outside', # Position text outside the bar for clarity
            marker_color='#636EFA' # Example primary color, adjust as needed
        )
        # Ensure x-axis labels and grid are visible on dark theme
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#444')

        # Display the chart
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        logger.error(f"Error displaying allocation bars: {e}", exc_info=True)
        st.warning("Could not display allocation chart.", icon="⚠️")


# --- Sidebar Manager ---
# (Updated to call display_allocation_bars)
def manage_sidebar(db_client: PostgresClient):
    """ Manages client ID input, loading, and portfolio display calls in the sidebar. """
    with st.sidebar:
        st.markdown("## Client Access")
        # Initialize session state keys (same as before)
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
        if client_id_input != st.session_state.client_id_input:
             st.session_state.client_id_input = client_id_input

        load_button = st.button("Load Client Data", key="load_client_button", use_container_width=True)

        if load_button:
            # ... (keep the existing logic for loading data inside the button block) ...
            st.session_state.load_error = None # Reset error on new attempt
            if not client_id_input:
                st.session_state.load_error = "Please enter a Client ID."
                st.session_state.portfolio_loaded = False
                st.session_state.client_context = None
                st.session_state.client_id = None
            else:
                validated_id = validate_client_id(client_id_input)
                if validated_id:
                    logger.info(f"Load button clicked for Client ID: {validated_id}. Triggering data fetch.")
                    st.session_state.client_id = validated_id
                    st.session_state.client_context = None # Clear old context before fetching
                    st.session_state.portfolio_loaded = False

                    # Call the cached function
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
                        logger.warning(st.session_state.load_error + f" (Raw ID: {client_id_input})")

                else:
                    st.session_state.load_error = f"Invalid Client ID format: '{client_id_input}'. Please check."
                    st.session_state.portfolio_loaded = False
                    st.session_state.client_context = None
                    st.session_state.client_id = None # Clear ID if format is invalid
                    logger.warning(st.session_state.load_error)
            st.rerun()

        # --- Display Logic ---
        if st.session_state.load_error:
            st.warning(st.session_state.load_error, icon="⚠️")

        if st.session_state.portfolio_loaded and st.session_state.client_context:
            client_name = st.session_state.client_context.get('name', st.session_state.client_id)
            st.divider()
            display_portfolio_summary(st.session_state.client_context)
            st.divider()
            # Call the NEW bar chart function
            display_allocation_bars(st.session_state.client_context.get('holdings', []), client_name)
        elif client_id_input and not st.session_state.portfolio_loaded and not st.session_state.load_error:
            st.info("Click 'Load Client Data' above.", icon="⬆️")
        elif not client_id_input and not st.session_state.client_id:
             st.info("Enter a Client ID to load data.", icon="🆔")
        elif st.session_state.client_id and not st.session_state.portfolio_loaded and st.session_state.load_error:
             st.info(f"Could not display portfolio for {st.session_state.client_id}.", icon="❌")