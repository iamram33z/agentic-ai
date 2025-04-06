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
from typing import Optional, Dict  # Added Dict type hint

# Third-party imports
from dotenv import load_dotenv

# Local application imports
# Ensure these modules are accessible (e.g., in the same directory or added to PYTHONPATH)
try:
    from agent import FinancialAgents
    from azure_postgres import AzurePostgresClient  # Assuming class exists
    # process_raw_portfolio_data moved into main.py
    from utils import (
        get_logger,
        generate_error_html,
        generate_warning_html,
        validate_query_text,
        validate_portfolio_data,  # Keep validation util
        validate_client_id,
        log_execution_time,  # Keep if used elsewhere or in utils
        # Add any other utils ACTUALLY used by main
    )
except ImportError as e:
    # If imports fail, show an error message and stop
    st.error(
        f"Failed to import necessary modules: {e}. Please ensure agent.py, utils.py, and azure_postgres.py are available.",
        icon="🚨")
    st.stop()

# --- Load Environment Variables & Initialize Logger ---
load_dotenv()
logger = get_logger("StreamlitApp")  # Use the configured logger from utils

# --- Constants ---
APP_TITLE = "FinTech AI Advisor"
APP_ICON = "💹"
USER_AVATAR = "👤"
ASSISTANT_AVATAR = "🤖"

# --- Modern Dark Theme Color Scheme ---
COLOR_PRIMARY = "#6366f1"  # Vibrant indigo (softer than previous)
COLOR_SECONDARY = "#8b5cf6"  # Purple (accent color)
COLOR_ACCENT = "#ec4899"  # Pink accent for highlights
COLOR_SUCCESS = "#10b981"  # Emerald green
COLOR_WARNING = "#f59e0b"  # Amber
COLOR_DANGER = "#ef4444"  # Red
COLOR_INFO = "#3b82f6"  # Blue for information

COLOR_BACKGROUND = "#0f172a"  # Dark navy blue (deep background)
COLOR_SURFACE = "#1e293b"  # Dark slate blue (card surfaces)
COLOR_SURFACE_LIGHT = "#334155"  # Lighter slate (for hover states)
COLOR_BORDER = "#475569"  # Medium slate (borders)
COLOR_BORDER_LIGHT = "#64748b"  # Lighter border

COLOR_TEXT_PRIMARY = "#f8fafc"  # Bright white (high contrast)
COLOR_TEXT_SECONDARY = "#e2e8f0"  # Off-white (secondary text)
COLOR_TEXT_MUTED = "#94a3b8"  # Muted slate (tertiary text)
COLOR_TEXT_DARK = "#1e293b"  # Dark text for light backgrounds

# Gradient colors for special elements
GRADIENT_PRIMARY = "linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%)"
GRADIENT_SECONDARY = "linear-gradient(135deg, #1e293b 0%, #0f172a 100%)"


# --- UI Setup & Styling (Enhanced Version) ---
def setup_ui():
    """Configure Streamlit page settings and inject enhanced custom CSS."""
    try:
        st.set_page_config(
            page_title=APP_TITLE,
            layout="wide",
            page_icon=APP_ICON,
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'mailto:support@example.com',
                'Report a bug': "mailto:support@example.com",
                'About': f"### {APP_TITLE}\nYour AI-powered financial guidance."
            }
        )

        # --- Modern Dark Theme CSS ---
        st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        /* --- Base & General Styling --- */
        body, .stApp, input, textarea, button, select, .stMarkdown {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }}

        .stApp {{
            background-color: {COLOR_BACKGROUND};
            color: {COLOR_TEXT_SECONDARY};
        }}

        /* Main content area - adjust padding */
        .main .block-container,
        section[data-testid="st.main"] > div:first-child {{
            background-color: {COLOR_SURFACE};
            padding: 2.5rem 3rem 4rem 3rem;
            border-radius: 16px;
            margin-top: 1.5rem;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.25);
            border: 1px solid {COLOR_BORDER};
        }}

        /* --- Typography Refinement --- */
        h1 {{
            color: {COLOR_TEXT_PRIMARY};
            font-weight: 700;
            background: {GRADIENT_PRIMARY};
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            padding-bottom: 0.6em;
            margin-bottom: 0.6em;
            font-size: 2.2rem;
            position: relative;
        }}
        h1::after {{
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            width: 100%;
            height: 2px;
            background: {GRADIENT_PRIMARY};
        }}

        h1 + p {{ /* Subtitle */
            color: {COLOR_TEXT_SECONDARY};
            font-size: 1.1rem;
            font-weight: 400;
            margin-top: -1.5rem;
            margin-bottom: 3rem;
            max-width: 65ch;
        }}

        h2 {{ /* Main section titles */
            color: {COLOR_TEXT_PRIMARY};
            font-weight: 600;
            margin-top: 2.5em;
            margin-bottom: 1.2em;
            padding-bottom: 0.5em;
            font-size: 1.6rem;
            position: relative;
        }}
        h2::after {{
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            width: 60px;
            height: 3px;
            background: {GRADIENT_PRIMARY};
            border-radius: 3px;
        }}

        h3 {{ /* Sub-section titles */
            color: {COLOR_TEXT_PRIMARY};
            font-weight: 600;
            margin-top: 2em;
            margin-bottom: 1em;
            font-size: 1.3rem;
        }}

        p, .stMarkdown p, .stText {{
            color: {COLOR_TEXT_SECONDARY};
            line-height: 1.7;
            font-size: 1rem;
        }}

        a {{ 
            color: {COLOR_PRIMARY}; 
            text-decoration: none; 
            font-weight: 500;
            transition: all 0.2s ease; 
        }}
        a:hover {{ 
            color: {COLOR_ACCENT}; 
            text-decoration: underline;
        }}

        /* --- Sidebar Enhancements --- */
        .stSidebar > div:first-child {{
            background: {GRADIENT_SECONDARY};
            border-right: 1px solid {COLOR_BORDER};
            padding: 1.8rem 1.5rem;
        }}

        .stSidebar h2 {{ /* "Client Access" */
            color: {COLOR_TEXT_PRIMARY};
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            padding-bottom: 0.8rem;
            position: relative;
        }}
        .stSidebar h2::after {{
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            width: 40px;
            height: 2px;
            background: {GRADIENT_PRIMARY};
            border-radius: 2px;
        }}

        /* Input field styling */
        .stSidebar .stTextInput {{
            margin-bottom: 0;
        }}
        .stSidebar .stTextInput label {{
            color: {COLOR_TEXT_SECONDARY};
            font-size: 0.85rem;
            font-weight: 500;
            margin-bottom: 0.5rem;
            display: block;
        }}
        .stSidebar .stTextInput input {{
            border-radius: 10px;
            border: 1px solid {COLOR_BORDER};
            background-color: rgba(15, 23, 42, 0.7);
            color: {COLOR_TEXT_PRIMARY};
            padding: 0.8rem 1rem;
            font-size: 0.95rem;
            width: 100%;
            transition: all 0.2s ease;
        }}
        .stSidebar .stTextInput input:focus {{
            border-color: {COLOR_PRIMARY};
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.3);
            outline: none;
            background-color: rgba(15, 23, 42, 0.9);
        }}

        /* Button styling */
        .stSidebar .stButton {{
            margin-top: 1rem;
        }}
        .stSidebar .stButton button {{
            background: {GRADIENT_PRIMARY};
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.8rem 1.2rem;
            width: 100%;
            font-weight: 600;
            font-size: 0.95rem;
            transition: all 0.2s ease;
            cursor: pointer;
            box-shadow: 0 2px 10px rgba(99, 102, 241, 0.3);
        }}
        .stSidebar .stButton button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
        }}
        .stSidebar .stButton button:active {{
            transform: translateY(0);
            box-shadow: 0 2px 5px rgba(99, 102, 241, 0.3);
        }}

        /* --- Portfolio Snapshot Enhancements --- */
        .stSidebar h5 {{ /* "Portfolio Snapshot" title */
            color: {COLOR_TEXT_PRIMARY};
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 1rem;
            margin-top: 2.5rem;
            padding-bottom: 0.6rem;
            position: relative;
        }}
        .stSidebar h5::after {{
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            width: 30px;
            height: 2px;
            background: {GRADIENT_PRIMARY};
            border-radius: 2px;
        }}

        .portfolio-summary {{
            padding: 1.5rem 1.2rem;
            margin-bottom: 1.5rem;
            border-radius: 12px;
            background: rgba(15, 23, 42, 0.5);
            border: 1px solid {COLOR_BORDER};
            backdrop-filter: blur(5px);
        }}

        .portfolio-summary .stMetric {{
            padding: 0.3rem 0;
            text-align: left;
            display: flex;
            flex-direction: column;
            align-items: flex-start;
        }}

        .portfolio-summary .stMetric > label {{
            color: {COLOR_TEXT_MUTED};
            font-size: 0.75rem;
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 0.5px;
            margin-bottom: 0.3rem;
            order: 1;
        }}

        .portfolio-summary .stMetric > div[data-testid="stMetricValue"] {{
            font-size: 1.25rem;
            color: {COLOR_TEXT_PRIMARY};
            font-weight: 700;
            line-height: 1.3;
            order: 2;
        }}

        /* Ensure columns in summary have good spacing */
        .portfolio-summary div[data-testid="stHorizontalBlock"] > div[data-testid="stVerticalBlock"] {{
            gap: 0.5rem;
        }}

        /* --- Holdings List Enhancements --- */
        .stSidebar .stExpander {{
            background-color: transparent;
            border: 1px solid {COLOR_BORDER};
            border-radius: 12px;
            margin-top: 1.5rem;
            overflow: hidden;
        }}

        .stSidebar .stExpander header {{
            font-size: 1rem;
            font-weight: 600;
            color: {COLOR_TEXT_PRIMARY};
            padding: 1rem 1.2rem;
            border-bottom: 1px solid {COLOR_BORDER};
            transition: all 0.2s ease;
        }}

        .stSidebar .stExpander:hover header {{
            background-color: rgba(30, 41, 59, 0.5);
        }}

        .stSidebar .stExpander[aria-expanded="true"] header {{
            border-bottom: 1px solid {COLOR_BORDER};
        }}

        .stSidebar .stExpander svg {{ fill: {COLOR_TEXT_SECONDARY}; }}

        .holding-list-container {{
            max-height: 350px;
            overflow-y: auto;
            padding: 8px 0;
            scrollbar-width: thin;
            scrollbar-color: {COLOR_BORDER} {COLOR_SURFACE};
            margin: 0;
        }}

        /* Holding Item Styling */
        .holding-item {{
            font-size: 0.9rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 16px;
            border-bottom: 1px solid {COLOR_BORDER};
            transition: all 0.15s ease;
            cursor: default;
        }}

        .holding-item:last-child {{ border-bottom: none; }}

        .holding-item:hover {{ 
            background-color: rgba(30, 41, 59, 0.7);
            transform: translateX(2px);
        }}

        .holding-item span:first-child {{
            font-weight: 600;
            color: {COLOR_TEXT_PRIMARY};
            flex-shrink: 0;
            margin-right: 12px;
        }}

        .holding-item .value-alloc {{
            text-align: right;
            color: {COLOR_TEXT_SECONDARY};
            font-weight: 500;
            font-size: 0.9rem;
            line-height: 1.3;
            flex-grow: 1;
            display: flex;
            flex-direction: column;
            align-items: flex-end;
        }}

        .holding-item .value-alloc span {{
            min-width: 45px;
            text-align: right;
            color: {COLOR_TEXT_PRIMARY};
            font-weight: 600;
            font-size: 0.9rem;
        }}

        .holding-item small {{
            color: {COLOR_TEXT_MUTED};
            font-size: 0.85rem;
            display: block;
            margin-top: 3px;
        }}

        /* --- Main Chat Area Enhancements --- */
        .stChatMessage {{
            background-color: {COLOR_SURFACE};
            border: 1px solid {COLOR_BORDER};
            border-radius: 14px;
            padding: 16px 22px;
            margin-bottom: 1.2rem;
            color: {COLOR_TEXT_SECONDARY};
            box-shadow: 0 4px 10px rgba(0,0,0,0.2);
            max-width: 80%;
            line-height: 1.7;
            display: flex;
            align-items: flex-start;
            clear: both;
            float: left;
        }}

        .stChatMessage span[data-testid^="chatAvatarIcon"] {{
            margin-right: 16px;
            margin-top: 4px;
            flex-shrink: 0;
            font-size: 1.2rem;
        }}

        .stChatMessage > div:not(:has(span[data-testid^="chatAvatarIcon"])) {{
            flex-grow: 1;
        }}

        /* User Message Specific Styles */
        div[data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-user"]) {{
            background: {GRADIENT_PRIMARY};
            border-color: {COLOR_PRIMARY};
            color: #e0e7ff;
            float: right;
            flex-direction: row-reverse;
        }}

        div[data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-user"]) span[data-testid^="chatAvatarIcon"] {{
            margin-right: 0;
            margin-left: 16px;
        }}

        div[data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-user"]) p,
        div[data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-user"]) div,
        div[data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-user"]) li {{
            color: #e0e7ff !important;
        }}

        div[data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-user"]) a {{
            color: #c7d2fe !important;
            text-decoration: underline;
        }}

        div[data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-user"]) code {{
            background-color: rgba(255, 255, 255, 0.15) !important;
            color: #e0e7ff !important;
        }}

        div[data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-user"]) strong {{
            color: white !important;
        }}

        /* Assistant Message Specific Styles */
        div[data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-assistant"]) {{
            color: {COLOR_TEXT_PRIMARY};
        }}

        div[data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-assistant"]) p,
        div[data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-assistant"]) li {{
            color: {COLOR_TEXT_PRIMARY};
        }}

        div[data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-assistant"]) strong {{
            color: {COLOR_TEXT_PRIMARY};
            font-weight: 700;
        }}

        /* --- Chat Input Enhancements --- */
        div[data-testid="stChatInput"] {{
            background-color: {COLOR_BACKGROUND};
            border-top: 1px solid {COLOR_BORDER};
            padding: 1rem 1.8rem 1.2rem 1.8rem;
            position: sticky;
            bottom: 0;
            z-index: 10;
        }}

        div[data-testid="stChatInput"] textarea {{
            font-family: 'Inter', sans-serif !important;
            background-color: {COLOR_SURFACE} !important;
            color: {COLOR_TEXT_PRIMARY} !important;
            border: 1px solid {COLOR_BORDER} !important;
            border-radius: 12px !important;
            padding: 1rem 1.3rem !important;
            font-size: 1rem !important;
            line-height: 1.6 !important;
            min-height: 60px;
            box-shadow: none;
            transition: all 0.2s ease;
        }}

        div[data-testid="stChatInput"] textarea:focus {{
            border-color: {COLOR_PRIMARY} !important;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.3) !important;
        }}

        /* Send button */
        div[data-testid="stChatInput"] button {{
            background: {GRADIENT_PRIMARY} !important;
            border-radius: 10px !important;
            bottom: 18px;
            right: 12px;
            transition: all 0.2s ease !important;
            box-shadow: 0 2px 8px rgba(99, 102, 241, 0.3) !important;
        }}

        div[data-testid="stChatInput"] button:hover {{
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4) !important;
        }}

        div[data-testid="stChatInput"] button svg {{
            fill: white !important;
        }}

        /* --- Error & Warning Messages Refined --- */
        .error-message, .token-warning {{
            border-left-width: 5px;
            padding: 1.2rem 1.5rem;
            border-radius: 8px;
            margin: 1.2rem 0;
            font-size: 0.95rem;
            line-height: 1.7;
            backdrop-filter: blur(5px);
        }}

        .error-message {{
            background-color: rgba(239, 68, 68, 0.15);
            border-left-color: {COLOR_DANGER};
            color: #fca5a5;
        }}

        .error-message strong {{ color: #f87171; }}

        .token-warning {{
            background-color: rgba(245, 158, 11, 0.15);
            border-left-color: {COLOR_WARNING};
            color: #fcd34d;
        }}

        .token-warning strong {{ color: #fbbf24; }}

        /* --- Spinner Refined --- */
        .stSpinner > div {{
            border-top-color: {COLOR_PRIMARY} !important;
            border-right-color: {COLOR_PRIMARY} !important;
            border-bottom-color: rgba(99, 102, 241, 0.3) !important;
            border-left-color: rgba(99, 102, 241, 0.3) !important;
            width: 28px !important;
            height: 28px !important;
        }}

        /* --- Markdown Content Styling Refined --- */
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{ border-bottom: 1px solid {COLOR_BORDER}; padding-bottom: 6px; margin-top: 2em; margin-bottom: 1.3em;}}
        .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {{ color: {COLOR_TEXT_PRIMARY}; font-weight: 600; margin-top: 1.7em; margin-bottom: 0.8em; }}
        .stMarkdown code {{ background-color: {COLOR_BORDER}; padding: 0.2em 0.5em; border-radius: 5px; font-size: 0.9em; color: {COLOR_TEXT_PRIMARY}; font-family: Consolas, Monaco, 'Andale Mono', 'Ubuntu Mono', monospace;}}
        .stMarkdown pre {{ background-color: {COLOR_BACKGROUND}; border: 1px solid {COLOR_BORDER}; padding: 1.2rem; border-radius: 10px; color: {COLOR_TEXT_PRIMARY}; font-family: Consolas, Monaco, 'Andale Mono', 'Ubuntu Mono', monospace; white-space: pre-wrap; word-wrap: break-word; font-size: 0.95em; margin: 1.5em 0; }}
        .stMarkdown pre code {{ background-color: transparent; padding: 0; font-size: inherit; }}
        .stMarkdown table {{ width: 100%; border-collapse: collapse; margin: 2em 0; font-size: 0.95rem; border: 1px solid {COLOR_BORDER}; box-shadow: 0 3px 6px rgba(0,0,0,0.1); }}
        .stMarkdown th {{ background-color: {COLOR_SURFACE}; border: 1px solid {COLOR_BORDER}; padding: 14px 18px; text-align: left; font-weight: 600; color: {COLOR_TEXT_PRIMARY}; }}
        .stMarkdown td {{ border: 1px solid {COLOR_BORDER}; padding: 14px 18px; color: {COLOR_TEXT_SECONDARY}; vertical-align: top; }}
        .stMarkdown tr:nth-child(even) td {{ background-color: rgba(30, 41, 59, 0.5); }}
        .stMarkdown ul, .stMarkdown ol {{ margin-left: 1.5em; padding-left: 1em; margin-bottom: 1.5em; color: {COLOR_TEXT_SECONDARY}; }}
        .stMarkdown li {{ margin-bottom: 0.8em; line-height: 1.7; }}
        .stMarkdown li > p {{ margin-bottom: 0.5em; }}
        .stMarkdown li::marker {{ color: {COLOR_TEXT_SECONDARY}; }}
        .stMarkdown blockquote {{ border-left: 5px solid {COLOR_PRIMARY}; margin-left: 0; padding: 1rem 2rem; background-color: rgba(30, 41, 59, 0.5); color: {COLOR_TEXT_SECONDARY}; font-style: italic; border-radius: 0 8px 8px 0;}}
        .stMarkdown blockquote p {{ color: inherit; margin-bottom: 0; }}

        /* Custom scrollbar */
        ::-webkit-scrollbar {{ width: 10px; height: 10px; }}
        ::-webkit-scrollbar-track {{ background: {COLOR_SURFACE}; border-radius: 10px; }}
        ::-webkit-scrollbar-thumb {{ background: #475569; border-radius: 10px; border: 2px solid {COLOR_SURFACE}; }}
        ::-webkit-scrollbar-thumb:hover {{ background: #64748b; }}

        /* --- Floating Action Button for Mobile --- */
        @media (max-width: 768px) {{
            .floating-action-btn {{
                position: fixed;
                bottom: 20px;
                right: 20px;
                width: 60px;
                height: 60px;
                border-radius: 50%;
                background: {GRADIENT_PRIMARY};
                color: white;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.5rem;
                box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
                z-index: 100;
                cursor: pointer;
                transition: all 0.3s ease;
            }}
            .floating-action-btn:hover {{
                transform: scale(1.1);
                box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5);
            }}
        }}
        </style>
        """, unsafe_allow_html=True)

    except Exception as e:
        logger.error(f"UI setup failed: {e}", exc_info=True)
        st.error("Failed to initialize application UI styling.")


# --- Portfolio Processing Logic (Defined within main.py) ---
# @log_execution_time # Optional performance logging
def process_raw_portfolio_data(client_id: str, raw_portfolio: dict) -> Optional[Dict]:
    """Processes raw portfolio data from DB into the standard context format."""
    local_logger = get_logger("PortfolioProcessing")
    if not isinstance(raw_portfolio, dict):
        local_logger.warning(f"Invalid raw portfolio data type for {client_id}: {type(raw_portfolio)}")
        return None
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
                    value = h.get('current_value')
                    if isinstance(value, (int, float)):
                        calculated_total_value += float(value)
                    else:
                        local_logger.warning(
                            f"Invalid 'current_value' type ({type(value)}) for holding {h.get('symbol')} in client {client_id}")
                else:
                    local_logger.warning(f"Holdings data for client {client_id} is not a list: {type(holdings_data)}")

        if isinstance(db_total_value, (int, float)) and db_total_value > 0:
            portfolio['portfolio_value'] = float(db_total_value)
        elif calculated_total_value >= 0:
            portfolio['portfolio_value'] = calculated_total_value
        else:
            local_logger.warning(f"Could not determine valid portfolio value for client {client_id}")

        if isinstance(holdings_data, list):
            total_val = portfolio['portfolio_value']
            for holding in holdings_data:
                if not isinstance(holding, dict) or not holding.get('symbol'):
                    local_logger.debug(f"Skipping invalid holding item: {holding}")
                    continue
                symbol = str(holding['symbol']).strip().upper()
                if not symbol:
                    local_logger.debug(f"Skipping holding item with empty symbol.")
                    continue

                value = holding.get('current_value', 0)
                if not isinstance(value, (int, float)):
                    local_logger.warning(
                        f"Holding '{symbol}' for client {client_id} has invalid value type ({type(value)}), using 0.")
                    value = 0.0
                else:
                    value = float(value)

                allocation = (value / total_val * 100) if total_val > 0 else 0.0
                portfolio['holdings'].append({"symbol": symbol, "value": value, "allocation": allocation})

        if validate_portfolio_data(portfolio):
            local_logger.debug(f"Processed portfolio validated successfully for {client_id}.")
            return portfolio
        else:
            local_logger.error(f"Final processed portfolio FAILED validation for client {client_id}. Data: {portfolio}")
            return None

    except Exception as e:
        local_logger.error(f"Critical error processing raw portfolio for {client_id}: {e}", exc_info=True)
        return None


# --- Database Interaction & Caching ---
@st.cache_data(ttl=300, show_spinner="Fetching client portfolio...")
def get_client_portfolio_cached(client_id: str) -> Optional[Dict]:
    """Cached wrapper for fetching and processing portfolio data."""
    local_logger = get_logger("PortfolioCache")
    validated_id = validate_client_id(client_id)
    if not validated_id:
        local_logger.warning(f"Invalid client ID format '{client_id}' blocked before cache fetch.")
        return None

    local_logger.info(f"Cache check/fetch for client ID: {validated_id}")
    try:
        db_client = AzurePostgresClient()
        raw_portfolio = db_client.get_client_portfolio(validated_id)

        if not raw_portfolio:
            local_logger.warning(f"No portfolio data returned from DB for {validated_id}")
            return None
        if not isinstance(raw_portfolio, dict):
            local_logger.error(f"Invalid data type received from DB for {validated_id}: {type(raw_portfolio)}")
            return None

        local_logger.info(f"Raw portfolio data retrieved from DB for {validated_id}, processing...")
        processed_portfolio = process_raw_portfolio_data(validated_id, raw_portfolio)

        if processed_portfolio:
            local_logger.info(f"Successfully processed/validated portfolio for {validated_id}. Caching result.")
            return processed_portfolio
        else:
            local_logger.error(f"Processed DB portfolio failed validation/processing for {validated_id}. Caching None.")
            return None
    except Exception as e:
        local_logger.error(f"Database or processing error during cache fetch for {validated_id}: {e}", exc_info=True)
        return None


# --- UI Components ---
def display_portfolio_summary(portfolio_data: Dict):
    """Displays enhanced portfolio summary using st.metric in the sidebar."""
    local_logger = get_logger("PortfolioDisplay")
    try:
        total_value = portfolio_data.get('portfolio_value', 0.0)
        risk_profile_raw = portfolio_data.get('risk_profile')
        risk_profile = risk_profile_raw.capitalize() if risk_profile_raw and isinstance(risk_profile_raw,
                                                                                        str) else 'N/A'

        # The container's border/bg is handled by CSS (.portfolio-summary)
        with st.sidebar.container():
            # Apply the class to the div wrapping the content
            st.markdown('<div class="portfolio-summary">', unsafe_allow_html=True)
            # Title is styled via h5 selector
            st.markdown("<h5>Portfolio Snapshot</h5>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                # CSS handles the label/value styling and alignment
                st.metric(label="Total Value", value=f"${total_value:,.2f}")
            with col2:
                st.metric(label="Risk Profile", value=risk_profile)
            st.markdown('</div>', unsafe_allow_html=True)  # Close the div

    except Exception as e:
        local_logger.error(f"Error displaying portfolio summary: {e}", exc_info=True)
        st.sidebar.warning("Could not display summary.", icon="⚠️")  # Use warning for less intrusion


def display_holdings_list(holdings_data: list):
    """Displays enhanced scrollable holdings list in the sidebar expander."""
    local_logger = get_logger("PortfolioDisplay")
    if not holdings_data:
        return
    try:
        client_name = st.session_state.get('client_context', {}).get('name', 'Client')
        # Expander styling is handled by CSS
        with st.sidebar.expander(f"{client_name} Holdings ({len(holdings_data)})", expanded=False):
            # Container for scrolling list
            st.markdown('<div class="holding-list-container">', unsafe_allow_html=True)
            sorted_holdings = sorted(holdings_data, key=lambda x: x.get('value', 0), reverse=True)
            for holding in sorted_holdings:
                symbol = holding.get('symbol', 'N/A')
                value = holding.get('value', 0.0)
                allocation = holding.get('allocation', 0.0)
                # Apply the holding-item class for styling from CSS
                # Structure uses flexbox for alignment (Symbol left, Value/Alloc right & stacked)
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
        local_logger.error(f"Error displaying holdings list: {e}", exc_info=True)
        st.sidebar.warning("Could not display holdings.", icon="⚠️")


def client_sidebar_manager():
    """Manages client ID input, loading, and portfolio display calls in the sidebar."""
    with st.sidebar:
        # Title styled via h2 selector
        st.markdown("## Client Access")
        st.session_state.setdefault('client_id_input', "")
        st.session_state.setdefault('client_id', None)
        st.session_state.setdefault('client_context', None)
        st.session_state.setdefault('portfolio_loaded', False)
        st.session_state.setdefault('load_error', None)

        # Input field - styled via .stTextInput selector
        # Use placeholder as implicit label
        client_id_input = st.text_input(
            "Client ID Input", # Label required by st.text_input but hidden
            value=st.session_state.client_id_input,
            placeholder="Enter Client ID (e.g., CLIENT123)",
            help="Enter the client identifier and click Load.",
            key="client_id_input_widget",
            label_visibility="collapsed" # Hide the actual label
        )
        st.session_state.client_id_input = client_id_input

        # Button - styled via .stButton selector
        load_button = st.button("Load Client Data", key="load_client_button", use_container_width=True)

        # --- Loading Logic (remains the same) ---
        if load_button:
            st.session_state.load_error = None
            if not client_id_input:
                st.session_state.load_error = "Please enter a Client ID."
                st.session_state.portfolio_loaded = False
                st.session_state.client_context = None
                st.session_state.client_id = None
                st.rerun()
            else:
                validated_id = validate_client_id(client_id_input)
                if validated_id:
                    if validated_id != st.session_state.get('client_id') or not st.session_state.get('portfolio_loaded'):
                        logger.info(f"Load button clicked for Client ID: {validated_id}")
                        st.session_state.client_id = validated_id
                        st.session_state.client_context = None
                        st.session_state.portfolio_loaded = False

                        portfolio = get_client_portfolio_cached(validated_id)
                        if portfolio:
                            st.session_state.client_context = portfolio
                            st.session_state.portfolio_loaded = True
                            st.session_state.load_error = None
                            client_name = portfolio.get('name', validated_id)
                            logger.info(f"Portfolio context loaded successfully via button for {client_name}")
                            st.toast(f"Loaded data for {client_name}", icon="✅")
                        else:
                            st.session_state.load_error = f"Failed to load/process data for '{validated_id}'. Verify ID."
                            st.session_state.portfolio_loaded = False
                            st.session_state.client_context = None
                            st.session_state.client_id = None
                            logger.warning(st.session_state.load_error + f" (Raw ID: {client_id_input})")
                        st.rerun()
                    else:
                        logger.debug(f"Client {validated_id} data already loaded. Skipping reload.")
                        st.toast(f"Client {st.session_state.client_context.get('name', validated_id)} already loaded.", icon="ℹ️")
                else:
                    st.session_state.load_error = f"Invalid Client ID format: '{client_id_input}'. Please check."
                    st.session_state.portfolio_loaded = False
                    st.session_state.client_context = None
                    st.session_state.client_id = None
                    logger.warning(st.session_state.load_error)
                    st.rerun()

        # --- Display Logic ---
        if st.session_state.load_error:
            # Display error using st.warning
            st.sidebar.warning(st.session_state.load_error, icon="⚠️")

        if st.session_state.portfolio_loaded and st.session_state.client_context:
            # Call display functions which now rely heavily on CSS
            display_portfolio_summary(st.session_state.client_context)
            display_holdings_list(st.session_state.client_context.get('holdings', []))
        elif client_id_input and not st.session_state.portfolio_loaded and not st.session_state.load_error and not load_button:
            st.sidebar.info("Click 'Load Client Data' above.", icon="⬆️")
        elif not client_id_input and not st.session_state.portfolio_loaded and not st.session_state.load_error:
             st.sidebar.info("Enter a Client ID to load data.", icon="🆔")


# --- Main Chat Interface ---
def main_chat_interface(agents: FinancialAgents):
    """Handles the main chat interaction area."""
    st.session_state.setdefault('conversation', [{"role": "assistant", "content": "Hello! How can I assist with your financial questions today?"}])

    # Display chat history - Styling handled by .stChatMessage CSS
    for message in st.session_state.conversation:
        avatar = USER_AVATAR if message["role"] == "user" else ASSISTANT_AVATAR
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"], unsafe_allow_html=True)

    # Handle new user input - Styling handled by stChatInput CSS
    if prompt := st.chat_input("Ask a financial question..."):
        st.session_state.conversation.append({"role": "user", "content": prompt})
        st.rerun()

    # Process agent response
    if st.session_state.conversation[-1]["role"] == "user":
        last_user_prompt = st.session_state.conversation[-1]["content"]

        if not validate_query_text(last_user_prompt):
            error_msg = "Your query seems too short, long, or invalid. Please ask a clear financial question (3-1500 chars)."
            styled_error_msg = generate_warning_html("Invalid Query", error_msg) # Uses util for HTML
            st.session_state.conversation.append({"role": "assistant", "content": styled_error_msg})
            st.rerun()
        else:
            with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
                message_placeholder = st.empty()
                message_placeholder.markdown("Thinking...") # Simple text indicator

                try:
                    client_context = st.session_state.get('client_context')
                    client_id = st.session_state.get('client_id')
                    log_prefix = f"Client {client_id}" if client_id else "Generic"
                    logger.info(f"{log_prefix}: Calling agent for query: '{last_user_prompt[:60]}...'")

                    response = agents.get_response(query=last_user_prompt, client_id=client_id,
                                                   client_context=client_context)

                    message_placeholder.markdown(response, unsafe_allow_html=True)
                    st.session_state.conversation.append({"role": "assistant", "content": response})

                    # DB Logging (remains same)
                    is_error_response = '<div class="error-message">' in response or '<div class="token-warning">' in response
                    if client_id and not is_error_response:
                        try:
                            db_client = AzurePostgresClient()
                            db_client.log_client_query(client_id=client_id, query=last_user_prompt[:1000], response=response[:2000])
                            logger.info(f"Logged query for client {client_id}")
                        except Exception as db_error:
                            logger.error(f"DB log FAILED for {client_id}: {db_error}", exc_info=True)
                    elif is_error_response:
                        logger.warning(f"Agent response contained error/warning for {log_prefix}, not logging to DB.")

                except Exception as agent_call_error:
                    logger.error(f"Critical error during FinancialAgents.get_response: {agent_call_error}", exc_info=True)
                    error_html = generate_error_html("Advisor Request Failed", f"An unexpected error occurred: {agent_call_error}")
                    message_placeholder.markdown(error_html, unsafe_allow_html=True)
                    st.session_state.conversation.append({"role": "assistant", "content": error_html})

            st.rerun()


# --- Application Entry Point ---
def main():
    """Main function to orchestrate the Streamlit application."""
    setup_ui() # Apply ENHANCED styling first

    # --- Header ---
    st.title(APP_TITLE)
    # Subtitle - styled via h1 + p selector
    st.markdown(f"<p>Your AI partner for financial insights and portfolio analysis.</p>", unsafe_allow_html=True)

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
            error_html = generate_error_html("System Initialization Failed!",
                                             f"The AI Advisor could not be loaded. Please check server logs or contact support. Error: {e}")
            st.error(error_html, icon="🚨")
            return None

    financial_agents = initialize_financial_agents()

    # --- Sidebar Manager ---
    client_sidebar_manager() # Manages client loading/display

    # --- Main Content Area (Chat Interface) ---
    if financial_agents:
        main_chat_interface(financial_agents)
    else:
        # Use markdown for a horizontal rule if agents fail
        st.markdown(f"<hr style='border-top: 1px solid {COLOR_BORDER}; margin: 2rem 0;'>", unsafe_allow_html=True)
        st.warning("🔴 AI Advisor features are unavailable due to an initialization error.", icon="⚠️")


# --- Final Application Exception Handler ---
if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass # Allow clean exit from st.stop()
    except Exception as e:
        critical_logger = get_logger("MainCriticalError")
        error_type = type(e).__name__
        error_message = str(e)
        full_traceback = traceback.format_exc()
        critical_logger.critical(f"Application encountered CRITICAL unhandled error: {error_type} - {error_message}", exc_info=True)
        critical_logger.critical(full_traceback)

        try:
            st.error(generate_error_html("Critical Application Error!",
                                         f"A critical error occurred: {error_type}. Please check the application logs or contact support."),
                     icon="💥")
        except Exception as st_err:
            critical_logger.critical(f"!!! FAILED to display critical error via st.error: {st_err}", exc_info=True)
            print(f"\n--- CRITICAL UNHANDLED ERROR ---", file=sys.stderr)
            print(f"Timestamp: {datetime.now()}", file=sys.stderr)
            print(f"Original Error: {error_type} - {error_message}", file=sys.stderr)
            print(f"Traceback:\n{full_traceback}", file=sys.stderr)
            print(f"Error during st.error display: {type(st_err).__name__} - {st_err}", file=sys.stderr)
            print(f"--- END CRITICAL ERROR ---\n", file=sys.stderr)