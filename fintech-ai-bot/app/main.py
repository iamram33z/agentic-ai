import streamlit as st
from agent import FinancialAgents
from dotenv import load_dotenv
import os
import time
from utils import get_logger
import json
import re
from datetime import datetime
from azure_postgres import AzurePostgresClient

load_dotenv()
logger = get_logger("StreamlitApp")


def setup_ui():
    """Configure Streamlit UI with professional layout"""
    try:
        st.set_page_config(
            page_title="FinTech AI Advisor",
            layout="wide",
            page_icon="💹",
            initial_sidebar_state="expanded"
        )

        # Professional CSS styling
        st.markdown("""
        <style>
            .card {
                padding: 20px;
                border-radius: 10px;
                background-color: #1e293b;
                margin-bottom: 20px;
                border-left: 3px solid #3b82f6;
            }
            .stock-card {
                padding: 15px;
                border-radius: 8px;
                background-color: #1e293b;
                margin: 10px 0;
                border: 1px solid #334155;
            }
            /* Your other existing CSS styles */
        </style>
        """, unsafe_allow_html=True)
    except Exception as e:
        logger.error(f"UI setup failed: {str(e)}")
        st.error("Failed to initialize application UI")


def get_client_portfolio(client_id: str) -> dict:
    """Get portfolio data from PostgreSQL database with enhanced error handling"""
    if not client_id or not isinstance(client_id, str):
        st.warning("Invalid client ID format")
        return None

    try:
        db_client = AzurePostgresClient()
        raw_portfolio = db_client.get_client_portfolio(client_id)

        if not raw_portfolio:
            st.warning(f"No portfolio found for client ID: {client_id}")
            return None

        # Calculate total value if not provided
        total_value = raw_portfolio.get('total_value', 0)
        if total_value == 0 and 'holdings' in raw_portfolio:
            total_value = sum(h.get('current_value', 0) for h in raw_portfolio['holdings'])

        portfolio = {
            "name": f"Client {client_id[-4:]}" if len(client_id) >= 4 else f"Client {client_id}",
            "risk_profile": raw_portfolio.get('risk_profile', 'Not specified'),
            "portfolio_value": total_value,
            "holdings": [],
            "last_update": raw_portfolio.get('last_updated', datetime.now().strftime("%Y-%m-%d"))
        }

        # Process holdings with validation
        for holding in raw_portfolio.get('holdings', []):
            if not holding.get('symbol'):
                continue

            current_value = holding.get('current_value', 0)
            allocation = (current_value / portfolio['portfolio_value']) * 100 if portfolio['portfolio_value'] > 0 else 0

            portfolio['holdings'].append({
                "symbol": holding['symbol'],
                "value": current_value,
                "allocation": allocation
            })

        return portfolio

    except Exception as e:
        logger.error(f"Database error for client {client_id}: {str(e)}", exc_info=True)
        st.error("Failed to load portfolio data. Please try again later.")
        return None


def display_portfolio(portfolio: dict):
    """Display portfolio information with validation"""
    if not portfolio or not isinstance(portfolio, dict):
        st.error("Invalid portfolio data")
        return

    try:
        with st.container():
            st.markdown(f"""
            <div class="card">
                <h3 style="margin-top: 0; color: #f8fafc;">📊 Portfolio Summary</h3>
                <div style="display: flex; justify-content: space-between; margin-bottom: 15px;">
                    <div>
                        <p style="font-size: 0.9rem; color: #94a3b8; margin-bottom: 5px;">Total Value</p>
                        <p style="font-size: 1.5rem; font-weight: 600; color: #f8fafc; margin: 0;">${portfolio['portfolio_value']:,.2f}</p>
                    </div>
                    <div>
                        <p style="font-size: 0.9rem; color: #94a3b8; margin-bottom: 5px;">Risk Profile</p>
                        <p style="font-size: 1.1rem; font-weight: 600; color: #f8fafc; margin: 0;">{portfolio['risk_profile']}</p>
                    </div>
                </div>
                <div style="margin-top: 15px;">
                    <p style="font-size: 0.9rem; color: #94a3b8; margin-bottom: 5px;">Last Updated</p>
                    <p style="font-size: 0.9rem; color: #f8fafc; margin: 0;">{portfolio['last_update']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("📦 Holdings Overview", expanded=True):
            st.markdown("""
            <div style="max-height: 300px; overflow-y: auto;">
            """, unsafe_allow_html=True)

            for holding in portfolio['holdings']:
                st.markdown(f"""
                <div class="stock-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                        <span style="font-weight: 600; font-size: 1rem;">{holding['symbol']}</span>
                        <span style="font-weight: 500; color: #f8fafc;">${holding['value']:,.2f}</span>
                    </div>
                    <div style="height: 6px; background-color: #334155; border-radius: 3px; overflow: hidden;">
                        <div style="height: 100%; width: {holding['allocation']}%; background-color: #3b82f6;"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: 5px;">
                        <span style="font-size: 0.8rem; color: #94a3b8;">{holding['allocation']:.1f}% of portfolio</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        logger.error(f"Portfolio display error: {str(e)}")
        st.error("Failed to display portfolio information")


def client_sidebar():
    """Enhanced client sidebar with better error handling"""
    with st.sidebar:
        try:
            client_id = st.text_input(
                "Client ID",
                placeholder="Enter Client ID (e.g., CLIENT110)",
                help="Enter your client ID to access your portfolio",
                key="client_id"
            )

            if st.session_state.get('client_id'):
                with st.spinner("Loading client data..."):
                    portfolio = get_client_portfolio(st.session_state.client_id)
                    if portfolio:
                        st.session_state.client_context = portfolio
                        display_portfolio(portfolio)
                    else:
                        st.error("No portfolio found for this client ID")
        except Exception as e:
            logger.error(f"Sidebar error: {str(e)}")
            st.error("Sidebar functionality temporarily unavailable")


def main_chat_interface(agents):
    """Enhanced chat interface with robust error handling"""
    try:
        st.title("💹 AI Financial Advisor")
        st.markdown("""
        <div style="border-bottom: 1px solid #334155; margin-bottom: 20px; padding-bottom: 10px;">
            <p style="font-size: 1.1rem; color: #94a3b8; margin: 0;">Get personalized investment advice powered by AI</p>
        </div>
        """, unsafe_allow_html=True)

        # Initialize conversation history
        if 'conversation' not in st.session_state:
            st.session_state.conversation = []

        # Display conversation history with validation
        for message in st.session_state.conversation:
            if not isinstance(message, dict) or 'role' not in message or 'content' not in message:
                continue
            with st.chat_message(message["role"], avatar="💼" if message["role"] == "user" else "🤖"):
                st.markdown(message["content"])

        # User input with validation
        if prompt := st.chat_input("Ask a financial question (e.g., 'Analyze my portfolio')"):
            if not isinstance(prompt, str) or len(prompt.strip()) == 0:
                st.warning("Please enter a valid question")
                return

            st.session_state.conversation.append({"role": "user", "content": prompt})

            with st.chat_message("user", avatar="💼"):
                st.markdown(prompt)

            client_context = st.session_state.get('client_context', {})
            client_id = st.session_state.get('client_id')

            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Analyzing your request..."):
                    try:
                        start_time = time.time()

                        # Safely prepare the query
                        context_str = f"My portfolio: {json.dumps(client_context, indent=2)}" if client_context else ""
                        enriched_prompt = f"{context_str}\n\n{prompt}" if context_str else prompt

                        response = agents.get_response(
                            query=enriched_prompt,
                            client_id=client_id,
                            client_context=client_context
                        )

                        if not response or not isinstance(response, str):
                            raise ValueError("Invalid response from agent")

                        processing_time = time.time() - start_time

                        st.session_state.conversation.append({
                            "role": "assistant",
                            "content": response
                        })

                        st.markdown(response)

                        st.markdown(f"""
                        <div style="font-size: 0.8rem; color: #64748b; text-align: right; margin-top: 10px;">
                            Processed in {processing_time:.2f}s
                        </div>
                        """, unsafe_allow_html=True)

                        # Log the interaction with validation
                        if client_id and isinstance(client_id, str):
                            try:
                                db_client = AzurePostgresClient()
                                db_client.log_client_query(
                                    client_id=client_id,
                                    query=prompt[:1000],  # Truncate
                                    response=response[:2000]  # Truncate
                                )
                            except Exception as db_error:
                                logger.error(f"Failed to log query: {str(db_error)}")

                    except Exception as e:
                        error_msg = f"⚠️ Error processing your request: {str(e)}"
                        logger.error(f"Chat processing error: {str(e)}", exc_info=True)
                        st.error(error_msg)
                        st.session_state.conversation.append({
                            "role": "assistant",
                            "content": error_msg
                        })

    except Exception as e:
        logger.critical(f"Chat interface failed: {str(e)}", exc_info=True)
        st.error("Chat functionality is currently unavailable. Please refresh the page.")


def main():
    try:
        setup_ui()
        client_sidebar()

        @st.cache_resource
        def get_agents():
            try:
                logger.info("Initializing FinancialAgents")
                return FinancialAgents()
            except Exception as e:
                logger.critical(f"Agent initialization failed: {str(e)}")
                st.error("System initialization failed. Please contact support.")
                return None

        agents = get_agents()
        if agents is None:
            return

        if 'client_id' in st.session_state:
            main_chat_interface(agents)
        else:
            st.info("Please enter your Client ID in the sidebar to begin")

    except Exception as e:
        logger.critical(f"Application crash: {str(e)}", exc_info=True)
        st.error("The application has encountered a critical error. Please try again later.")


if __name__ == "__main__":
    main()