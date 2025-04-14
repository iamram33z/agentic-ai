# src/fintech_ai_bot/ui/chat_interface.py
# ENHANCED UI/UX: Improved thinking indicator

import streamlit as st
from fintech_ai_bot.config import settings
from fintech_ai_bot.core.orchestrator import AgentOrchestrator
from fintech_ai_bot.utils import get_logger, validate_query_text, generate_error_html, generate_warning_html
from fintech_ai_bot.db.postgres_client import PostgresClient # For logging

logger = get_logger(__name__)

def display_chat_messages():
    """Displays the chat history from session state."""
    # Initialize conversation state if it doesn't exist
    if 'conversation' not in st.session_state:
        st.session_state.conversation = [{"role": "assistant", "content": "Hello! How can I assist with your financial questions today? Please load a client profile using the sidebar if needed."}]

    # Display all messages
    for message in st.session_state.conversation:
        avatar = settings.user_avatar if message["role"] == "user" else settings.assistant_avatar
        with st.chat_message(message["role"], avatar=avatar):
            # Use unsafe_allow_html=True carefully, only needed if content contains intended HTML (like error messages)
            st.markdown(message["content"], unsafe_allow_html=True)

def handle_chat_input(orchestrator: AgentOrchestrator, db_client: PostgresClient):
    """Handles user input, orchestrator calls, response display, and logging."""
    prompt = st.chat_input("Ask a financial question...")

    if prompt:
        # Append user message immediately and rerun to show it
        st.session_state.conversation.append({"role": "user", "content": prompt})
        st.rerun()

    # Check if the last message is from the user and needs processing
    # This logic runs *after* the potential rerun from submitting input
    if st.session_state.conversation[-1]["role"] == "user":
        last_user_prompt = st.session_state.conversation[-1]["content"]
        client_id = st.session_state.get('client_id') # Get current client context
        client_context = st.session_state.get('client_context')
        log_prefix = f"Client '{client_id}'" if client_id else "Unidentified User"

        # --- Input Validation ---
        if not validate_query_text(last_user_prompt):
            error_msg = "Your query seems too short, long, or potentially invalid. Please provide a clear financial question (between 3 and 1500 characters)."
            # Using generate_warning_html assumes it creates themed or basic HTML
            styled_warning_msg = generate_warning_html("Invalid Query Format", error_msg)
            st.session_state.conversation.append({"role": "assistant", "content": styled_warning_msg})
            logger.warning(f"{log_prefix}: Invalid query format blocked: '{last_user_prompt[:60]}...'")
            st.rerun() # Rerun to display the validation warning
            return # Stop processing this invalid turn

        # --- Call Orchestrator and Display Response ---
        with st.chat_message("assistant", avatar=settings.assistant_avatar):
            message_placeholder = st.empty()
            # Improved thinking indicator
            message_placeholder.markdown("Processing your query... ⏳")

            try:
                logger.info(f"{log_prefix}: Calling orchestrator for query: '{last_user_prompt[:80]}...'")

                # Call the orchestrator with query and context
                response = orchestrator.get_response(
                    query=last_user_prompt,
                    client_id=client_id,
                    client_context=client_context # Pass loaded context if available
                )

                # Update the placeholder with the actual response
                message_placeholder.markdown(response, unsafe_allow_html=True) # Allow HTML for formatted errors/warnings from orchestrator
                st.session_state.conversation.append({"role": "assistant", "content": response})
                logger.info(f"{log_prefix}: Successfully received response from orchestrator.")

                # --- Database Logging ---
                # Log successful, non-error interactions to DB if client_id exists
                # Check if response indicates an error/warning (adjust check based on generate_error/warning_html output)
                is_error_response = '<div class="error-message">' in response or '<div class="token-warning">' in response or "Error:" in response[:20]

                if client_id and not is_error_response:
                    try:
                        # Log query and a summary of the response
                        db_client.log_client_query(
                            client_id=client_id,
                            query=last_user_prompt[:1000], # Limit logged query length
                            response=response[:2000] # Limit logged response length
                        )
                        # db_client should handle its own logging success/failure messages
                    except Exception as db_log_error:
                        # Log error if DB logging fails, but don't show to user
                        logger.error(f"UI layer: Failed to log interaction to DB for {log_prefix}: {db_log_error}", exc_info=False) # Avoid traceback spam for logging errors
                elif is_error_response:
                    logger.warning(f"{log_prefix}: Agent response contained error/warning, skipping DB logging for this interaction.")
                # else: No client ID, cannot log to client query table

            except Exception as orchestrator_error:
                # Catch errors explicitly raised by orchestrator.get_response or during the call
                logger.error(f"{log_prefix}: Error during orchestrator call: {orchestrator_error}", exc_info=True)
                # Use generate_error_html for consistent error display format
                error_html = generate_error_html("Advisor Request Failed", f"Sorry, I encountered an unexpected issue while processing your request. Please try again later or contact support if the problem persists.")
                message_placeholder.markdown(error_html, unsafe_allow_html=True)
                st.session_state.conversation.append({"role": "assistant", "content": error_html})

            # No st.rerun() needed here, as the message_placeholder is updated directly.
            # Adding a rerun would clear the thinking message prematurely.