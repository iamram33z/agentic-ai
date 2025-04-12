import streamlit as st
from fintech_ai_bot.config import settings
from fintech_ai_bot.core.orchestrator import AgentOrchestrator
from fintech_ai_bot.utils import get_logger, validate_query_text, generate_error_html, generate_warning_html
from fintech_ai_bot.db.postgres_client import PostgresClient # For logging

logger = get_logger(__name__)

def display_chat_messages():
    """Displays the chat history from session state."""
    st.session_state.setdefault('conversation', [{"role": "assistant", "content": "Hello! How can I assist with your financial questions today?"}])
    for message in st.session_state.conversation:
        avatar = settings.user_avatar if message["role"] == "user" else settings.assistant_avatar
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"], unsafe_allow_html=True) # Allow HTML for errors/warnings

def handle_chat_input(orchestrator: AgentOrchestrator, db_client: PostgresClient):
    """Handles user input, orchestrator calls, response display, and logging."""
    if prompt := st.chat_input("Ask a financial question..."):
        st.session_state.conversation.append({"role": "user", "content": prompt})
        # Rerun immediately to display the user message
        st.rerun()

    # Process the latest user message if it hasn't been processed yet
    if st.session_state.conversation[-1]["role"] == "user":
        last_user_prompt = st.session_state.conversation[-1]["content"]
        client_id = st.session_state.get('client_id')
        client_context = st.session_state.get('client_context')
        log_prefix = f"Client {client_id}" if client_id else "Generic"

        if not validate_query_text(last_user_prompt):
            error_msg = "Your query seems too short, long, or invalid. Please ask a clear financial question (3-1500 chars)."
            styled_error_msg = generate_warning_html("Invalid Query", error_msg)
            st.session_state.conversation.append({"role": "assistant", "content": styled_error_msg})
            st.rerun() # Rerun to display the warning
            return # Stop processing this turn

        # Display thinking indicator and call orchestrator
        with st.chat_message("assistant", avatar=settings.assistant_avatar):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")

            try:
                logger.info(f"{log_prefix}: Calling orchestrator for query: '{last_user_prompt[:60]}...'")
                response = orchestrator.get_response(
                    query=last_user_prompt,
                    client_id=client_id,
                    client_context=client_context
                )
                message_placeholder.markdown(response, unsafe_allow_html=True)
                st.session_state.conversation.append({"role": "assistant", "content": response})

                # Log successful, non-error responses to DB if client_id exists
                is_error_response = '<div class="error-message">' in response or '<div class="token-warning">' in response
                if client_id and not is_error_response:
                    try:
                        # Log using the passed db_client instance
                        db_client.log_client_query(
                            client_id=client_id,
                            query=last_user_prompt[:1000],
                            response=response[:2000] # Log summary
                        )
                        # No need to log info here, db_client handles it
                    except Exception as db_log_error:
                         # db_client already logs errors, maybe add specific context here if needed
                        logger.error(f"UI layer caught DB log error for {client_id}: {db_log_error}")
                elif is_error_response:
                    logger.warning(f"Agent response contained error/warning for {log_prefix}, not logging interaction to DB.")

            except Exception as orchestrator_error:
                # This catches errors raised explicitly by orchestrator.get_response
                logger.error(f"Error during orchestrator.get_response call: {orchestrator_error}", exc_info=True)
                error_html = generate_error_html("Advisor Request Failed", f"An unexpected error occurred while processing your request.")
                message_placeholder.markdown(error_html, unsafe_allow_html=True)
                st.session_state.conversation.append({"role": "assistant", "content": error_html})

            # No need to rerun here, message placeholder is updated directly
            # st.rerun() # Avoid rerun if possible after response display