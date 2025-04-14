# src/fintech_ai_bot/ui/chat_interface.py
# CORRECTED: Safely handle None value for client_context

import streamlit as st
import pandas as pd
from fintech_ai_bot.config import settings
from fintech_ai_bot.core.orchestrator import AgentOrchestrator
from fintech_ai_bot.utils import get_logger, validate_query_text
from fintech_ai_bot.db.postgres_client import PostgresClient
import time

logger = get_logger(__name__)

# --- Display Logic ---

def render_structured_content(content_list):
    """Renders a list of structured content elements (FUTURE USE)."""
    if not isinstance(content_list, list):
        st.warning("⚠️ Expected content to be a list for structured response.")
        st.markdown(str(content_list))
        return

    for element in content_list:
         if not isinstance(element, dict) or "type" not in element:
            st.warning(f"⚠️ Skipping invalid element in structured response: {element}")
            continue

         el_type = element.get("type")
         try:
            if el_type == "header":
                level = element.get("level", 2)
                text = element.get("text", "")
                if level == 1: st.header(text)
                elif level == 2: st.subheader(text)
                elif level == 3: st.markdown(f"### {text}")
                elif level == 4: st.markdown(f"#### {text}")
                else: st.markdown(f"##### {text}")
            elif el_type == "markdown":
                st.markdown(element.get("text", ""), unsafe_allow_html=False)
            elif el_type == "table":
                data = element.get("data")
                if isinstance(data, list) and data:
                    df = pd.DataFrame(data)
                    st.dataframe(df, use_container_width=True)
                elif isinstance(data, pd.DataFrame):
                     st.dataframe(data, use_container_width=True)
                else:
                    st.warning("⚠️ Table data is missing or not in expected list/DataFrame format.")
                    if data: st.json(data)
            else:
                st.warning(f"⚠️ Unknown content type '{el_type}' in structured response.")
                st.json(element)
         except Exception as e:
            logger.error(f"Error rendering element type '{el_type}': {e}", exc_info=True)
            st.error(f"Could not render element: {element.get('type', 'Unknown')}")


def display_chat_messages():
    """Displays the chat history, rendering messages based on their type."""
    # Initialize conversation state if it doesn't exist
    if 'conversation' not in st.session_state:
        # --- CORRECTED CONTEXT HANDLING ---
        client_context = st.session_state.get('client_context') # Get context, could be None
        client_name = None
        # Only try to get 'name' if client_context is actually a dictionary
        if isinstance(client_context, dict):
            client_name = client_context.get('name')
        # --- END CORRECTION ---

        # Now check client_name (which might be None)
        if client_name and client_name != st.session_state.get('client_id'): # Check if a real name is loaded
             welcome_msg = f"Hello! Client **{client_name}** is loaded. How can I assist?"
        else:
             welcome_msg = "Hello! How can I assist? Load a client profile for specific analysis."
        st.session_state.conversation = [{"role": "assistant", "content": welcome_msg, "type": "markdown"}] # Default type

    # Display all messages
    for i, message in enumerate(st.session_state.conversation):
        avatar = settings.user_avatar if message["role"] == "user" else settings.assistant_avatar
        with st.chat_message(message["role"], avatar=avatar):
            msg_type = message.get("type", "markdown") # Default to markdown if type is missing
            content = message.get("content")

            if content is None:
                st.warning(f"⚠️ Empty message content found in history (index {i}).")
                continue

            try:
                # Handle specific error/warning types first
                if msg_type == "error" or (isinstance(content, str) and content.strip().startswith("Error:")):
                    # Remove "Error:" prefix for cleaner display if present
                    display_content = content.replace("Error:", "", 1).strip()
                    st.error(display_content)
                elif msg_type == "warning" or (isinstance(content, str) and content.strip().startswith("⚠️")):
                    st.warning(content)
                # Handle potential future structured responses
                elif msg_type == "structured_response":
                    render_structured_content(content)
                # Default: Render content as Markdown (handles tables/headers if present in string)
                else:
                    st.markdown(str(content), unsafe_allow_html=False) # unsafe_allow_html=False is safer

            except Exception as e:
                 logger.error(f"Failed to render message index {i} (type: {msg_type}): {e}", exc_info=True)
                 st.error(f"Could not display message content.")


# --- Input Handling Logic ---
def handle_chat_input(orchestrator: AgentOrchestrator, db_client: PostgresClient):
    """Handles user input, orchestrator calls (receiving single string), response display, and logging."""
    prompt = st.chat_input("Ask a financial question...")

    if prompt:
        st.session_state.conversation.append({"role": "user", "content": prompt, "type": "markdown"})
        st.rerun()

    if st.session_state.conversation[-1]["role"] == "user":
        last_user_prompt = st.session_state.conversation[-1]["content"]
        client_id = st.session_state.get('client_id')
        client_context = st.session_state.get('client_context')
        log_prefix = f"Client '{client_id}'" if client_id else "Unidentified User"

        # --- Input Validation ---
        if not validate_query_text(last_user_prompt):
            # Orchestrator also validates, but catch early in UI
            error_msg = "⚠️ **Invalid Query Format:** Query is too short/long or invalid. Please ask a clear question (3-1500 chars)."
            st.session_state.conversation.append({"role": "assistant", "content": error_msg, "type": "warning"})
            logger.warning(f"{log_prefix}: Invalid query format blocked by UI: '{last_user_prompt[:60]}...'")
            st.rerun()
            return

        # --- Call Orchestrator and Display Response ---
        with st.chat_message("assistant", avatar=settings.assistant_avatar):
            response_str = None # Initialize
            with st.spinner("Thinking..."):
                try:
                    logger.info(f"{log_prefix}: Calling orchestrator for query: '{last_user_prompt[:80]}...'")

                    # --- Call Orchestrator (Returns single string) ---
                    response_str = orchestrator.get_response(
                        query=last_user_prompt,
                        client_id=client_id,
                        client_context=client_context
                    )
                    logger.info(f"{log_prefix}: Successfully received response from orchestrator.")

                    # --- Display Full Response ---
                    if response_str:
                         # Check if response is an error string from orchestrator
                         if response_str.strip().startswith("Error:"):
                              error_content = response_str.replace("Error:", "", 1).strip()
                              st.error(error_content)
                              st.session_state.conversation.append({"role": "assistant", "content": error_content, "type": "error"})
                         else:
                              # Display as Markdown - RELIES ON ORCHESTRATOR FOR FORMATTING (headers, tables)
                              st.markdown(response_str, unsafe_allow_html=False)
                              st.session_state.conversation.append({"role": "assistant", "content": response_str, "type": "markdown"})
                    else:
                         logger.warning(f"{log_prefix}: Orchestrator returned an empty response.")
                         st.warning("Received an empty response from the advisor.")
                         st.session_state.conversation.append({"role": "assistant", "content": "[Received empty response]", "type": "warning"})

                except Exception as e:
                    # Catch unexpected errors during the call itself (less likely if orchestrator handles its errors)
                    logger.error(f"{log_prefix}: Unexpected error calling orchestrator: {e}", exc_info=True)
                    error_msg = f"🚨 **System Error:** Failed to get response from advisor: `{e}`."
                    st.error(error_msg)
                    st.session_state.conversation.append({"role": "assistant", "content": error_msg, "type": "error"})
                    response_str = None # Ensure no logging happens

            # --- Database Logging (after response or error) ---
            if response_str and client_id and not response_str.strip().startswith("Error:"):
                 try:
                    db_client.log_client_query(
                        client_id=client_id,
                        query_text=last_user_prompt[:1000],
                        response_summary=response_str[:2000]
                    )
                 except TypeError as te:
                      logger.error(f"UI layer: TypeError during DB log for {log_prefix}. Check args. Error: {te}", exc_info=False)
                 except Exception as db_log_error:
                    logger.error(f"UI layer: Failed to log interaction to DB for {log_prefix}: {db_log_error}", exc_info=False)
            elif response_str and response_str.strip().startswith("Error:"):
                 logger.warning(f"{log_prefix}: Orchestrator returned an error response, skipping DB logging.")