# Import necessary libraries
import streamlit as st
import pandas as pd

# Assuming these imports are correct relative to your project structure
from fintech_ai_bot.config import settings
from fintech_ai_bot.core.orchestrator import AgentOrchestrator
from fintech_ai_bot.utils import get_logger, validate_query_text
from fintech_ai_bot.db.postgres_client import PostgresClient

logger = get_logger(__name__)

# Structured Content Rendering Logic
def render_structured_content(content_list):
    """Renders a list of structured content elements."""
    if not isinstance(content_list, list):
        #st.warning("⚠️ Expected content to be a list for structured response.")
        # Attempt to display non-list content as markdown or json as fallback
        try:
            st.markdown(str(content_list))
        except Exception:
             try:
                 st.json(content_list)
             except Exception:
                  st.error("Could not display the unexpected content format.")
        return

    for element in content_list:
        if not isinstance(element, dict) or "type" not in element:
            st.warning(f"⚠️ Skipping invalid element in structured response: {element}")
            continue

        el_type = element.get("type")
        content = element.get("content")
        text = element.get("text")
        data = element.get("data")

        try:
            if el_type == "header":
                level = element.get("level", 4)
                header_text = text or ""
                # Render headers as bold markdown for style
                st.markdown(f"**{header_text}**")


            elif el_type == "markdown":
                markdown_text = text or ""
                if markdown_text.strip(): # Avoid rendering empty markdown strings
                     st.markdown(markdown_text, unsafe_allow_html=False) # Security: Keep False

            elif el_type == "table":
                df = None
                if isinstance(data, list) and data and all(isinstance(row, dict) for row in data):
                    try:
                        df = pd.DataFrame(data)
                    except Exception as df_err:
                         logger.error(f"Error creating DataFrame from list: {df_err}", exc_info=True)
                         st.warning("⚠️ Could not create table from data list.")
                         st.json(data)
                elif isinstance(data, pd.DataFrame):
                    df = data

                if df is not None and not df.empty:
                    # hide_index=True often looks better for display tables
                    st.dataframe(df, use_container_width=True, hide_index=True)
                elif df is not None and df.empty:
                     st.caption("[Empty Table]") # Indicate empty table explicitly
                elif el_type == "table": # Only warn if type was table but df creation failed/no data
                    st.warning("⚠️ Table data is missing or not in expected list[dict]/DataFrame format.")
                    if data: st.json(data) # Display raw data if format is wrong but not empty

            elif el_type == "error":
                error_content = content or "An unspecified error occurred."
                st.error(str(error_content)) # Display as Streamlit error

            elif el_type == "warning":
                warning_content = content or "Received an unspecified warning."
                st.warning(str(warning_content)) # Display as Streamlit warning

            else:
                # Handle unknown types gracefully
                st.warning(f"⚠️ Unknown content type '{el_type}' in structured response.")
                st.json(element) # Display unknown structure as JSON

        except Exception as e:
            logger.error(f"Error rendering element type '{el_type}': {e}", exc_info=True)
            st.error(f"Could not render element: {element.get('type', 'Unknown')}")
            st.json(element) # Show raw element on render failure


# Chat History Display Logic

def display_chat_messages():
    """
    Displays the chat history using st.chat_message.
    Handles both standard markdown and new structured_response types.
    """
    # Initialize conversation state if it doesn't exist
    if 'conversation' not in st.session_state:
        client_context = st.session_state.get('client_context')
        client_name = None
        if isinstance(client_context, dict):
            client_name = client_context.get('name')

        # Use client name in welcome message if available and different from ID
        client_id_in_state = st.session_state.get('client_id')
        if client_name and client_id_in_state and client_name != client_id_in_state :
             welcome_msg = f"Hello! Client **{client_name}** (`{client_id_in_state}`) is loaded. How can I assist?"
        elif client_id_in_state:
             welcome_msg = f"Hello! Client ID **{client_id_in_state}** is loaded. How can I assist?"
        else:
             welcome_msg = "Hello! How can I assist? Load a client profile for specific analysis."

        # Store initial message with markdown type
        st.session_state.conversation = [{"role": "assistant", "content": welcome_msg, "type": "markdown"}]

    # Display all messages from history
    for i, message in enumerate(st.session_state.conversation):
        # Safely get avatar URLs
        avatar_url = None
        role = message.get("role", "assistant") # Default to assistant if role missing
        if role == "user":
            avatar_url = getattr(settings, 'user_avatar', None)
        else: # Assistant or other roles
            avatar_url = getattr(settings, 'assistant_avatar', None)

        # Display message in chat container
        with st.chat_message(role, avatar=avatar_url):
            msg_type = message.get("type", "markdown") # Default to markdown
            content = message.get("content")

            if content is None:
                st.warning(f"⚠️ Empty message content found in history (index {i}).")
                continue

            try:
                # Render based on type
                if msg_type == "structured_response":
                    render_structured_content(content) # Use the dedicated renderer
                elif msg_type == "error":
                    # Errors might now be part of structured response, but handle direct error type too
                    st.error(str(content))
                elif msg_type == "warning":
                    st.warning(str(content))
                elif isinstance(content, str): # Default to markdown if it's a string
                    st.markdown(content, unsafe_allow_html=False)
                else: # Fallback for unexpected content types stored directly
                    st.warning(f"⚠️ Unexpected message content type '{type(content)}' found in history (index {i}). Displaying as JSON.")
                    st.json(content)

            except Exception as e:
                logger.error(f"Failed to render message index {i} (type: {msg_type}, role: {role}): {e}", exc_info=True)
                st.error(f"Could not display message content for index {i}.")


# Input Handling Logic
def handle_chat_input(orchestrator: AgentOrchestrator, db_client: PostgresClient):
    """Handles user input, orchestrator calls, structured response display, and logging."""
    prompt = st.chat_input("Ask a financial question...")

    if prompt:
        # Append user message (will align right)
        st.session_state.conversation.append({"role": "user", "content": prompt, "type": "markdown"})
        # Rerun immediately to show the user's input
        st.rerun()

    # Check if the last message is from the user, indicating the assistant should respond
    if st.session_state.conversation and st.session_state.conversation[-1]["role"] == "user":
        last_user_prompt = st.session_state.conversation[-1]["content"]
        client_id = st.session_state.get('client_id')
        client_context = st.session_state.get('client_context')
        log_prefix = f"Client '{client_id}'" if client_id else "Unidentified User"

        # Input Validation
        if not validate_query_text(last_user_prompt):
            error_msg = "⚠️ **Invalid Query Format:** Your query must be between 3 and 1500 characters and follow the expected format."
            # Append assistant warning (will align left)
            st.session_state.conversation.append({"role": "assistant", "content": error_msg, "type": "warning"})
            logger.warning(f"{log_prefix}: Invalid query format blocked by UI. Query: '{last_user_prompt[:60]}...'")
            st.stop()

        # Call Orchestrator and Display Response (within assistant's chat message)
        assistant_avatar_url = getattr(settings, 'assistant_avatar', None)
        # Use chat_message context manager to ensure spinner and response appear in the assistant bubble
        with st.chat_message("assistant", avatar=assistant_avatar_url):
            response_obj = None # Will hold the list[dict] or error structure
            assistant_message_to_log = {"role": "assistant"} # Prepare message to add to history

            with st.spinner("Thinking..."):
                try:
                    logger.info(f"{log_prefix}: Calling orchestrator for query: '{last_user_prompt[:80]}...'")
                    # Orchestrator now returns List[Dict[str, Any]]
                    response_obj = orchestrator.get_response(
                        query=last_user_prompt,
                        client_id=client_id,
                        client_context=client_context
                    )
                    logger.info(f"{log_prefix}: Successfully received response structure from orchestrator.")

                    # Render the structured response immediately
                    render_structured_content(response_obj)

                    # Prepare message for history logging
                    assistant_message_to_log["content"] = response_obj
                    assistant_message_to_log["type"] = "structured_response"

                except Exception as e:
                    # Catch errors during the orchestrator call itself
                    logger.error(f"{log_prefix}: Unexpected error calling orchestrator: {e}", exc_info=True)
                    error_msg_content = f"🚨 **System Error:** Failed to get response from advisor. Please check logs or try again later."
                    # Display user-friendly error within the assistant's chat bubble
                    st.error(error_msg_content)
                    # Log a structured error message to history
                    response_obj = [{"type": "error", "content": f"System Error (UI Layer): {e}"}] # Create error structure
                    assistant_message_to_log = {"role": "assistant", "content": response_obj, "type": "structured_response"}

            # Append the assistant's full message (which is the structured list) to history *after* displaying it
            st.session_state.conversation.append(assistant_message_to_log)

            # Database Logging
            is_error_response = isinstance(response_obj, list) and len(response_obj) == 1 and response_obj[0].get("type") == "error"
            response_summary_for_db = "[Structured Response Provided]"
            if is_error_response:
                 response_summary_for_db = f"[Error Response]: {response_obj[0].get('content', '')}"[:2000] # Log error content
                 logger.warning(f"{log_prefix}: Orchestrator returned an error response, logging error summary.")

            # Log only if we have a client ID and the response wasn't purely an error caught *before* logging
            if client_id and response_obj: # Check response_obj exists
                 if not is_error_response: # Log success placeholder
                     try:
                         db_client.log_client_query(
                             client_id=client_id,
                             query_text=last_user_prompt[:1000],
                             response_summary=response_summary_for_db
                         )
                     except TypeError as te:
                         logger.error(f"UI layer: TypeError during DB log for {log_prefix}. Check log_client_query args. Error: {te}", exc_info=False)
                     except Exception as db_log_error:
                         logger.error(f"UI layer: Failed to log interaction to DB for {log_prefix}: {db_log_error}", exc_info=False)