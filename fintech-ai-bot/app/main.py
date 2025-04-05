import streamlit as st
from agent import FinancialAgent
from utils import get_logger
import os
from dotenv import load_dotenv

load_dotenv()
logger = get_logger("StreamlitUI")


class FinTechApp:
    def __init__(self):
        self.agent = FinancialAgent()

    def run(self):
        st.set_page_config(page_title="FinTech AI", page_icon="💸")
        st.title("AI Financial Assistant")

        # Client ID Input
        client_id = st.text_input(
            "🔑 Client ID",
            placeholder="CLIENT001",
            help="Use CLIENT001, CLIENT002, or CLIENT003 for demo"
        )

        # Chat Interface
        query = st.text_area("💬 Ask a financial question:")

        if st.button("Get Advice"):
            if client_id and query:
                with st.spinner("Analyzing..."):
                    response = self.agent.get_response(query, client_id)
                    st.markdown(response)
            else:
                st.warning("Please enter Client ID and question")


if __name__ == "__main__":
    FinTechApp().run()