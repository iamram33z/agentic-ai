from groq import Groq
from agno import Agent
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools
from faiss_db import FAISSDB
from azure_postgres import AzurePostgres
import os
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()


class FinancialAgent:
    def __init__(self):
        self.groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.db = AzurePostgres()
        self.faiss = FAISSDB()
        self.tools = self._setup_tools()

    def _setup_tools(self) -> List[Agent.Tool]:
        # Agnos Tools
        market_tool = YFinanceTools()
        research_tool = DuckDuckGoTools()

        # Custom Tools
        @Agent.Tool
        def get_portfolio(client_id: str) -> Dict:
            return self.db.get_portfolio(client_id)

        @Agent.Tool
        def search_policies(query: str) -> str:
            return self.faiss.search(query)

        return [market_tool, research_tool, get_portfolio, search_policies]

    def get_response(self, query: str, client_id: str) -> str:
        # Retrieve context
        portfolio = self.db.get_portfolio(client_id)
        policies = self.faiss.search(query)

        # Generate prompt
        prompt = f"""
        **Client**: {client_id}  
        **Risk Profile**: {portfolio.get('risk_profile')}  
        **Portfolio**: {portfolio.get('holdings')}  

        **Question**: {query}  
        **Relevant Policies**: {policies}  

        Provide a concise, actionable response.
        """

        # Groq API call
        response = self.groq.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="mixtral-8x7b-32768",
            temperature=0.7
        )

        return response.choices[0].message.content