# Import Libraries
import os
from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools

# Load .env variables
load_dotenv()

# Get API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
AGNOS_API_KEY = os.getenv("AGNOS_API_KEY")

# Sanity check
print("GROQ_API_KEY loaded:", bool(GROQ_API_KEY))
print("AGNOS_API_KEY loaded:", bool(AGNOS_API_KEY))

# Initialize News-Agent
news_agent = Agent(
    name="News Analyst",
    role="Get top 3 market-moving news items",
    model=Groq(id="llama3-8b-8192"),  # More stable than 70b for function calls
    tools=[DuckDuckGoTools()],
    instructions=[
        "Format:",
        "1. [Date] Headline (Source)",
        "2. Sentiment: Positive/Neutral/Negative",
        "3. Key point (1 sentence)"
    ],
    markdown=True
)

# Initialize Financial Agent
financial_agent = Agent(
    name="Financial Analyst",
    role="Get key financial metrics",
    model=Groq(id="llama3-8b-8192"),
    tools=[YFinanceTools(
        stock_price=True,
        stock_fundamentals=True,
        key_financial_ratios=True
    )],
    instructions=[
        "Show data in this table format:",
        "| Metric       | Value |",
        "|--------------|-------|",
        "| Price        |       |",
        "| P/E Ratio    |       |",
        "| 52-Week High |       |",
        "| 52-Week Low  |       |"
    ],
    markdown=True
)

# Initialize Recommendation Agent
recommendation_agent = Agent(
    name="Investment Advisor",
    role="Provide concise recommendations",
    model=Groq(id="llama3-8b-8192"),
    instructions=[
        "Provide:",
        "1. Short-term outlook (1 sentence)",
        "2. Long-term outlook (1 sentence)",
        "3. Key risk (1 bullet point)"
    ],
    markdown=True
)

# Initialize Coordinator Agent
coordinator = Agent(
    name="Research Coordinator",
    role="Combine key insights",
    model=Groq(id="llama3-8b-8192"),
    team=[news_agent, financial_agent, recommendation_agent],
    instructions=[
        "Create brief report with:",
        "1. Top News (3 items max)",
        "2. Financial Snapshot (table)",
        "3. Recommendation Summary"
    ],
    markdown=True
)

# Debugging individual agents
print("\n🧪 Testing Financial Agent...")
try:
    financial_agent.print_response("Give financial metrics of Tesla", stream=True)
except Exception as e:
    print("❌ Financial Agent failed:", e)

print("\n🧪 Testing News Agent...")
try:
    news_agent.print_response("Top market-moving news about Tesla", stream=True)
except Exception as e:
    print("❌ News Agent failed:", e)

print("\n🧪 Testing Recommendation Agent...")
try:
    recommendation_agent.print_response("Give short and long term outlook of Tesla", stream=True)
except Exception as e:
    print("❌ Recommendation Agent failed:", e)

# Final execution through coordinator
print("\n🚀 Running Coordinator Agent...")
try:
    coordinator.print_response("Provide a concise analysis of Tesla's current market position", stream=True)
except Exception as e:
    print("❌ Coordinator Agent failed:", e)