import psycopg2
from typing import Dict, List
import os
from dotenv import load_dotenv

load_dotenv()


class AzurePostgres:
    def __init__(self):
        self.conn = psycopg2.connect(
            host=os.getenv("AZURE_PG_HOST"),
            database=os.getenv("AZURE_PG_DB"),
            user=os.getenv("AZURE_PG_USER"),
            password=os.getenv("AZURE_PG_PASSWORD"),
            sslmode=os.getenv("AZURE_PG_SSL")
        )

    def get_portfolio(self, client_id: str) -> Dict:
        cur = self.conn.cursor()
        cur.execute(f"""
            SELECT u.risk_profile, h.symbol, h.shares 
            FROM users u
            JOIN holdings h ON u.id = h.user_id
            WHERE u.client_id = '{client_id}'
        """)
        holdings = [{"symbol": row[1], "shares": row[2]} for row in cur.fetchall()]
        return {
            "client_id": client_id,
            "risk_profile": "moderate",  # Mock
            "holdings": holdings
        }