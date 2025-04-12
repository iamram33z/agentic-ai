import sys
import traceback

import psycopg2
import random
from datetime import datetime, timedelta
import numpy as np
from faker import Faker
import os
from dotenv import load_dotenv
from pathlib import Path

# Determine project root relative to this script's location
# Assumes script is in /scripts/ and .env is in project root
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# Initialize Faker
fake = Faker()

# Database connection using environment variables
def get_db_connection():
    # Use os.getenv as before, assuming dotenv loaded them
    return psycopg2.connect(
        host=os.getenv("AZURE_PG_HOST"),
        database=os.getenv("AZURE_PG_DB"),
        user=os.getenv("AZURE_PG_USER"),
        password=os.getenv("AZURE_PG_PASSWORD"),
        sslmode=os.getenv("AZURE_PG_SSL", "require"),
        connect_timeout=5,
        # Optional: Set schema directly if needed, though client code also sets search_path
        # options=f'-c search_path={os.getenv("AZURE_PG_SCHEMA", "profiles")},public'
    )

# --- Asset configuration (Keep ASSET_TYPES, ASSET_PRICES, TRANSACTION_TYPES as before) ---
ASSET_TYPES = {
    'conservative': { 'etfs': ['BND', 'VTI', 'VXUS', 'GLD', 'IEF'], 'stocks': ['JNJ', 'PG', 'KO', 'PEP', 'WMT', 'MCD', 'T', 'VZ'], 'crypto': [] },
    'moderate': { 'etfs': ['VTI', 'VXUS', 'QQQ', 'IWM', 'XLF'], 'stocks': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM'], 'crypto': ['BTC-USD', 'ETH-USD'] },
    'aggressive': { 'etfs': ['ARKK', 'SOXX', 'IBB'], 'stocks': ['TSLA', 'NVDA', 'AMD', 'SNOW', 'MRNA', 'SHOP'], 'crypto': ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'DOT-USD'] }
}
ASSET_PRICES = {
    'BND': 72.50, 'VTI': 220.00, 'VXUS': 55.30, 'GLD': 180.40, 'IEF': 85.20, 'QQQ': 350.20, 'IWM': 190.75, 'XLF': 35.40, 'ARKK': 45.20, 'SOXX': 480.30, 'IBB': 135.60,
    'JNJ': 150.75, 'PG': 140.30, 'KO': 60.25, 'PEP': 170.40, 'WMT': 145.60, 'MCD': 250.30, 'T': 18.75, 'VZ': 40.20, 'AAPL': 175.25, 'MSFT': 420.50, 'GOOGL': 1250.75, 'AMZN': 150.30, 'META': 300.40, 'TSLA': 250.30, 'NVDA': 600.40, 'AMD': 110.25, 'SNOW': 160.75, 'MRNA': 120.40, 'SHOP': 70.30, 'JPM': 150.20,
    'BTC-USD': 62000.00, 'ETH-USD': 3400.00, 'SOL-USD': 120.00, 'ADA-USD': 0.45, 'DOT-USD': 6.80
}
TRANSACTION_TYPES = ['BUY', 'SELL']

# --- Generation functions (generate_clients, generate_holdings, generate_transactions) - Keep as before ---
def generate_clients(num_clients=20):
     # (Same logic as original data_generator.py)
     clients = []
     risk_distribution = ['conservative'] * 5 + ['moderate'] * 10 + ['aggressive'] * 5
     for i in range(num_clients):
         client_id = f"CLIENT{100 + i + 1}"
         risk = risk_distribution[i] if i < len(risk_distribution) else random.choice(['conservative', 'moderate', 'aggressive'])
         clients.append({
             'client_id': client_id, 'first_name': fake.first_name(), 'last_name': fake.last_name(),
             'risk_profile': risk, 'investment_horizon': random.choice(['short', 'medium', 'long'])
         })
     return clients

def generate_holdings(clients, min_holdings=500):
     # (Same logic as original data_generator.py)
     holdings = []
     holdings_per_client = max(min_holdings // len(clients), 4)
     for client in clients:
         risk = client['risk_profile']
         assets = ASSET_TYPES[risk]
         allocation = {'etfs': 0.6, 'stocks': 0.4, 'crypto': 0.0} if risk == 'conservative' else \
                      {'etfs': 0.5, 'stocks': 0.45, 'crypto': 0.05} if risk == 'moderate' else \
                      {'etfs': 0.3, 'stocks': 0.5, 'crypto': 0.2}
         for _ in range(holdings_per_client):
             asset_type = np.random.choice(['etfs', 'stocks', 'crypto'], p=[allocation['etfs'], allocation['stocks'], allocation['crypto']])
             if not assets[asset_type]: continue # Skip if no assets of this type for the profile
             symbol = random.choice(assets[asset_type])
             shares = round(random.uniform(0.1, 10.0), 4) if asset_type == 'crypto' else \
                      random.randint(10, 200) if symbol in ['BND', 'VTI', 'QQQ'] else \
                      random.randint(1, 100)
             avg_cost = ASSET_PRICES[symbol] * random.uniform(0.8, 1.2)
             current_value = shares * ASSET_PRICES[symbol] * random.uniform(0.9, 1.1)
             purchase_date = fake.date_between(start_date='-5y', end_date='today')
             holdings.append({
                 'client_id': client['client_id'], 'symbol': symbol, 'shares': shares,
                 'avg_cost': round(avg_cost, 2), 'current_value': round(current_value, 2), 'purchase_date': purchase_date
             })
     return holdings

def generate_transactions(holdings):
      # (Same logic as original data_generator.py)
     transactions = []
     for holding in holdings:
         for i in range(random.randint(1, 3)):
             transaction_date = fake.date_between(start_date=holding['purchase_date'], end_date='today')
             transactions.append({
                 'client_id': holding['client_id'], 'symbol': holding['symbol'], 'transaction_type': 'BUY',
                 'shares': holding['shares'] * random.uniform(0.3, 1.0), 'price': holding['avg_cost'] * random.uniform(0.95, 1.05),
                 'fee': round(random.uniform(5, 15), 2), 'transaction_time': transaction_date,
                 'notes': 'Initial purchase' if i == 0 else 'Additional purchase'
             })
             if random.random() < 0.3:
                 transactions.append({
                     'client_id': holding['client_id'], 'symbol': holding['symbol'], 'transaction_type': 'SELL',
                     'shares': holding['shares'] * random.uniform(0.1, 0.5), 'price': holding['avg_cost'] * random.uniform(1.05, 1.25),
                     'fee': round(random.uniform(5, 15), 2), 'transaction_time': fake.date_between(start_date=transaction_date, end_date='today'),
                     'notes': 'Profit taking' if random.random() < 0.7 else 'Portfolio rebalance'
                 })
     return transactions

# --- Schema/Loading functions (create_tables, load_data) - Keep as before ---
# Note: Ensure schema name used here matches settings.azure_pg_schema if using options in connect
SCHEMA_NAME = os.getenv("AZURE_PG_SCHEMA", "profiles")

def create_tables(conn):
     """Create database tables if they don't exist"""
     # (Same logic as original, ensure SCHEMA_NAME is used)
     with conn.cursor() as cur:
         cur.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA_NAME}")
         # Use f-string or psycopg2.sql.Identifier for schema name in table definitions
         cur.execute(f"""
         CREATE TABLE IF NOT EXISTS {SCHEMA_NAME}.users ( ... );
         CREATE TABLE IF NOT EXISTS {SCHEMA_NAME}.holdings ( ... );
         CREATE TABLE IF NOT EXISTS {SCHEMA_NAME}.transactions ( ... );
         CREATE TABLE IF NOT EXISTS {SCHEMA_NAME}.query_logs ( ... );
         CREATE INDEX IF NOT EXISTS idx_users_client_id ON {SCHEMA_NAME}.users(client_id);
         -- Add other indexes and functions/triggers using SCHEMA_NAME
         """)
         conn.commit()


def load_data(conn, clients, holdings, transactions):
     """Load generated data into database"""
     # (Same logic as original, ensure SCHEMA_NAME is used)
     with conn.cursor() as cur:
         # Insert clients
         for client in clients:
             cur.execute(f"INSERT INTO {SCHEMA_NAME}.users (...) VALUES (...) ON CONFLICT (client_id) DO NOTHING")
         # Get user IDs
         cur.execute(f"SELECT id, client_id FROM {SCHEMA_NAME}.users")
         user_map = {row[1]: row[0] for row in cur.fetchall()}
         # Insert holdings
         for holding in holdings:
             user_id = user_map.get(holding['client_id'])
             if user_id:
                 cur.execute(f"INSERT INTO {SCHEMA_NAME}.holdings (...) VALUES (...)")
         # Insert transactions
         for transaction in transactions:
              user_id = user_map.get(transaction['client_id'])
              if user_id:
                  cur.execute(f"INSERT INTO {SCHEMA_NAME}.transactions (...) VALUES (...)")
         conn.commit()


# --- Main execution and SQL script generation ---
def generate_sql_script(clients, holdings, transactions):
    """Generate SQL script for backup/reference"""
    # Define output path using PROJECT_ROOT
    output_path = PROJECT_ROOT / "data" / "database_export.sql"
    # (Same logic as original, ensure SCHEMA_NAME is used, write to output_path)
    with open(output_path, 'w') as f:
        f.write("-- Auto-generated SQL script\n")
        f.write(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA_NAME};\n\n")
        f.write("-- Clients\n")
        for client in clients:
            f.write(f"INSERT INTO {SCHEMA_NAME}.users ...;\n")
        f.write("\n-- Holdings\n")
        for holding in holdings:
             f.write(f"INSERT INTO {SCHEMA_NAME}.holdings ...;\n")
        f.write("\n-- Transactions\n")
        for transaction in transactions:
             f.write(f"INSERT INTO {SCHEMA_NAME}.transactions ...;\n")
    print(f"Generated SQL script: {output_path}")


def main():
    print("Starting data generation...")
    conn = None # Initialize conn
    try:
        conn = get_db_connection()
        # Create tables
        print("Creating tables (if they don't exist)...")
        create_tables(conn) # Pass connection object

        # Generate data
        print("Generating synthetic data...")
        clients = generate_clients(20)
        holdings = generate_holdings(clients, 500)
        transactions = generate_transactions(holdings)

        # Load data
        print("Loading data into database...")
        load_data(conn, clients, holdings, transactions) # Pass connection object
        print(f"Successfully loaded: {len(clients)} clients, {len(holdings)} holdings, {len(transactions)} transactions")

        # Generate SQL script backup
        print("Generating SQL backup script...")
        generate_sql_script(clients, holdings, transactions)

    except psycopg2.Error as db_err:
         print(f"Database Error: {db_err}", file=sys.stderr)
         if conn:
             conn.rollback() # Rollback any partial changes on error
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    main()