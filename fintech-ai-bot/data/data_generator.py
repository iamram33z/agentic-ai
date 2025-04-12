# scripts/data_generator.py

import sys
import traceback
import psycopg2
from psycopg2 import sql # Import sql for safe identifiers
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
    """Establishes a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            host=os.getenv("AZURE_PG_HOST"),
            database=os.getenv("AZURE_PG_DB"),
            user=os.getenv("AZURE_PG_USER"),
            password=os.getenv("AZURE_PG_PASSWORD"),
            sslmode=os.getenv("AZURE_PG_SSL", "require"),
            connect_timeout=5,
        )
        print("Database connection successful.")
        return conn
    except psycopg2.OperationalError as e:
        print(f"FATAL: Database connection failed: {e}", file=sys.stderr)
        sys.exit(1) # Exit if connection fails

# --- Asset configuration ---
ASSET_TYPES = {
    'conservative': { 'etfs': ['BND', 'VTI', 'VXUS', 'GLD', 'IEF'], 'stocks': ['JNJ', 'PG', 'KO', 'PEP', 'WMT', 'MCD', 'T', 'VZ'], 'crypto': [] },
    'moderate': { 'etfs': ['VTI', 'VXUS', 'QQQ', 'IWM', 'XLF'], 'stocks': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM'], 'crypto': ['BTC-USD', 'ETH-USD'] },
    'aggressive': { 'etfs': ['ARKK', 'SOXX', 'IBB'], 'stocks': ['TSLA', 'NVDA', 'AMD', 'SNOW', 'MRNA', 'SHOP'], 'crypto': ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'DOT-USD'] }
}
ASSET_PRICES = {
    'BND': 72.50, 'VTI': 220.00, 'VXUS': 55.30, 'GLD': 180.40, 'IEF': 85.20, 'QQQ': 350.20, 'IWM': 190.75, 'XLF': 35.40, 'ARKK': 45.20, 'SOXX': 480.30, 'IBB': 135.60,
    'JNJ': 150.75, 'PG': 140.30, 'KO': 60.25, 'PEP': 170.40, 'WMT': 145.60, 'MCD': 250.30, 'T': 18.75, 'VZ': 40.20, 'AAPL': 175.25, 'MSFT': 420.50, 'GOOGL': 175.75, # Adjusted GOOGL price
    'AMZN': 180.30, 'META': 450.40, 'TSLA': 170.30, 'NVDA': 880.40, 'AMD': 160.25, 'SNOW': 160.75, 'MRNA': 105.40, 'SHOP': 65.30, 'JPM': 190.20, # Adjusted prices
    'BTC-USD': 68000.00, 'ETH-USD': 3500.00, 'SOL-USD': 150.00, 'ADA-USD': 0.55, 'DOT-USD': 7.20 # Adjusted crypto prices
}
TRANSACTION_TYPES = ['BUY', 'SELL']

# --- Generation functions ---
def generate_clients(num_clients=20):
     """Generates a list of synthetic client data."""
     clients = []
     # Ensure diverse risk profiles even with small numbers
     risks = ['conservative', 'moderate', 'aggressive']
     risk_distribution = (risks * ((num_clients // len(risks)) + 1))[:num_clients]
     random.shuffle(risk_distribution)

     for i in range(num_clients):
         client_id = f"CLIENT{100 + i + 1}"
         clients.append({
             'client_id': client_id,
             'first_name': fake.first_name(),
             'last_name': fake.last_name(),
             'risk_profile': risk_distribution[i],
             'investment_horizon': random.choice(['short', 'medium', 'long'])
         })
     return clients

def generate_holdings(clients, min_total_holdings=500):
    """Generates a list of synthetic holdings for the clients."""
    holdings = []
    holdings_per_client = max(min_total_holdings // len(clients), 4)

    for client in clients:
        risk = client['risk_profile']
        assets = ASSET_TYPES[risk]
        # Define allocation based on risk profile
        if risk == 'conservative':
            allocation = {'etfs': 0.6, 'stocks': 0.4, 'crypto': 0.0}
        elif risk == 'moderate':
            allocation = {'etfs': 0.5, 'stocks': 0.45, 'crypto': 0.05}
        else: # aggressive
            allocation = {'etfs': 0.3, 'stocks': 0.5, 'crypto': 0.2}

        # Ensure probabilities sum to 1
        total_prob = sum(allocation.values())
        if not np.isclose(total_prob, 1.0):
            print(f"Warning: Allocation probabilities for {risk} do not sum to 1 ({total_prob}). Adjusting.")
            # Simple normalization (adjust as needed)
            factor = 1.0 / total_prob
            allocation = {k: v * factor for k, v in allocation.items()}


        client_holdings_count = 0
        attempts = 0
        max_attempts = holdings_per_client * 3 # Prevent infinite loop

        while client_holdings_count < holdings_per_client and attempts < max_attempts:
            attempts += 1
            # Choose asset type based on allocation probabilities
            asset_type = np.random.choice(
                list(allocation.keys()),
                p=list(allocation.values())
            )

            # Ensure there are assets of the chosen type available for this risk profile
            if not assets[asset_type]:
                continue # Skip if no assets of this type defined

            symbol = random.choice(assets[asset_type])

            # Check if price exists, skip if not found
            if symbol not in ASSET_PRICES:
                print(f"Warning: Price not found for symbol '{symbol}'. Skipping.")
                continue

            # Generate realistic share amounts
            if asset_type == 'crypto':
                # Allow smaller crypto amounts too
                shares = round(random.uniform(0.001, 5.0), 6) # More precision for crypto
            elif symbol in ['BND', 'VTI', 'QQQ', 'IEF', 'XLF']: # Common ETFs might have higher share counts
                shares = float(random.randint(10, 250))
            else: # Stocks or other ETFs
                shares = float(random.randint(1, 150))

            # Generate cost basis and current value with some variance around the defined price
            base_price = ASSET_PRICES[symbol]
            avg_cost = base_price * random.uniform(0.85, 1.15) # Wider variance for cost basis
            current_value = shares * base_price * random.uniform(0.95, 1.05) # Smaller variance for current price snapshot
            purchase_date = fake.date_between(start_date='-5y', end_date='-1w') # Purchase date in the past

            holdings.append({
                'client_id': client['client_id'],
                'symbol': symbol,
                'shares': shares,
                'avg_cost': round(avg_cost, 4), # Allow more precision for avg cost
                'current_value': round(current_value, 2),
                'purchase_date': purchase_date
            })
            client_holdings_count += 1

        if attempts >= max_attempts:
            print(f"Warning: Max attempts reached for client {client['client_id']}. Generated {client_holdings_count}/{holdings_per_client} holdings.")

    return holdings

def generate_transactions(holdings):
    """Generates a list of synthetic transactions based on holdings."""
    transactions = []
    for holding in holdings:
        # Must have at least one BUY transaction (initial purchase)
        initial_purchase_date = holding['purchase_date']
        # Generate initial buy transaction slightly before or on purchase date
        initial_buy_date = initial_purchase_date - timedelta(days=random.randint(0, 30))
        transactions.append({
            'client_id': holding['client_id'],
            'symbol': holding['symbol'],
            'transaction_type': 'BUY',
            'shares': holding['shares'], # Initial buy covers the whole holding shares
            'price': holding['avg_cost'], # Use avg_cost as the price for simplicity
            'fee': round(random.uniform(1, 10), 2), # Smaller fee range
            'transaction_time': initial_buy_date,
            'notes': 'Initial purchase'
        })

        # Generate 0-2 additional transactions per holding
        for _ in range(random.randint(0, 2)):
            # Choose transaction type (more likely BUY than SELL)
            transaction_type = 'BUY' if random.random() < 0.7 else 'SELL'
            # Transaction must happen after the initial purchase
            transaction_date = fake.date_between(
                start_date=initial_purchase_date + timedelta(days=1),
                end_date=datetime.now().date() # Ensure transaction date is not in the future
            )

            # Determine shares and price based on type
            if transaction_type == 'BUY':
                # Buy additional shares
                shares = holding['shares'] * random.uniform(0.1, 0.5) # Buy a fraction more
                price = holding['avg_cost'] * random.uniform(0.98, 1.10) # Price might be slightly different
                notes = 'Additional purchase'
            else: # SELL
                # Sell a fraction of the initial shares
                shares = holding['shares'] * random.uniform(0.1, 0.4) # Sell a smaller fraction
                price = holding['avg_cost'] * random.uniform(1.02, 1.20) # Selling price usually higher than avg cost
                notes = 'Profit taking' if random.random() < 0.7 else 'Portfolio rebalance'

            transactions.append({
                'client_id': holding['client_id'],
                'symbol': holding['symbol'],
                'transaction_type': transaction_type,
                'shares': round(shares, 6), # Allow precision
                'price': round(price, 4), # Allow precision
                'fee': round(random.uniform(1, 10), 2),
                'transaction_time': transaction_date,
                'notes': notes
            })
    return transactions

# --- Schema/Loading functions ---
SCHEMA_NAME = os.getenv("AZURE_PG_SCHEMA", "profiles")

def create_tables(conn):
    """Creates database tables within the specified schema if they don't exist."""
    # Use sql.Identifier for safe schema and table names
    schema_id = sql.Identifier(SCHEMA_NAME)

    # SQL statements matching the original definitions
    users_sql = sql.SQL("""
    CREATE TABLE IF NOT EXISTS {schema}.users (
        id SERIAL PRIMARY KEY,
        client_id VARCHAR(50) UNIQUE NOT NULL,
        first_name VARCHAR(50),
        last_name VARCHAR(50),
        risk_profile VARCHAR(20) CHECK (risk_profile IN ('conservative','moderate','aggressive')),
        investment_horizon VARCHAR(20) CHECK (investment_horizon IN ('short','medium','long')),
        last_review_date DATE,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    )
    """).format(schema=schema_id)

    holdings_sql = sql.SQL("""
    CREATE TABLE IF NOT EXISTS {schema}.holdings (
        id SERIAL PRIMARY KEY,
        user_id INTEGER NOT NULL REFERENCES {schema}.users(id) ON DELETE CASCADE,
        symbol VARCHAR(15) NOT NULL, -- Increased length for crypto pairs
        shares NUMERIC(18,8) NOT NULL, -- Increased precision for shares (esp. crypto)
        avg_cost NUMERIC(18,8), -- Increased precision for costs
        current_value NUMERIC(18,4), -- Increased precision for value
        purchase_date DATE DEFAULT CURRENT_DATE,
        last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT positive_shares CHECK (shares >= 0) -- Allow 0 shares temporarily during transactions
    )
    """).format(schema=schema_id)

    transactions_sql = sql.SQL("""
    CREATE TABLE IF NOT EXISTS {schema}.transactions (
        id SERIAL PRIMARY KEY,
        user_id INTEGER NOT NULL REFERENCES {schema}.users(id) ON DELETE CASCADE,
        symbol VARCHAR(15) NOT NULL, -- Increased length for crypto pairs
        transaction_type VARCHAR(4) NOT NULL CHECK (transaction_type IN ('BUY','SELL')),
        shares NUMERIC(18,8) NOT NULL, -- Increased precision
        price NUMERIC(18,8) NOT NULL, -- Increased precision
        fee NUMERIC(10,2) DEFAULT 0,
        transaction_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        notes TEXT
    )
    """).format(schema=schema_id)

    query_logs_sql = sql.SQL("""
    CREATE TABLE IF NOT EXISTS {schema}.query_logs (
        id SERIAL PRIMARY KEY,
        client_id VARCHAR(50) NOT NULL, -- Consider FK to users.client_id if always linked
        query_text TEXT,
        response_summary TEXT,
        query_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    )
    """).format(schema=schema_id)

    # Indexes
    idx_users_client_id_sql = sql.SQL("CREATE INDEX IF NOT EXISTS idx_users_client_id ON {schema}.users(client_id)").format(schema=schema_id)
    idx_holdings_user_id_sql = sql.SQL("CREATE INDEX IF NOT EXISTS idx_holdings_user_id ON {schema}.holdings(user_id)").format(schema=schema_id)
    idx_holdings_symbol_sql = sql.SQL("CREATE INDEX IF NOT EXISTS idx_holdings_symbol ON {schema}.holdings(symbol)").format(schema=schema_id)
    idx_transactions_user_symbol_sql = sql.SQL("CREATE INDEX IF NOT EXISTS idx_transactions_user_symbol ON {schema}.transactions(user_id, symbol)").format(schema=schema_id)
    idx_query_logs_client_id_sql = sql.SQL("CREATE INDEX IF NOT EXISTS idx_query_logs_client_id ON {schema}.query_logs(client_id)").format(schema=schema_id)

    # Update timestamp function and trigger (Optional but good practice)
    update_timestamp_func_sql = sql.SQL("""
    CREATE OR REPLACE FUNCTION {schema}.trigger_set_timestamp()
    RETURNS TRIGGER AS $$
    BEGIN
      NEW.updated_at = NOW();
      RETURN NEW;
    END;
    $$ LANGUAGE plpgsql;
    """).format(schema=schema_id)

    users_trigger_sql = sql.SQL("""
    DO $$ BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'set_timestamp_users') THEN
            CREATE TRIGGER set_timestamp_users
            BEFORE UPDATE ON {schema}.users
            FOR EACH ROW
            EXECUTE PROCEDURE {schema}.trigger_set_timestamp();
        END IF;
    END $$;
    """).format(schema=schema_id)

    holdings_trigger_sql = sql.SQL("""
     DO $$ BEGIN
         IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'set_timestamp_holdings') THEN
             CREATE TRIGGER set_timestamp_holdings
             BEFORE UPDATE ON {schema}.holdings
             FOR EACH ROW
             EXECUTE PROCEDURE {schema}.trigger_set_timestamp();
         END IF;
     END $$;
     """).format(schema=schema_id)


    with conn.cursor() as cur:
        print(f"Creating schema '{SCHEMA_NAME}' if not exists...")
        cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(schema_id))
        print("Creating table 'users'...")
        cur.execute(users_sql)
        print("Creating table 'holdings'...")
        cur.execute(holdings_sql)
        print("Creating table 'transactions'...")
        cur.execute(transactions_sql)
        print("Creating table 'query_logs'...")
        cur.execute(query_logs_sql)
        print("Creating indexes...")
        cur.execute(idx_users_client_id_sql)
        cur.execute(idx_holdings_user_id_sql)
        cur.execute(idx_holdings_symbol_sql)
        cur.execute(idx_transactions_user_symbol_sql)
        cur.execute(idx_query_logs_client_id_sql)
        print("Creating timestamp update function and triggers...")
        cur.execute(update_timestamp_func_sql)
        cur.execute(users_trigger_sql)
        cur.execute(holdings_trigger_sql)
        conn.commit()
    print("Schema and tables checked/created successfully.")


def load_data(conn, clients, holdings, transactions):
    """Loads the generated data into the database tables using parameterized queries."""
    # Use sql.Identifier for schema
    schema_id = sql.Identifier(SCHEMA_NAME)

    with conn.cursor() as cur:
        # Insert clients
        print(f"Inserting {len(clients)} clients...")
        client_insert_sql = sql.SQL("""
            INSERT INTO {schema}.users
                (client_id, first_name, last_name, risk_profile, investment_horizon, last_review_date)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (client_id) DO NOTHING
        """).format(schema=schema_id)
        client_data = [
            (c['client_id'], c['first_name'], c['last_name'], c['risk_profile'],
             c['investment_horizon'], fake.date_between(start_date='-1y', end_date='today'))
            for c in clients
        ]
        cur.executemany(client_insert_sql, client_data)

        # Get user IDs mapping client_id to internal id
        cur.execute(sql.SQL("SELECT id, client_id FROM {schema}.users").format(schema=schema_id))
        user_map = {row[1]: row[0] for row in cur.fetchall()}
        print("User map created.")

        # Insert holdings
        print(f"Inserting {len(holdings)} holdings...")
        holding_insert_sql = sql.SQL("""
            INSERT INTO {schema}.holdings
                (user_id, symbol, shares, avg_cost, current_value, purchase_date)
            VALUES (%s, %s, %s, %s, %s, %s)
        """).format(schema=schema_id)
        holding_data = []
        for h in holdings:
            user_id = user_map.get(h['client_id'])
            if user_id:
                holding_data.append((
                    user_id, h['symbol'], h['shares'], h['avg_cost'],
                    h['current_value'], h['purchase_date']
                ))
            else:
                print(f"Warning: User ID not found for client '{h['client_id']}' while inserting holdings.")
        cur.executemany(holding_insert_sql, holding_data)

        # Insert transactions
        print(f"Inserting {len(transactions)} transactions...")
        transaction_insert_sql = sql.SQL("""
            INSERT INTO {schema}.transactions
                (user_id, symbol, transaction_type, shares, price, fee, transaction_time, notes)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """).format(schema=schema_id)
        transaction_data = []
        for t in transactions:
            user_id = user_map.get(t['client_id'])
            if user_id:
                transaction_data.append((
                    user_id, t['symbol'], t['transaction_type'], t['shares'],
                    t['price'], t['fee'], t['transaction_time'], t['notes']
                ))
            else:
                print(f"Warning: User ID not found for client '{t['client_id']}' while inserting transactions.")
        cur.executemany(transaction_insert_sql, transaction_data)

        conn.commit()
    print("Data loaded successfully.")

# --- Main execution and SQL script generation ---
def generate_sql_script(clients, holdings, transactions):
    """Generates an SQL script file for backup/reference."""
    output_path = PROJECT_ROOT / "data" / "database_export.sql"
    schema_name = SCHEMA_NAME # Use the globally defined schema name

    print(f"Generating SQL script at: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("-- Auto-generated SQL script\n")
        f.write(f"-- Generated on: {datetime.now().isoformat()}\n")
        f.write(f"CREATE SCHEMA IF NOT EXISTS \"{schema_name}\";\n\n") # Quote identifier

        f.write("-- Clients\n")
        for client in clients:
            # Handle potential single quotes in names
            first_name_sql = client['first_name'].replace("'", "''")
            last_name_sql = client['last_name'].replace("'", "''")
            last_review_date = fake.date_between(start_date='-1y', end_date='today')
            f.write(
                f"INSERT INTO \"{schema_name}\".users (client_id, first_name, last_name, risk_profile, investment_horizon, last_review_date) "
                f"VALUES ('{client['client_id']}', '{first_name_sql}', '{last_name_sql}', "
                f"'{client['risk_profile']}', '{client['investment_horizon']}', '{last_review_date}') "
                f"ON CONFLICT (client_id) DO NOTHING;\n"
            )

        f.write("\n-- Holdings\n")
        for holding in holdings:
            # Ensure numeric values are formatted correctly (not as strings with quotes)
            f.write(
                f"INSERT INTO \"{schema_name}\".holdings (user_id, symbol, shares, avg_cost, current_value, purchase_date) "
                f"VALUES ((SELECT id FROM \"{schema_name}\".users WHERE client_id = '{holding['client_id']}'), "
                f"'{holding['symbol']}', {holding['shares']:.8f}, {holding['avg_cost']:.8f}, " # Use precision
                f"{holding['current_value']:.4f}, '{holding['purchase_date']}');\n"
            )

        f.write("\n-- Transactions\n")
        for transaction in transactions:
            # Handle potential single quotes in notes
            notes_sql = transaction['notes'].replace("'", "''")
            # Format timestamp correctly for SQL
            transaction_time_sql = transaction['transaction_time'].isoformat()
            f.write(
                f"INSERT INTO \"{schema_name}\".transactions (user_id, symbol, transaction_type, shares, price, fee, transaction_time, notes) "
                f"VALUES ((SELECT id FROM \"{schema_name}\".users WHERE client_id = '{transaction['client_id']}'), "
                f"'{transaction['symbol']}', '{transaction['transaction_type']}', {transaction['shares']:.8f}, " # Use precision
                f"{transaction['price']:.8f}, {transaction['fee']:.2f}, '{transaction_time_sql}', "
                f"'{notes_sql}');\n"
            )

    print(f"Generated SQL script: {output_path.name}")


def main():
    """Main function to run the data generation and loading process."""
    print("Starting data generation...")
    conn = None # Initialize conn
    try:
        conn = get_db_connection() # Exits if connection fails

        # Create tables
        print("-" * 30)
        create_tables(conn) # Pass connection object
        print("-" * 30)

        # Generate data
        print("Generating synthetic data...")
        clients = generate_clients(20)
        print(f"- Generated {len(clients)} clients.")
        holdings = generate_holdings(clients, 500)
        print(f"- Generated {len(holdings)} holdings.")
        transactions = generate_transactions(holdings)
        print(f"- Generated {len(transactions)} transactions.")
        print("-" * 30)

        # Load data
        print("Loading data into database...")
        load_data(conn, clients, holdings, transactions) # Pass connection object
        print("-" * 30)

        # Generate SQL script backup
        generate_sql_script(clients, holdings, transactions)
        print("-" * 30)
        print("Data generation and loading complete.")

    except psycopg2.Error as db_err:
         print(f"Database Error Encountered: {db_err}", file=sys.stderr)
         if conn:
             try:
                 conn.rollback() # Rollback any partial changes on error
                 print("Database transaction rolled back.")
             except Exception as rb_err:
                 print(f"Error during rollback: {rb_err}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    main()