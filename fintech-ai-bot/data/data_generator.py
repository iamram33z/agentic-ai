import psycopg2
import random
from datetime import datetime, timedelta
import numpy as np
from faker import Faker
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Faker
fake = Faker()

# Database connection
def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("AZURE_PG_HOST"),
        database=os.getenv("AZURE_PG_DB"),
        user=os.getenv("AZURE_PG_USER"),
        password=os.getenv("AZURE_PG_PASSWORD"),
        sslmode=os.getenv("AZURE_PG_SSL", "require")
    )

# Asset configuration
ASSET_TYPES = {
    'conservative': {
        'etfs': ['BND', 'VTI', 'VXUS', 'GLD', 'IEF'],
        'stocks': ['JNJ', 'PG', 'KO', 'PEP', 'WMT', 'MCD', 'T', 'VZ'],
        'crypto': []
    },
    'moderate': {
        'etfs': ['VTI', 'VXUS', 'QQQ', 'IWM', 'XLF'],
        'stocks': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM'],
        'crypto': ['BTC-USD', 'ETH-USD']
    },
    'aggressive': {
        'etfs': ['ARKK', 'SOXX', 'IBB'],
        'stocks': ['TSLA', 'NVDA', 'AMD', 'SNOW', 'MRNA', 'SHOP'],
        'crypto': ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'DOT-USD']
    }
}

ASSET_PRICES = {
    # ETFs
    'BND': 72.50, 'VTI': 220.00, 'VXUS': 55.30, 'GLD': 180.40, 'IEF': 85.20,
    'QQQ': 350.20, 'IWM': 190.75, 'XLF': 35.40, 'ARKK': 45.20, 'SOXX': 480.30,
    'IBB': 135.60,
    # Stocks
    'JNJ': 150.75, 'PG': 140.30, 'KO': 60.25, 'PEP': 170.40, 'WMT': 145.60,
    'MCD': 250.30, 'T': 18.75, 'VZ': 40.20, 'AAPL': 175.25, 'MSFT': 420.50,
    'GOOGL': 1250.75, 'AMZN': 150.30, 'META': 300.40, 'TSLA': 250.30,
    'NVDA': 600.40, 'AMD': 110.25, 'SNOW': 160.75, 'MRNA': 120.40, 'SHOP': 70.30,
    'JPM': 150.20,
    # Crypto
    'BTC-USD': 62000.00, 'ETH-USD': 3400.00, 'SOL-USD': 120.00,
    'ADA-USD': 0.45, 'DOT-USD': 6.80
}

TRANSACTION_TYPES = ['BUY', 'SELL']

def generate_clients(num_clients=20):
    """Generate client data with realistic risk profiles"""
    clients = []
    risk_distribution = ['conservative'] * 5 + ['moderate'] * 10 + ['aggressive'] * 5

    for i in range(num_clients):
        client_id = f"CLIENT{100 + i + 1}"
        risk = risk_distribution[i] if i < len(risk_distribution) else random.choice(
            ['conservative', 'moderate', 'aggressive'])
        clients.append({
            'client_id': client_id,
            'first_name': fake.first_name(),
            'last_name': fake.last_name(),
            'risk_profile': risk,
            'investment_horizon': random.choice(['short', 'medium', 'long'])
        })
    return clients

def generate_holdings(clients, min_holdings=500):
    """Generate portfolio holdings for clients"""
    holdings = []
    holdings_per_client = max(min_holdings // len(clients), 4)

    for client in clients:
        risk = client['risk_profile']
        assets = ASSET_TYPES[risk]

        # Determine asset allocation based on risk
        if risk == 'conservative':
            allocation = {'etfs': 0.6, 'stocks': 0.4, 'crypto': 0.0}
        elif risk == 'moderate':
            allocation = {'etfs': 0.5, 'stocks': 0.45, 'crypto': 0.05}
        else:
            allocation = {'etfs': 0.3, 'stocks': 0.5, 'crypto': 0.2}

        # Generate holdings
        for _ in range(holdings_per_client):
            # Choose asset type based on allocation
            asset_type = np.random.choice(
                ['etfs', 'stocks', 'crypto'],
                p=[allocation['etfs'], allocation['stocks'], allocation['crypto']]
            )
            symbol = random.choice(assets[asset_type])

            # Generate realistic share amounts
            if asset_type == 'crypto':
                shares = round(random.uniform(0.1, 10.0), 4)
            elif symbol in ['BND', 'VTI', 'QQQ']:
                shares = random.randint(10, 200)
            else:
                shares = random.randint(1, 100)

            # Generate cost basis with some variance
            avg_cost = ASSET_PRICES[symbol] * random.uniform(0.8, 1.2)
            current_value = shares * ASSET_PRICES[symbol] * random.uniform(0.9, 1.1)
            purchase_date = fake.date_between(start_date='-5y', end_date='today')

            holdings.append({
                'client_id': client['client_id'],
                'symbol': symbol,
                'shares': shares,
                'avg_cost': round(avg_cost, 2),
                'current_value': round(current_value, 2),
                'purchase_date': purchase_date
            })

    return holdings

def generate_transactions(holdings):
    """Generate transaction history for holdings"""
    transactions = []

    for holding in holdings:
        # Generate 1-3 transactions per holding
        for i in range(random.randint(1, 3)):
            transaction_date = fake.date_between(
                start_date=holding['purchase_date'],
                end_date='today'
            )

            # For BUY transactions
            transactions.append({
                'client_id': holding['client_id'],
                'symbol': holding['symbol'],
                'transaction_type': 'BUY',
                'shares': holding['shares'] * random.uniform(0.3, 1.0),
                'price': holding['avg_cost'] * random.uniform(0.95, 1.05),
                'fee': round(random.uniform(5, 15), 2),
                'transaction_time': transaction_date,
                'notes': 'Initial purchase' if i == 0 else 'Additional purchase'
            })

            # Occasionally generate SELL transactions
            if random.random() < 0.3:
                transactions.append({
                    'client_id': holding['client_id'],
                    'symbol': holding['symbol'],
                    'transaction_type': 'SELL',
                    'shares': holding['shares'] * random.uniform(0.1, 0.5),
                    'price': holding['avg_cost'] * random.uniform(1.05, 1.25),
                    'fee': round(random.uniform(5, 15), 2),
                    'transaction_time': fake.date_between(
                        start_date=transaction_date,
                        end_date='today'
                    ),
                    'notes': 'Profit taking' if random.random() < 0.7 else 'Portfolio rebalance'
                })

    return transactions

def create_tables(conn):
    """Create database tables if they don't exist"""
    with conn.cursor() as cur:
        # Create schema if not exists
        cur.execute("CREATE SCHEMA IF NOT EXISTS profiles")

        # Create users table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS profiles.users (
            id SERIAL PRIMARY KEY,
            client_id VARCHAR(50) UNIQUE NOT NULL,
            first_name VARCHAR(50),
            last_name VARCHAR(50),
            risk_profile VARCHAR(20) CHECK (risk_profile IN ('conservative','moderate','aggressive')),
            investment_horizon VARCHAR(20) CHECK (investment_horizon IN ('short','medium','long')),
            last_review_date DATE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # Create holdings table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS profiles.holdings (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES profiles.users(id) ON DELETE CASCADE,
            symbol VARCHAR(10) NOT NULL,
            shares NUMERIC(10,4) NOT NULL,
            avg_cost NUMERIC(10,2),
            current_value NUMERIC(12,2),
            purchase_date DATE DEFAULT CURRENT_DATE,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT positive_shares CHECK (shares > 0)
        )
        """)

        # Create transactions table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS profiles.transactions (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES profiles.users(id) ON DELETE CASCADE,
            symbol VARCHAR(10) NOT NULL,
            transaction_type VARCHAR(4) CHECK (transaction_type IN ('BUY','SELL')),
            shares NUMERIC(10,4) NOT NULL,
            price NUMERIC(10,2) NOT NULL,
            fee NUMERIC(10,2) DEFAULT 0,
            transaction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            notes TEXT
        )
        """)

        # Create query_logs table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS profiles.query_logs (
            id SERIAL PRIMARY KEY,
            client_id VARCHAR(50) NOT NULL,
            query_text TEXT NOT NULL,
            response_summary TEXT NOT NULL,
            query_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # Create indexes
        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_users_client_id ON profiles.users(client_id)
        """)
        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_holdings_user_id ON profiles.holdings(user_id)
        """)
        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_holdings_symbol ON profiles.holdings(symbol)
        """)
        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_transactions_user_symbol ON profiles.transactions(user_id, symbol)
        """)

        # Create function for automatic current_value updates
        cur.execute("""
        CREATE OR REPLACE FUNCTION profiles.update_holding_value()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.current_value = NEW.shares * (
                SELECT price FROM (
                    VALUES 
                        ('AAPL', 175.25), ('MSFT', 420.50), ('TSLA', 250.30),
                        ('BTC-USD', 62000.00), ('ETH-USD', 3400.00)
                ) AS prices(sym, price)
                WHERE sym = NEW.symbol
                LIMIT 1
            );
            NEW.last_updated = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
        """)

        # Create trigger for value updates
        cur.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_trigger 
                WHERE tgname = 'trigger_update_holding_value'
            ) THEN
                CREATE TRIGGER trigger_update_holding_value
                BEFORE INSERT OR UPDATE ON profiles.holdings
                FOR EACH ROW EXECUTE FUNCTION profiles.update_holding_value();
            END IF;
        END
        $$
        """)

        # Create view for portfolio summaries
        cur.execute("""
        CREATE OR REPLACE VIEW profiles.portfolio_summary AS
        SELECT 
            u.client_id,
            u.risk_profile,
            COUNT(h.id) AS holding_count,
            SUM(h.current_value) AS total_value,
            ROUND(SUM(h.current_value * CASE 
                WHEN h.symbol IN ('BTC-USD','ETH-USD','SOL-USD') THEN 0.2
                WHEN h.symbol IN ('ARKK','SOXX','IBB') THEN 0.15
                ELSE 0.05
            END), 2) AS estimated_risk
        FROM profiles.users u
        LEFT JOIN profiles.holdings h ON u.id = h.user_id
        GROUP BY u.id, u.client_id, u.risk_profile
        """)

        conn.commit()

def load_data(conn, clients, holdings, transactions):
    """Load generated data into database"""
    with conn.cursor() as cur:
        # Insert clients
        for client in clients:
            cur.execute("""
            INSERT INTO profiles.users 
                (client_id, first_name, last_name, risk_profile, investment_horizon, last_review_date)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (client_id) DO NOTHING
            """, (
                client['client_id'],
                client['first_name'],
                client['last_name'],
                client['risk_profile'],
                client['investment_horizon'],
                fake.date_between(start_date='-1y', end_date='today')
            ))

        # Get user IDs
        cur.execute("SELECT id, client_id FROM profiles.users")
        user_map = {row[1]: row[0] for row in cur.fetchall()}

        # Insert holdings
        for holding in holdings:
            user_id = user_map.get(holding['client_id'])
            if user_id:
                cur.execute("""
                INSERT INTO profiles.holdings 
                    (user_id, symbol, shares, avg_cost, current_value, purchase_date)
                VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    user_id,
                    holding['symbol'],
                    holding['shares'],
                    holding['avg_cost'],
                    holding['current_value'],
                    holding['purchase_date']
                ))

        # Insert transactions
        for transaction in transactions:
            user_id = user_map.get(transaction['client_id'])
            if user_id:
                cur.execute("""
                INSERT INTO profiles.transactions 
                    (user_id, symbol, transaction_type, shares, price, fee, transaction_time, notes)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    user_id,
                    transaction['symbol'],
                    transaction['transaction_type'],
                    transaction['shares'],
                    transaction['price'],
                    transaction['fee'],
                    transaction['transaction_time'],
                    transaction['notes']
                ))

        conn.commit()

def main():
    print("Starting data generation...")
    conn = get_db_connection()

    try:
        # Create tables
        create_tables(conn)

        # Generate data
        clients = generate_clients(20)
        holdings = generate_holdings(clients, 500)
        transactions = generate_transactions(holdings)

        # Load data
        load_data(conn, clients, holdings, transactions)
        print(f"Successfully loaded:")
        print(f"- {len(clients)} clients")
        print(f"- {len(holdings)} holdings")
        print(f"- {len(transactions)} transactions")

        # Generate SQL script backup
        generate_sql_script(clients, holdings, transactions)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

def generate_sql_script(clients, holdings, transactions):
    """Generate SQL script for backup/reference"""
    with open('database_export.sql', 'w') as f:
        f.write("-- Auto-generated SQL script\n")
        f.write("-- Create schema\n")
        f.write("CREATE SCHEMA IF NOT EXISTS profiles;\n\n")

        f.write("-- Clients\n")
        for client in clients:
            f.write(
                f"INSERT INTO profiles.users (client_id, first_name, last_name, risk_profile, investment_horizon, last_review_date) "
                f"VALUES ('{client['client_id']}', '{client['first_name']}', '{client['last_name']}', "
                f"'{client['risk_profile']}', '{client['investment_horizon']}', '{fake.date_between(start_date='-1y', end_date='today')}');\n"
            )

        f.write("\n-- Holdings\n")
        for holding in holdings:
            f.write(
                f"INSERT INTO profiles.holdings (user_id, symbol, shares, avg_cost, current_value, purchase_date) "
                f"VALUES ((SELECT id FROM profiles.users WHERE client_id = '{holding['client_id']}'), "
                f"'{holding['symbol']}', {holding['shares']}, {holding['avg_cost']}, "
                f"{holding['current_value']}, '{holding['purchase_date']}');\n"
            )

        f.write("\n-- Transactions\n")
        for transaction in transactions:
            f.write(
                f"INSERT INTO profiles.transactions (user_id, symbol, transaction_type, shares, price, fee, transaction_time, notes) "
                f"VALUES ((SELECT id FROM profiles.users WHERE client_id = '{transaction['client_id']}'), "
                f"'{transaction['symbol']}', '{transaction['transaction_type']}', {transaction['shares']}, "
                f"{transaction['price']}, {transaction['fee']}, '{transaction['transaction_time']}', "
                f"'{transaction['notes']}');\n"
            )

        print("Generated SQL script: database_export.sql")

if __name__ == "__main__":
    main()