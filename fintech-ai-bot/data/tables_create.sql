-- Create users table
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
);

-- Create holdings table
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
);

-- Create transactions table
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
);

-- Create query_logs table
CREATE TABLE IF NOT EXISTS profiles.query_logs (
    id SERIAL PRIMARY KEY,
    client_id VARCHAR(50) NOT NULL,
    query_text TEXT NOT NULL,
    response_summary TEXT NOT NULL,
    query_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_users_client_id ON profiles.users(client_id);
CREATE INDEX IF NOT EXISTS idx_holdings_user_id ON profiles.holdings(user_id);
CREATE INDEX IF NOT EXISTS idx_holdings_symbol ON profiles.holdings(symbol);
CREATE INDEX IF NOT EXISTS idx_transactions_user_symbol ON profiles.transactions(user_id, symbol);

-- Create function for automatic current_value updates
CREATE OR REPLACE FUNCTION update_holding_value()
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
$$ LANGUAGE plpgsql;

-- Create trigger for value updates
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger
        WHERE tgname = 'trigger_update_holding_value'
    ) THEN
        CREATE TRIGGER trigger_update_holding_value
        BEFORE INSERT OR UPDATE ON profiles.holdings
        FOR EACH ROW EXECUTE FUNCTION update_holding_value();
    END IF;
END
$$;

-- Create view for portfolio summaries
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
GROUP BY u.id, u.client_id, u.risk_profile;