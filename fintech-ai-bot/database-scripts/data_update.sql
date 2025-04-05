-- Users Table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    client_id VARCHAR(50) UNIQUE NOT NULL,
    risk_profile VARCHAR(20) CHECK (risk_profile IN ('conservative', 'moderate', 'aggressive'))
);

-- Holdings Table
CREATE TABLE holdings (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    symbol VARCHAR(10),
    shares NUMERIC(10, 2)
);

-- Insert Sample Data
INSERT INTO users (client_id, risk_profile) VALUES
('CLIENT001', 'moderate'),
('CLIENT002', 'aggressive'),
('CLIENT003', 'conservative');

INSERT INTO holdings (user_id, symbol, shares) VALUES
(1, 'AAPL', 50),
(1, 'MSFT', 30),
(2, 'TSLA', 100),
(3, 'VTI', 200);