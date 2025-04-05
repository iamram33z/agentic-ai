import psycopg2
from psycopg2 import sql, OperationalError, InterfaceError, DatabaseError, DataError, IntegrityError, ProgrammingError
from typing import Dict, Optional, List
import os
from dotenv import load_dotenv
from datetime import datetime
from utils import get_logger, retry_db_operation

load_dotenv()
logger = get_logger("PostgresClient")


def get_db_connection():
    """Create and return a new database connection"""
    return psycopg2.connect(
        host=os.getenv("AZURE_PG_HOST"),
        database=os.getenv("AZURE_PG_DB"),
        user=os.getenv("AZURE_PG_USER"),
        password=os.getenv("AZURE_PG_PASSWORD"),
        sslmode=os.getenv("AZURE_PG_SSL", "require"),
        connect_timeout=5
    )


class AzurePostgresClient:
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self.schema = os.getenv("AZURE_PG_SCHEMA", "profiles")

    def _execute_query(self, query, params=None, commit=False):
        """Execute a query with automatic connection handling"""

        @retry_db_operation(max_retries=self.max_retries)
        def execute():
            conn = None
            cur = None  # Initialize cur
            try:
                conn = get_db_connection()
                cur = conn.cursor()  # Create cursor
                cur.execute(query, params or ())
                if commit:
                    conn.commit()
                    return None  # Return None for commit operations
                else:
                    result = cur.fetchall()  # Fetch all results before closing
                    return result
            except (OperationalError, InterfaceError) as e:
                logger.error(f"Database connection error: {str(e)}", exc_info=True)
                raise
            except (DataError, IntegrityError, ProgrammingError) as e:
                logger.error(f"Database query error: {str(e)}", exc_info=True)
                raise
            except DatabaseError as e:
                logger.error(f"General database error: {str(e)}", exc_info=True)
                raise
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}", exc_info=True)
                raise
            finally:
                if cur:
                    cur.close()  # Close the cursor first
                if conn:
                    conn.close()  # Close the connection

        if commit:
            execute()
            return None
        else:
            return execute()

    def get_client_portfolio(self, client_id: str) -> Optional[Dict]:
        """Retrieve client portfolio with proper error handling"""
        logger.debug(f"Received client_id: {client_id}")

        if not client_id or not isinstance(client_id, str):
            logger.warning(f"Invalid client_id: {client_id}")
            return None

        query = sql.SQL("""
        SELECT
            jsonb_agg(jsonb_build_object(
                'symbol', h.symbol,
                'shares', h.shares,
                'avg_cost', h.avg_cost,
                'current_value', h.current_value,
                'purchase_date', h.purchase_date,
                'performance', CASE
                    WHEN h.shares = 0 OR h.avg_cost = 0 THEN 0
                    ELSE ROUND((h.current_value - (h.shares * h.avg_cost)) /
                         (h.shares * h.avg_cost) * 100, 2)
                END
            )) FILTER (WHERE h.symbol IS NOT NULL) as holdings,
            COALESCE(SUM(h.current_value), 0) as total_value,
            u.risk_profile
        FROM {schema}.holdings h
        JOIN {schema}.users u ON u.id = h.user_id
        WHERE u.client_id = %s
        GROUP BY u.risk_profile
        """).format(schema=sql.Identifier(self.schema))

        try:
            start_time = datetime.now()
            results = self._execute_query(query, (client_id,))

            if results:
                result = results[0]
                logger.info(f"Retrieved portfolio in {(datetime.now() - start_time).total_seconds():.2f}s")
                return {
                    'holdings': result[0] or [],
                    'total_value': float(result[1]),
                    'risk_profile': result[2] or 'Not specified'
                }
            else:
                logger.warning(f"No portfolio found for client {client_id}")
                return None

        except Exception as e:
            logger.error(f"Error fetching portfolio: {str(e)}", exc_info=True)
            return None

    def log_client_query(self, client_id: str, query: str, response: str) -> bool:
        """Log client interactions with proper validation"""
        if not all([client_id, query, response]):
            return False

        insert_query = sql.SQL("""
        INSERT INTO {schema}.query_logs 
        (client_id, query_text, response_summary, query_timestamp)
        VALUES (%s, %s, %s, NOW())
        """).format(schema=sql.Identifier(self.schema))

        try:
            self._execute_query(
                insert_query,
                (client_id, query[:500], response[:1000]),
                commit=True
            )
            return True
        except Exception as e:
            logger.error(f"Failed to log query: {str(e)}")
            return False


"""if __name__ == "__main__":
    client = AzurePostgresClient()

    # Test with valid client
    portfolio = client.get_client_portfolio("CLIENT101")
    print(f"Portfolio: {portfolio}")

    # Test with another valid client
    portfolio = client.get_client_portfolio("CLIENT102")
    print(f"Another valid client test: {portfolio}")

    # Test query logging
    success = client.log_client_query(
        "CLIENT101",
        "What's my portfolio value?",
        "Your portfolio is worth $10,000"
    )
    print(f"Logging test: {'Success' if success else 'Failed'}")"""