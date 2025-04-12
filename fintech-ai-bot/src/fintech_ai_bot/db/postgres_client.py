import psycopg2
from psycopg2 import sql, pool, OperationalError, InterfaceError, DatabaseError, DataError, IntegrityError, ProgrammingError
from typing import Dict, Optional, List, Any, Tuple
from contextlib import contextmanager
import time
from fintech_ai_bot.config import settings
from fintech_ai_bot.utils import get_logger

logger = get_logger(__name__)

class PostgresClient:
    """Client for interacting with the Azure PostgreSQL database."""
    _pool = None

    def __init__(self, min_conn=1, max_conn=5):
        if not settings.db_connection_string:
            logger.critical("Database connection string is not configured.")
            raise ValueError("Database connection string is missing in settings.")

        if PostgresClient._pool is None:
            logger.info(f"Initializing PostgreSQL connection pool for {settings.azure_pg_host}...")
            try:
                PostgresClient._pool = psycopg2.pool.SimpleConnectionPool(
                    min_conn,
                    max_conn,
                    dsn=str(settings.db_connection_string), # Use DSN from settings
                    connect_timeout=5,
                    options=f'-c search_path={settings.azure_pg_schema},public' # Set schema context
                )
                logger.info("PostgreSQL connection pool initialized successfully.")
            except OperationalError as e:
                logger.critical(f"Failed to initialize PostgreSQL connection pool: {e}", exc_info=True)
                raise
            except Exception as e:
                logger.critical(f"An unexpected error occurred during pool initialization: {e}", exc_info=True)
                raise

    @contextmanager
    def get_connection(self):
        """Provides a connection from the pool."""
        if self._pool is None:
            logger.error("Connection pool is not initialized.")
            raise RuntimeError("Connection pool not available.")
        conn = None
        try:
            conn = self._pool.getconn()
            yield conn
            conn.commit() # Commit changes by default when context exits cleanly
        except DatabaseError as e:
            logger.error(f"Database error occurred: {e}", exc_info=True)
            if conn:
                conn.rollback() # Rollback on error
            raise # Re-raise the exception
        except Exception as e:
            logger.error(f"Unexpected error with DB connection: {e}", exc_info=True)
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                self._pool.putconn(conn)

    @contextmanager
    def get_cursor(self, conn):
        """Provides a cursor from a connection."""
        cursor = None
        try:
            cursor = conn.cursor()
            yield cursor
        finally:
            if cursor:
                cursor.close()

    def _execute(self, query: sql.SQL | str, params: Tuple | None = None, fetch: bool = True, commit_in_context: bool = False) -> Optional[List[Tuple]]:
        """Executes a query with retry logic using the connection pool."""
        last_exception = None
        for attempt in range(settings.db_max_retries):
            try:
                with self.get_connection() as conn:
                    # If commit_in_context is True, commit happens automatically in get_connection context
                    # If False, we assume read operation or commit is handled elsewhere
                    with self.get_cursor(conn) as cur:
                        cur.execute(query, params or ())
                        if fetch:
                            return cur.fetchall()
                        else:
                            return None # Indicate success for non-fetch operations
            except (OperationalError, InterfaceError) as e: # Connection errors
                last_exception = e
                wait_time = settings.db_retry_delay * (2 ** attempt)
                logger.warning(f"DB connection error (Attempt {attempt + 1}/{settings.db_max_retries}): {e}. Retrying in {wait_time:.2f}s...")
                if attempt < settings.db_max_retries - 1:
                    time.sleep(wait_time)
                else:
                    logger.error(f"DB operation failed after {settings.db_max_retries} retries.")
                    raise last_exception
            except (DataError, IntegrityError, ProgrammingError) as e: # Query/Data errors - don't retry
                logger.error(f"Database query/data error: {e}", exc_info=True)
                raise
            except DatabaseError as e: # Other DB errors - potentially retry
                last_exception = e
                wait_time = settings.db_retry_delay * (2 ** attempt)
                logger.warning(f"Database error (Attempt {attempt + 1}/{settings.db_max_retries}): {e}. Retrying in {wait_time:.2f}s...")
                if attempt < settings.db_max_retries - 1:
                    time.sleep(wait_time)
                else:
                    logger.error(f"DB operation failed after {settings.db_max_retries} retries.")
                    raise last_exception
            except Exception as e: # Unexpected errors
                logger.error(f"Unexpected error during DB execution: {e}", exc_info=True)
                raise

        # Should not be reached if retries fail, as exception is raised
        return None

    def get_client_portfolio(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a client's portfolio summary."""
        if not client_id or not isinstance(client_id, str):
            logger.warning(f"Invalid client_id provided: {client_id}")
            return None

        query = sql.SQL("""
        SELECT
            u.id, -- Include user id for potential future use
            u.risk_profile,
            COALESCE(SUM(h.current_value), 0) as total_value,
            jsonb_agg(jsonb_build_object(
                'symbol', h.symbol,
                'shares', h.shares,
                'avg_cost', h.avg_cost,
                'current_value', h.current_value,
                'purchase_date', h.purchase_date,
                'performance', CASE
                    WHEN h.shares = 0 OR h.avg_cost = 0 THEN 0
                    ELSE ROUND((h.current_value - (h.shares * h.avg_cost)) / (h.shares * h.avg_cost) * 100, 2)
                END
            )) FILTER (WHERE h.symbol IS NOT NULL) as holdings
        FROM users u
        LEFT JOIN holdings h ON u.id = h.user_id -- Use LEFT JOIN to include users with no holdings
        WHERE u.client_id = %s
        GROUP BY u.id, u.risk_profile -- Group by user id and risk profile
        LIMIT 1 -- Ensure only one row per client_id
        """)
        # Note: search_path is set in the connection pool options

        try:
            results = self._execute(query, (client_id,), fetch=True)
            if results:
                result = results[0]
                logger.info(f"Successfully retrieved portfolio for client {client_id}")
                return {
                    'user_db_id': result[0], # Internal DB id
                    'risk_profile': result[1] or 'Not specified',
                    'total_value': float(result[2]),
                    'holdings': result[3] or [], # Ensure holdings is always a list
                    'client_id': client_id # Add client_id back for consistency if needed elsewhere
                }
            else:
                logger.warning(f"No portfolio data found for client {client_id}")
                return None
        except Exception as e:
            logger.error(f"Error fetching portfolio for client {client_id}: {e}", exc_info=True)
            return None

    def log_client_query(self, client_id: str, query_text: str, response_summary: str) -> bool:
        """Logs a client query and response summary."""
        if not all([client_id, query_text, response_summary]):
            logger.warning("Attempted to log incomplete client query data.")
            return False

        query = sql.SQL("""
        INSERT INTO query_logs (client_id, query_text, response_summary, query_timestamp)
        VALUES (%s, %s, %s, NOW())
        """)

        try:
            # Use commit_in_context=True if get_connection handles commit
            self._execute(
                query,
                (client_id, query_text[:1000], response_summary[:2000]),
                fetch=False,
                commit_in_context=True # Let context manager handle commit
            )
            logger.info(f"Logged query for client {client_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to log query for client {client_id}: {e}", exc_info=True)
            return False

    def close_pool(self):
        """Closes the connection pool."""
        if self._pool:
            logger.info("Closing PostgreSQL connection pool.")
            self._pool.closeall()
            PostgresClient._pool = None
            logger.info("PostgreSQL connection pool closed.")

# Example Usage (within Streamlit app or Orchestrator):
# db_client = PostgresClient()
# portfolio = db_client.get_client_portfolio("CLIENT101")
# if portfolio:
#     print(portfolio)
# db_client.log_client_query("CLIENT101", "Test query", "Test response")
# Remember to call db_client.close_pool() when the application shuts down if necessary.