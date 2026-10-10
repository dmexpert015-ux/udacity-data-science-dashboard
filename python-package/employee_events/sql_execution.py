"""Small SQLite execution helpers with deterministic connection cleanup."""

from contextlib import closing
from sqlite3 import connect
from pathlib import Path
from functools import wraps
import pandas as pd

db_path = Path(__file__).resolve().parent / "employee_events.db"


class QueryMixin:
    """Execute SQL through a short-lived connection to the bundled database."""

    def pandas_query(self, sql_query, params=()):
        with closing(connect(db_path)) as connection:
            return pd.read_sql_query(sql_query, connection, params=params)

    def query(self, sql_query, params=()):
        with closing(connect(db_path)) as connection:
            return connection.execute(sql_query, params).fetchall()


def query(func):
    """Execute the SQL string returned by a decorated method."""

    @wraps(func)
    def run_query(*args, **kwargs):
        with closing(connect(db_path)) as connection:
            return connection.execute(func(*args, **kwargs)).fetchall()

    return run_query
