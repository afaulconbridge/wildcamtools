import logging
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import sqlmodel
from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel

logger = logging.getLogger(__name__)


@event.listens_for(Engine, "connect")
def _set_sqlite_pragma(dbapi_connection: Any, connection_record: Any) -> None:
    dialect = connection_record.dbapi_connection
    if "sqlite" in str(dialect):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def create_engine_and_tables(connection_string: str) -> Engine:
    engine = sqlmodel.create_engine(connection_string)
    logger.info("Creating database tables")
    SQLModel.metadata.create_all(engine)
    return engine


@contextmanager
def get_session(engine: Engine) -> Generator[Session]:
    session = Session(engine)
    try:
        yield session
    finally:
        session.close()
