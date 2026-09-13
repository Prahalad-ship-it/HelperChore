from __future__ import annotations

from collections.abc import Callable
from typing import Any
from fastapi import HTTPException, Request
from neo4j import Driver, GraphDatabase, ManagedTransaction
from .config import settings


def create_driver() -> Driver:
    return GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password), max_connection_pool_size=50)


def initialize_schema(driver: Driver) -> None:
    queries = (
        "CREATE CONSTRAINT resident_id_unique IF NOT EXISTS FOR (n:Resident) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT task_id_unique IF NOT EXISTS FOR (n:Task) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT skill_name_unique IF NOT EXISTS FOR (n:Skill) REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT report_id_unique IF NOT EXISTS FOR (n:Report) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT coupon_id_unique IF NOT EXISTS FOR (n:Coupon) REQUIRE n.id IS UNIQUE",
        "CREATE POINT INDEX resident_location IF NOT EXISTS FOR (n:Resident) ON (n.location)",
    )
    with driver.session() as session:
        for query in queries:
            session.run(query).consume()


def driver_from_request(request: Request) -> Driver:
    driver = getattr(request.app.state, "driver", None)
    if driver is None:
        raise HTTPException(status_code=503, detail="Database is not initialized")
    return driver


def read(driver: Driver, callback: Callable[..., Any], **params: Any) -> Any:
    try:
        with driver.session() as session:
            return session.execute_read(callback, **params)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Database read failed") from exc


def write(driver: Driver, callback: Callable[..., Any], **params: Any) -> Any:
    try:
        with driver.session() as session:
            return session.execute_write(callback, **params)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Database transaction failed") from exc


def require_record(record: Any, detail: str = "Record not found") -> dict[str, Any]:
    if record is None:
        raise HTTPException(status_code=404, detail=detail)
    return record.data() if hasattr(record, "data") else record
