from __future__ import annotations
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.api.endpoints import router
from app.core.config import settings
from app.core.database import create_driver, initialize_schema

@asynccontextmanager
async def lifespan(app: FastAPI):
    driver = create_driver()
    try:
        driver.verify_connectivity()
        initialize_schema(driver)
        app.state.driver = driver
        yield
    finally:
        driver.close()

app = FastAPI(title="HelperChore", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"], allow_credentials=True, allow_methods=["GET", "POST"], allow_headers=["*"])
app.include_router(router)
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")
