from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

load_dotenv()


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)
    neo4j_uri: str = Field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"))
    neo4j_user: str = Field(default_factory=lambda: os.getenv("NEO4J_USER", "neo4j"))
    neo4j_password: str = Field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", "password"))
    nvidia_api_key: str = Field(default_factory=lambda: os.getenv("NVIDIA_API_KEY", ""))
    nim_base_url: str = "https://nvidia.com"
    nim_model: str = "meta/muse-glimmer-30b"
    proof_directory: Path = Field(default_factory=lambda: Path(os.getenv("PROOF_DIRECTORY", "./dispute_proofs")))
    media_directory: Path = Field(default_factory=lambda: Path(os.getenv("MEDIA_DIRECTORY", "./chore_media")))
    max_upload_bytes: int = Field(default_factory=lambda: int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024))))


settings = Settings()
