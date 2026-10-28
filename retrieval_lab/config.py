import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    supabase_url: str
    supabase_key: str
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    chunk_size: int = 500
    chunk_overlap: int = 50
    rrf_k: int = 60
    max_upload_bytes: int = MAX_UPLOAD_BYTES


def load_settings() -> Settings:
    load_dotenv(ROOT / ".env")
    missing = [
        name
        for name in ("OPENAI_API_KEY", "SUPABASE_URL", "SUPABASE_KEY")
        if not os.environ.get(name)
    ]
    if missing:
        raise RuntimeError(
            "Missing environment variables: " + ", ".join(missing)
        )
    return Settings(
        openai_api_key=os.environ["OPENAI_API_KEY"],
        supabase_url=os.environ["SUPABASE_URL"],
        supabase_key=os.environ["SUPABASE_KEY"],
    )
