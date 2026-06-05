"""Runtime config for the retrieval POC.

Reads env vars once at import. POC-only; no live mutation surface
(unlike chatbot/config.py which is mutable).
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
CORPUS_DIR = Path(os.getenv("CORPUS_DIR", "./corpus"))
INDEX_DB_PATH = Path(os.getenv("INDEX_DB_PATH", "./data/index.db"))
MAX_PICKS = int(os.getenv("MAX_PICKS", "3"))
RETRIEVE_K = int(os.getenv("RETRIEVE_K", "10"))

# Make paths absolute relative to the package dir (so they work no
# matter the caller's CWD).
_PKG_DIR = Path(__file__).parent
if not CORPUS_DIR.is_absolute():
    CORPUS_DIR = (_PKG_DIR / CORPUS_DIR).resolve()
if not INDEX_DB_PATH.is_absolute():
    INDEX_DB_PATH = (_PKG_DIR / INDEX_DB_PATH).resolve()
