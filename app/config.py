import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"        # put your PDFs here
CHROMA_DIR = BASE_DIR / "data" / "chroma"  # vector DB lives here

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = 1000       # max characters per text chunk
CHUNK_OVERLAP = 200     # trailing characters (whole lines) shared with the next chunk
MIN_CHUNK_CHARS = 60    # drop tiny chunks (figure captions, stray labels)
TOP_K = 5               # chunks retrieved per question
