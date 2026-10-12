"""Keep tokenizer tests offline by using the course's checked-in vocabulary cache."""
import os
from pathlib import Path
os.environ['TIKTOKEN_CACHE_DIR'] = str(Path(__file__).resolve().parents[1] / 'data' / 'tiktoken-cache')
