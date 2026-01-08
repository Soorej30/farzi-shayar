import sys
from pathlib import Path
import types
import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

nltk_mod = types.ModuleType("nltk")
tokenize_mod = types.ModuleType("nltk.tokenize")
def word_tokenize(text):
    return str(text).split()

tokenize_mod.word_tokenize = word_tokenize
nltk_mod.tokenize = tokenize_mod
nltk_mod.download = lambda *args, **kwargs: None
sys.modules['nltk'] = nltk_mod
sys.modules['nltk.tokenize'] = tokenize_mod

from src.preprocessing import preprocessing

def test_preprocessing_writes_csv(tmp_path):
    raw_dir = repo_root / 'data' / 'raw'
    processed_dir = repo_root / 'data' / 'processed'
    raw_dir.mkdir(parents=True, exist_ok=True)
    if processed_dir.exists():
        for p in processed_dir.glob('*'):
            p.unlink()
    csv_path = raw_dir / 'shayari_db.csv'
    pd.DataFrame([{'AUTHOR': 'a', 'POEM': 'Hello world'}]).to_csv(csv_path, index=False)

    preprocessing()

    out_file = processed_dir / 'processed_poetry.csv'
    assert out_file.exists(), f"Expected processed CSV at {out_file}"
    df = pd.read_csv(out_file)
    assert 'CLEAN_POEM' in df.columns
    assert 'TOKENS' in df.columns