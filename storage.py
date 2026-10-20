"""Persistent local data and trained models for the desktop Streamlit app."""
from pathlib import Path
import json
import joblib
from engine import OPERATORS, load_draws

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'local_data'
MODEL_DIR = ROOT / 'saved_models'


def data_path(operator):
    return DATA_DIR / OPERATORS[operator]


def has_data(operator):
    return data_path(operator).is_file()


def get_data(operator):
    return load_draws(data_path(operator))


def save_data(operator, dataframe):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(data_path(operator), index=False)


def model_paths(operator, prize):
    slug = OPERATORS[operator].replace('_draws.csv', '')
    stem = f'{slug}_{prize}'
    return MODEL_DIR / f'{stem}.joblib', MODEL_DIR / f'{stem}.json'


def save_model(operator, prize, models, metadata):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_file, meta_file = model_paths(operator, prize)
    joblib.dump(models, model_file)
    meta_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')


def get_model(operator, prize):
    model_file, meta_file = model_paths(operator, prize)
    if not model_file.exists() or not meta_file.exists():
        return None, None
    # Only load models generated locally by this app. Never open untrusted joblib files.
    return joblib.load(model_file), json.loads(meta_file.read_text(encoding='utf-8'))
