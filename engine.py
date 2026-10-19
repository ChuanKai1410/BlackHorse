"""Four-digit educational forecast engine. No real lottery predictability is implied."""
from __future__ import annotations
import io
from dataclasses import dataclass
import numpy as np
import pandas as pd

OPERATORS = {
    "Magnum 4D": "magnum_draws.csv",
    "Sports Toto": "sportstoto_draws.csv",
    "Da Ma Cai": "damacai_draws.csv",
}
BASE_URL = "https://raw.githubusercontent.com/deadboy18/malaysia-4d/main/data/"
PRIZES = {"1st Prize": "prize_1", "2nd Prize": "prize_2", "3rd Prize": "prize_3"}
REQUIRED = {"date", "prize_1", "prize_2", "prize_3"}
ALL_NUMBERS = np.array([f"{i:04d}" for i in range(10000)])
ALL_DIGITS = np.array([[int(c) for c in s] for s in ALL_NUMBERS], dtype=np.int16)


def clean_number(x):
    s = str(x).strip()
    if s.endswith('.0'):
        s = s[:-2]
    if not s.isdigit() or len(s) > 4:
        return None
    return s.zfill(4)


def load_draws(source):
    """Read the third-party source schema. Preserve leading zeroes and never fabricate data."""
    df = pd.read_csv(source, dtype=str, encoding='utf-8-sig')
    df.columns = df.columns.str.strip().str.lower()
    if not REQUIRED.issubset(df.columns):
        raise ValueError(f"CSV must contain {sorted(REQUIRED)}. Found: {list(df.columns)}")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    for col in [c for c in df.columns if c.startswith(('prize_', 'special_', 'consol_'))]:
        df[col] = df[col].map(clean_number)
    df = df.dropna(subset=['date', 'prize_1', 'prize_2', 'prize_3'])
    if 'draw_seq' in df.columns:
        df = df.drop_duplicates(subset=['date', 'draw_seq'], keep='last')
    else:
        df = df.drop_duplicates(subset=['date'], keep='last')
    df = df.sort_values(['date'] + (['draw_seq'] if 'draw_seq' in df.columns else []))
    return df.reset_index(drop=True)


def digit_matrix(values):
    return np.array([[int(c) for c in s] for s in values], dtype=np.int16)


def make_features(history: np.ndarray):
    """Feature extraction strictly uses data available BEFORE prediction."""
    x = []
    for lag in range(1, 6):
        x.extend(history[-lag].tolist())
    for w in (20, 60, 120):
        segment = history[-w:]
        for pos in range(4):
            cnt = np.bincount(segment[:, pos], minlength=10)
            x.extend((cnt / len(segment)).tolist())
    for pos in range(4):
        x.append(float(np.mean(history[-20:, pos])))
    return np.asarray(x, dtype=np.float32)


def make_xy(digits, start=120, end=None):
    end = len(digits) if end is None else end
    if end <= start:
        raise ValueError('Not enough historical draws for feature construction.')
    return np.stack([make_features(digits[:i]) for i in range(start, end)]), digits[start:end]


def normalize(p):
    p = np.maximum(np.asarray(p, dtype=float), 1e-15)
    return p / p.sum(axis=1, keepdims=True)


def frequency_probs(history, window=120, alpha=10.0):
    h = history[-window:]
    counts = np.stack([np.bincount(h[:, j], minlength=10) for j in range(4)])
    return normalize(counts + alpha / 10.0)


def markov_probs(history, alpha=15.0):
    if len(history) < 2:
        return np.full((4, 10), 0.1)
    result = np.empty((4, 10))
    for pos in range(4):
        prev = history[:-1, pos]
        nxt = history[1:, pos]
        observed = nxt[prev == history[-1, pos]]
        result[pos] = np.bincount(observed, minlength=10) + alpha / 10.0
    return normalize(result)


def fit_models(X, y, kind='XGBoost'):
    models = []
    if kind == 'XGBoost':
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError('XGBoost not installed. Run pip install -r requirements.txt') from exc
        for pos in range(4):
            model = XGBClassifier(n_estimators=120, max_depth=3, learning_rate=0.04,
                                  subsample=0.8, colsample_bytree=0.8,
                                  objective='multi:softprob', num_class=10,
                                  random_state=42, n_jobs=2)
            model.fit(X, y[:, pos])
            models.append(model)
    return models


def model_probs(models, feature):
    if not models:
        return np.full((4, 10), 0.1)
    rows = []
    for m in models:
        raw = m.predict_proba(feature[None, :])[0]
        aligned = np.full(10, 1e-9)
        aligned[np.asarray(m.classes_, dtype=int)] = raw
        rows.append(aligned)
    return normalize(np.stack(rows))


def combined_probs(history, models=None, use_ml=True):
    f = frequency_probs(history)
    m = markov_probs(history)
    if use_ml and models:
        ml = model_probs(models, make_features(history))
        return normalize(0.30*f + 0.15*m + 0.55*ml)
    return normalize(0.70*f + 0.30*m)


def rank_numbers(probabilities, top_n=10):
    """Under factorized-position approximation, product scores sum to one over all 10000."""
    scores = np.prod(probabilities[np.arange(4)[None, :], ALL_DIGITS], axis=1)
    order = np.argsort(-scores, kind='stable')[:top_n]
    return pd.DataFrame({'Rank': range(1, len(order)+1),
                         'Number': ALL_NUMBERS[order],
                         'Model score (%)': np.round(scores[order]*100, 5)})


def backtest(digits, train_fraction=0.8, top_k=(1, 10, 50, 100), use_ml=True,
             max_test=250, progress=None):
    """Chronological holdout. ML fitted only on training; test features use preceding actual draws."""
    if len(digits) < 300:
        raise ValueError('Need at least 300 draws for the train/test evaluation.')
    split = max(240, int(len(digits)*train_fraction))
    test_start = max(split, len(digits)-max_test)
    if test_start >= len(digits):
        raise ValueError('No holdout draws available.')
    models = fit_models(*make_xy(digits, end=split)) if use_ml else []
    modes = ['Frequency', 'Markov', 'Ensemble' if use_ml else 'Statistical blend', 'Uniform baseline']
    counts = {name: {k:0 for k in top_k} for name in modes}
    ll = {name:[] for name in modes}
    uniform = np.full((4,10),0.1)
    for index, i in enumerate(range(test_start,len(digits))):
        hist = digits[:i]
        f = frequency_probs(hist)
        m = markov_probs(hist)
        blend = combined_probs(hist, models, use_ml=use_ml)
        model_map = {modes[0]:f, modes[1]:m, modes[2]:blend, modes[3]:uniform}
        target = int(''.join(map(str, digits[i])))
        for name, p in model_map.items():
            scores = np.prod(p[np.arange(4)[None,:], ALL_DIGITS], axis=1)
            # deterministic ordering on ties is NOT a random forecast; expectation is analytical.
            if name != 'Uniform baseline':
                ranks = np.argsort(-scores,kind='stable')[:max(top_k)]
                for k in top_k:
                    counts[name][k] += int(target in ranks[:k])
            ll[name].append(-np.log(max(scores[target], 1e-15)))
        if progress and (index+1)%25==0:
            progress((index+1)/(len(digits)-test_start))
    n = len(digits)-test_start
    rows = []
    for name in modes:
        row = {'Method':name}
        for k in top_k:
            row[f'Top {k} hit (%)'] = round((100*k/10000) if name == 'Uniform baseline' else 100*counts[name][k]/n, 3)
        row['Avg. negative log likelihood'] = round(np.mean(ll[name]), 4)
        rows.append(row)
    return pd.DataFrame(rows), models, test_start


def occurrence_chart(digits, window=120):
    counts = np.stack([np.bincount(digits[-window:,i], minlength=10) for i in range(4)])
    return pd.DataFrame(counts.T, index=list(range(10)), columns=['Digit 1','Digit 2','Digit 3','Digit 4'])
