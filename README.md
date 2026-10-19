# 4D Forecast Lab (MVP)

A local research-oriented Streamlit app for Magnum 4D, Sports Toto, and Da Ma Cai historical result analysis. It is **not** a winning-number predictor, and no genuine predictive advantage has been established.

## Run on Windows

1. Install Python 3.10+ from python.org (check **Add Python to PATH**).
2. Extract this ZIP to a folder.
3. Double-click `run_windows.bat`. First launch installs dependencies and needs internet access.
4. Open the local browser URL shown by Streamlit (normally http://localhost:8501).
5. Pick an operator. The default mode fetches the **third-party historical CSV** from `https://github.com/deadboy18/malaysia-4d`. If unavailable, download the CSV separately and use **Upload CSV file**.
6. Choose 1st / 2nd / 3rd prize, run a forecast and chronological backtest.

## Data schema

`date,prize_1,prize_2,prize_3` are mandatory. Optional `draw_seq,special_1...special_10,consol_1...consol_10` are preserved. Numbers must be four-digit strings (e.g. `0007`); integer-like inputs are padded during cleaning. Historic dataset is publicly described as updated through **April 26, 2026**; it is not live and may contain errors. Always verify with official results.

## Technical details

- Each operator is analyzed **separately**.
- First-version prediction target is **one chosen top-three prize per draw** (not 23 winning results at once).
- Features use only preceding draws: 1–5 previous 4-digit targets, positional digit frequencies over 20/60/120 previous draws, and recent positional means.
- Four independent XGBoost classifiers predict the distributions for positions 1–4.
- Frequency smoothing and first-order position Markov estimates provide statistical comparisons.
- A fixed, illustrative ensemble blends probabilities (30% frequency, 15% Markov, 55% XGBoost); without XGBoost uses 70% frequency / 30% Markov.
- A full 10,000-number ranking is calculated under the **independence approximation across digit positions**. Dependence and calibration are not modeled. Values are not established real-world odds.
- Backtesting uses chronological training/holdout. XGBoost fits on training only. Each holdout forecast uses preceding known outcomes. Uniform baseline Top-K hit rates are **theoretical**, not an observed random run. Other models are tested on observed history. This is not an expanding-window refit of XGBoost.
- Save forecasts and evaluation tables as CSV.

## Limitations / next engineering steps

1. Add official source verification, incremental syncing, and snapshots of data checks.
2. Consider multiple-prize 23-number draw modeling separately, rather than treating it as the same single-prize target.
3. Add nested time-series model selection, uncertainty intervals, probability calibration, class-imbalance auditing, and permutation tests for purported improvement over chance.
4. Handle source-specific draw-date ambiguity more comprehensively if extra draws occur on the same day; `draw_seq` is used for de-duplication when present.
5. Evaluate on untouched future draws before claiming any advantage. Most apparent historical patterns in independent lotteries are noise.

## Attribution

Original third-party historical dataset: https://github.com/deadboy18/malaysia-4d (MIT license; independently review dataset terms). This app is original code and does not bundle a copy of upstream historical data.
