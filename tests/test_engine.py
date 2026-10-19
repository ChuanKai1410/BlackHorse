import numpy as np
import pandas as pd
from engine import clean_number, load_draws, digit_matrix, make_xy, frequency_probs, markov_probs, rank_numbers, backtest

def test_padding():
    assert clean_number('7')=='0007'
    assert clean_number('9999')=='9999'
    assert clean_number('10000') is None

def test_scores():
    p=np.full((4,10),.1)
    rank=rank_numbers(p,10)
    assert rank['Number'].iloc[0]=='0000'
    assert abs(rank['Model score (%)'].iloc[0]-.01)<1e-10

def test_features():
    rng=np.random.default_rng(7)
    digits=rng.integers(0,10,(310,4))
    x,y=make_xy(digits)
    assert x.shape==(190,144) and y.shape==(190,4)
    assert np.allclose(frequency_probs(digits).sum(axis=1),1)
    assert np.allclose(markov_probs(digits).sum(axis=1),1)

def test_csv_cleaning():
    from io import StringIO
    data='date,prize_1,prize_2,prize_3\n2026-01-02,7,8,9\n2026-01-01,1234,0020,9999\n'
    df=load_draws(StringIO(data))
    assert df.iloc[1]['prize_1']=='0007' and df.iloc[0]['prize_2']=='0020'

def test_backtest_statistical():
    rng=np.random.default_rng(4)
    d=rng.integers(0,10,(340,4),dtype=np.int16)
    stats,models,idx=backtest(d,use_ml=False,max_test=40)
    assert len(stats)==4 and idx==300
    assert float(stats.loc[stats.Method=='Uniform baseline','Top 10 hit (%)'].iloc[0])==.1
