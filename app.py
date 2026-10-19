from __future__ import annotations
import io
import streamlit as st
import pandas as pd
from engine import (OPERATORS, BASE_URL, PRIZES, load_draws, digit_matrix,
                    frequency_probs, markov_probs, make_xy, fit_models,
                    combined_probs, rank_numbers, backtest, occurrence_chart)

st.set_page_config(page_title='4D Forecast Lab', page_icon='🔢', layout='wide')
st.markdown('''<style>
.block-container{padding-top:2rem;max-width:1200px}
h1{letter-spacing:-.035em}
[data-testid="stMetricValue"]{font-weight:750}
</style>''', unsafe_allow_html=True)
st.title('🔢 4D Forecast Lab')
st.caption('Historical analysis · Machine learning · Chronological backtesting | Research prototype, not a winning-number prediction guarantee')
st.warning('Lottery results may be independent and random. Forecast rankings are unverified model scores, not reliable winning probabilities. Never use them as evidence of an advantage.')

@st.cache_data(show_spinner=False, ttl=3600)
def download_operator(name):
    # Executed on the USER computer when this app runs locally; network is needed.
    import requests
    url=BASE_URL + OPERATORS[name]
    response=requests.get(url, timeout=30, headers={'User-Agent':'4D-Forecast-Lab/1.0'})
    response.raise_for_status()
    return load_draws(io.BytesIO(response.content))

with st.sidebar:
    st.header('Data source')
    operator=st.selectbox('Operator',list(OPERATORS))
    source=st.radio('Load data', ['Download public historical CSV','Upload CSV file'])
    uploaded=st.file_uploader('CSV file',type=['csv']) if source=='Upload CSV file' else None
    prize_label=st.selectbox('Target',list(PRIZES),help='First version trains a separate forecast for one prize position, not all 23 winners.')
    use_ml=st.checkbox('Use XGBoost model',value=True)
    st.caption('Source: github.com/deadboy18/malaysia-4d. This is a third-party repository, not an official data feed.')

try:
    if source=='Download public historical CSV':
        with st.spinner('Downloading dataset…'):
            df=download_operator(operator)
    else:
        if uploaded is None:
            st.info('Upload a CSV to begin. Required columns: date, prize_1, prize_2, prize_3.')
            st.stop()
        df=load_draws(uploaded)
except Exception as exc:
    st.error(f'Cannot load the dataset: {exc}')
    st.info('If GitHub is blocked, open the source repository in your browser, download the CSV, and choose Upload CSV file.')
    st.stop()

if len(df)<300:
    st.error('At least 300 historical draws are required for this version.')
    st.stop()
values=df[PRIZES[prize_label]].tolist()
digits=digit_matrix(values)

metrics=st.columns(4)
metrics[0].metric('Draws',f'{len(df):,}')
metrics[1].metric('First draw',df['date'].min().strftime('%Y-%m-%d'))
metrics[2].metric('Latest draw in file',df['date'].max().strftime('%Y-%m-%d'))
metrics[3].metric('Target',prize_label)
if df['date'].max()<pd.Timestamp.today()-pd.Timedelta(days=14):
    st.info('Historical data may be stale. Download and verify newer results before making a next-draw forecast.')

tab1,tab2,tab3,tab4=st.tabs(['🎯 Forecast','📉 Backtesting','📊 Patterns','🗂 Data quality'])

with tab1:
    st.subheader(f'{operator} · Next {prize_label}')
    st.write('Generate a complete ranking of 0000–9999 using four digit-position models. The model only learns this selected prize column.')
    top_n=st.slider('Display top N',5,100,10,5)
    if st.button('Train and generate forecast',type='primary'):
        with st.spinner('Training models / computing number ranks…'):
            try:
                models=fit_models(*make_xy(digits)) if use_ml else []
                probs=combined_probs(digits,models,use_ml=use_ml)
                result=rank_numbers(probs,top_n)
                st.session_state['forecast']=(operator,prize_label,use_ml,len(df),result,probs)
            except Exception as exc:
                st.error(f'Model training failed: {exc}')
    saved=st.session_state.get('forecast')
    if saved and saved[:4]==(operator,prize_label,use_ml,len(df)):
        _,_,_,_,result,probs=saved
        st.dataframe(result.head(top_n),hide_index=True,use_container_width=True)
        st.download_button('Download ranked forecast CSV', result.head(top_n).to_csv(index=False).encode(),file_name='4d_forecast.csv',mime='text/csv')
        st.caption('Score is a factorized-position model estimate and may be uncalibrated. A truly uniform, independent 4-digit first-prize draw has 0.01% probability per number.')
        per_pos=pd.DataFrame(probs.T,columns=['Position 1','Position 2','Position 3','Position 4'],index=list(range(10)))
        st.write('Per-digit model estimates')
        st.bar_chart(per_pos)

with tab2:
    st.subheader('Chronological holdout (no future leakage)')
    st.write('Fit XGBoost only on earlier draws, then forecast the later draws using information already known before each draw. Compare observed Top-K hits against uniform random expectation.')
    n_test=st.slider('Evaluate latest N draws',50,min(500,len(df)-240),min(200,len(df)-240),25)
    if st.button('Run evaluation'):
        with st.spinner('Backtesting…'):
            try:
                bar=st.progress(0)
                stats,models,test_start=backtest(digits,use_ml=use_ml,max_test=n_test,progress=bar.progress)
                bar.progress(1.0)
                st.session_state['backtest']=(operator,prize_label,use_ml,len(df),n_test,stats,test_start)
            except Exception as exc:
                st.error(f'Backtest failed: {exc}')
    saved=st.session_state.get('backtest')
    if saved and saved[:5]==(operator,prize_label,use_ml,len(df),n_test):
        stats=saved[5]
        st.dataframe(stats,hide_index=True,use_container_width=True)
        st.download_button('Download backtest results',stats.to_csv(index=False),file_name='backtest.csv')
        st.caption('Uniform baseline Top-K values are theoretical expectations, not observed random picks. Finite samples fluctuate. Lower negative log likelihood is better. Model weights are experimental, not optimized or calibrated.')

with tab3:
    window=st.slider('Recent draw window',20,min(300,len(df)),min(120,len(df)),10)
    st.bar_chart(occurrence_chart(digits,window))
    last=df[['date',PRIZES[prize_label]]].tail(20).copy()
    last['date']=last['date'].dt.strftime('%Y-%m-%d')
    st.write('Recent selected-prize results')
    st.dataframe(last.iloc[::-1],hide_index=True,use_container_width=True)

with tab4:
    st.write('Source rows:',len(df))
    st.write('Column names:',', '.join(df.columns))
    st.write('Duplicate dates:',int(df['date'].duplicated().sum()),'(special draw identifiers can share a date)')
    st.write('Missing selected prize:',int(df[PRIZES[prize_label]].isna().sum()))
    st.dataframe(df.tail(12).iloc[::-1],hide_index=True,use_container_width=True)
    st.download_button('Download cleaned data',df.to_csv(index=False).encode('utf-8-sig'),file_name=f'{OPERATORS[operator]}',mime='text/csv')

st.divider()
st.caption('For research and educational use. No betting integration, financial advice, or automated purchase actions. Third-party data must be verified against official draw records.')
