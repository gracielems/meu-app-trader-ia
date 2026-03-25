import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="IA Market Intelligence", layout="wide")

# Estilo Visual
st.markdown("<h1 style='text-align: center; color: #00FFCC;'>🏦 IA Intelligence - Scanner de Oportunidades</h1>", unsafe_allow_html=True)

# Lista de ativos incluindo Dólar e Mini Dólar
# Nota: No Yahoo Finance, o Mini Dólar é representado por WDO=F (Contrato Futuro)
ACOES_SCANNER = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'MGLU3.SA']
CAMBIO_SCANNER = ['USDBRL=X', 'WDO=F', 'EURBRL=X']

def analisar_ativo(ticker):
    # Busca dados de 5 dias para ter a Média 200 no gráfico de 5m
    dados = yf.download(ticker, period="5d", interval="5m", progress=False)
    if dados.empty: return None
    
    if dados.columns.nlevels > 1:
        dados.columns = dados.columns.get_level_values(0)
    
    # Médias e RSI
    ma20 = dados['Close'].rolling(window=20).mean()
    ma200 = dados['Close'].rolling(window=200).mean()
    delta = dados['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rsi = 100 - (100 / (1 + (gain / loss)))
    
    c = dados.iloc[-1]
    
    # Score de Inteligência (0 a 3)
    score = 0
    if c['Close'] > ma200.iloc[-1]: score += 1
    if c['Close'] > ma20.iloc[-1]: score += 1
    if 30 < rsi.iloc[-1] < 70: score += 1
    
    return {
        "Ticker": ticker,
        "Preço": round(c['Close'], 3),
        "RSI": round(rsi.iloc[-1], 2),
        "Sinal": "COMPRA" if score >= 2 else "AGUARDAR",
        "Tendencia": "ALTA" if c['Close'] > ma200.iloc[-1] else "BAIXA"
    }

# Layout de Colunas
col_cambio, col_acoes = st.columns(2)

with col_cambio:
    st.subheader("💵 Câmbio e Futuros")
    if st.button("🔄 Atualizar Dólar"):
        for moed in CAMBIO_SCANNER:
            res = analisar_ativo(moed)
            if res:
                nome = "Mini Dólar" if "WDO" in moed else "Dólar Comercial" if "USD" in moed else "Euro"
                cor = "green" if res['Sinal'] == "COMPRA" else "yellow"
                st.info(f"**{nome}** ({res['Ticker']}): **R$ {res['Preço']}** | Tendência: {res['Tendencia']}")

with col_acoes:
    st.subheader("📊 Scanner de Ações (Top B3)")
    if st.button("🚀 Escanear Ações"):
        for acao in ACOES_SCANNER:
            res = analisar_ativo(acao)
            if res:
                cor = "green" if res['Sinal'] == "COMPRA" else "yellow"
                st.markdown(f"**{res['Ticker']}**: R$ {res['Preço']} | RSI: {res['RSI']} | :{cor}[{res['Sinal']}]")

# Seção de Notícias Real-Time
st.divider()
col_news, col_chart = st.columns([1, 2])

with col_news:
    st.subheader("📰 Notícias de Impacto")
    ativo_foco = st.selectbox("Ver notícias de:", CAMBIO_SCANNER + ACOES_SCANNER)
    ticker_obj = yf.Ticker(ativo_foco)
    news = ticker_obj.news[:4]
    for item in news:
        st.write(f"🔗 [{item['title']}]({item['link']})")
        st.caption(f"{item['publisher']} | {datetime.fromtimestamp(item['providerPublishTime']).strftime('%H:%M')}")

with col_chart:
    st.subheader(f"📈 Gráfico Analítico: {ativo_foco}")
    df_grafico = yf.download(ativo_foco, period="2d", interval="5m")
    if not df_grafico.empty:
        if df_grafico.columns.nlevels > 1: df_grafico.columns = df_grafico.columns.get_level_values(0)
        fig = go.Figure(data=[go.Candlestick(x=df_grafico.index, open=df_grafico['Open'], high=df_grafico['High'], low=df_grafico['Low'], close=df_grafico['Close'], name='Candles')])
        fig.update_layout(height=400, template="plotly_dark", margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)
