import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="IA Elite Trader", layout="wide")

# Interface Estilizada
st.markdown("<h1 style='text-align: center; color: #00FFCC;'>💎 IA Elite Trader - Intelligence</h1>", unsafe_allow_html=True)

# 1. LISTA DE MONITORAMENTO PROFISSIONAL
ATIVOS = {
    "Ações B3": ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'MGLU3.SA'],
    "Câmbio/Futuros": ['WDO=F', 'USDBRL=X', 'BTC-USD'],
    "Global (EUA)": ['^GSPC', '^IXIC'] # S&P 500 e Nasdaq
}

def calcular_indicadores(df):
    # Médias Móveis
    df['MA8'] = df['Close'].rolling(window=8).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    return df

# --- SIDEBAR: GESTÃO E NOTÍCIAS GLOBAIS ---
with st.sidebar:
    st.header("🌍 Sentimento Global")
    try:
        sp500 = yf.Ticker("^GSPC").history(period="1d")
        variacao = ((sp500['Close'].iloc[-1] / sp500['Open'].iloc[-1]) - 1) * 100
        cor_sp = "green" if variacao > 0 else "red"
        st.metric("S&P 500 (EUA)", f"{variacao:.2f}%", delta_color="normal")
        st.caption("Se o S&P 500 está negativo, cuidado com compras na B3.")
    except:
        st.write("Aguardando abertura de NY...")

    st.divider()
    st.header("⚙️ Configurações")
    capital = st.number_input("Capital da Operação (R$)", value=1000.0)

# --- CORPO PRINCIPAL ---
tab1, tab2 = st.tabs(["🚀 Scanner de Sinais", "📰 Radar de Notícias"])

with tab1:
    st.subheader("Melhores Oportunidades Agora")
    if st.button("🔍 Escanear Todo o Mercado"):
        col1, col2 = st.columns(2)
        
        for cat, lista in ATIVOS.items():
            for ticker in lista:
                df = yf.download(ticker, period="5d", interval="5m", progress=False)
                if df.empty: continue
                if df.columns.nlevels > 1: df.columns = df.columns.get_level_values(0)
                
                df = calcular_indicadores(df)
                c = df.iloc[-1]
                
                # Lógica de Elite: Tendência + Momentum + Volume
                if c['Close'] > c['MA200'] and c['MA8'] > c['MA20'] and 30 < c['RSI'] < 65:
                    with st.expander(f"✅ OPORTUNIDADE EM {ticker}", expanded=True):
                        # Cálculo de Compra e Venda
                        stop_loss = c['Close'] * 0.992 # 0.8% de stop
                        take_profit = c['Close'] * 1.02 # 2% de alvo
                        
                        st.markdown(f"### **Sinal: COMPRAR AGORA**")
                        st.write(f"📍 **Entrada:** R$ {c['Close']:.2f}")
                        st.write(f"🛑 **Venda (Stop Loss):** R$ {stop_loss:.2f}")
                        st.write(f"🎯 **Venda (Alvo/Lucro):** R$ {take_profit:.2f}")
                        st.info(f"Análise: Ativo em forte tendência de alta acima da média de 200.")

with tab2:
    st.subheader("Fatos Relevantes e Notícias Globais")
    ativo_news = st.selectbox("Notícias de:", ATIVOS['Ações B3'] + ATIVOS['Câmbio/Futuros'])
    ticker_obj = yf.Ticker(ativo_news)
    for n in ticker_obj.news[:5]:
        st.markdown(f"⭐ **[{n['title']}]({n['link']})**")
        st.caption(f"Fonte: {n['publisher']}")
