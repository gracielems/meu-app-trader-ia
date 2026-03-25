import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# Configuração para parecer um App Nativo no celular
st.set_page_config(page_title="IA Trader", layout="wide", initial_sidebar_state="collapsed")

# Estilo CSS para esconder menus e focar no conteúdo
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div.stButton > button { width: 100%; border-radius: 10px; height: 3em; background-color: #2e7d32; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("🎯 IA DayTrade Personal")

# --- ENTRADA DE DADOS ---
col_input1, col_input2 = st.columns([2, 1])
with col_input1:
    ticker = st.text_input("Ativo (Ex: PETR4.SA, AAPL, BTC-USD)", value="PETR4.SA").upper()
with col_input2:
    moeda = st.radio("Moeda", ["R$", "US$"])

# --- FUNÇÃO DE ANÁLISE TÉCNICA ---
def calcular_indicadores(df):
    # RSI (Relative Strength Index) - Período 14
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Médias Móveis
    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    return df

try:
    # Busca dados em tempo real (intervalo de 5 min para Day Trade)
    df = yf.download(ticker, period="1d", interval="5m")
    df = calcular_indicadores(df)
    
    ultimo_preco = df['Close'].iloc[-1]
    rsi_atual = df['RSI'].iloc[-1]
    ema9 = df['EMA_9'].iloc[-1]
    
    # --- PAINEL DE PALPITE ---
    st.subheader("💡 Veredito da IA")
    
    if rsi_atual < 30:
        st.success(f"**COMPRA FORTE:** Ativo sobrevendido (RSI: {rsi_atual:.2f}). Chance alta de repique!")
    elif rsi_atual > 70:
        st.error(f"**VENDA FORTE:** Ativo sobrecomprado (RSI: {rsi_atual:.2f}). Risco de queda iminente!")
    elif ultimo_preco > ema9:
        st.info("**TENDÊNCIA DE ALTA:** Preço acima da média rápida. Procure compra.")
    else:
        st.warning("**TENDÊNCIA DE BAIXA:** Preço abaixo da média rápida. Cuidado com compras.")

    # --- GRÁFICO ---
    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Preço")])
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_9'], line=dict(color='yellow', width=1), name="Média 9"))
    fig.update_layout(xaxis_rangeslider_visible=False, template="plotly_dark", height=350, margin=dict(l=0,r=0,b=0,t=0))
    st.plotly_chart(fig, use_container_width=True)

    # --- NOTÍCIAS (SIMULADO) ---
    st.write("---")
    st.subheader("🌍 Sentimento Global")
    # Aqui você pode integrar uma API de notícias real depois
    st.caption("Fluxo de ordens estrangeiras entrando no Brasil. Dólar pressionado.")

except:
    st.error("Erro ao buscar dados. Verifique o código do ativo.")

# Botão de atualização rápida
if st.button('🔄 Atualizar Agora'):
    st.rerun()
