import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go

st.set_page_config(page_title="IA Trader Pro", layout="wide")

st.title("🎯 IA DayTrade - Sinais de Operação")

# Configurações na lateral
with st.sidebar:
    st.header("Configurações")
    ticker = st.text_input("Ativo (Ex: PETR4.SA, BTC-USD)", "PETR4.SA")
    intervalo = st.selectbox("Tempo Gráfico", ["1m", "5m", "15m"], index=1)

if st.button("🚀 Analisar Agora"):
    try:
        # Busca dados do dia
        dados = yf.download(ticker, period="1d", interval=intervalo)
        
        if not dados.empty:
            # Organiza as colunas
            if isinstance(dados.columns, pd.MultiIndex):
                dados.columns = dados.columns.get_level_values(0)

            # Cálculos de Indicadores
            dados['EMA_8'] = ta.ema(dados['Close'], length=8)
            dados['EMA_20'] = ta.ema(dados['Close'], length=20)
            dados['RSI'] = ta.rsi(dados['Close'], length=14)

            ultimo = dados.iloc[-1]
            penultimo = dados.iloc[-2]

            # Lógica de Sinais
            cruzamento_alta = (penultimo['EMA_8'] <= penultimo['EMA_20']) and (ultimo['EMA_8'] > ultimo['EMA_20'])
            cruzamento_baixa = (penultimo['EMA_8'] >= penultimo['EMA_20']) and (ultimo['EMA_8'] < ultimo['EMA_20'])

            # Exibição do sinal
            if cruzamento_alta and ultimo['RSI'] < 70:
                st.success("✅ SINAL DE COMPRA: Tendência de alta confirmada!")
            elif cruzamento_baixa and ultimo['RSI'] > 30:
                st.error("🚨 SINAL DE VENDA: Tendência de baixa confirmada!")
            else:
                st.warning("⚪ AGUARDAR: Mercado sem tendência clara")

            # Gráfico de Candles
            fig = go.Figure(data=[go.Candlestick(
                x=dados.index, open=dados['Open'], high=dados['High'],
                low=dados['Low'], close=dados['Close'], name='Preço'
            )])
            fig.add_trace(go.Scatter(x=dados.index, y=dados['EMA_8'], name='Média 8 (Rápida)', line=dict(color='cyan')))
            fig.add_trace(go.Scatter(x=dados.index, y=dados['EMA_20'], name='Média 20 (Lenta)', line=dict(color='yellow')))
            
            st.plotly_chart(fig, use_container_width=True)
            st.write(f"**RSI Atual:** {ultimo['RSI']:.2f}")

    except Exception as e:
        st.error(f"Erro ao processar: {e}")



