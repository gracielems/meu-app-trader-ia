 
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="IA Trader PRO", layout="wide")

st.title("🎯 IA DayTrade Personal")

ticker = st.text_input("Digite o Ativo (Ex: PETR4.SA, BTC-USD)", "BTC-USD")

if st.button("Analisar Agora"):
    try:
        # Busca dados de hoje
        dados = yf.download(ticker, period="1d", interval="5m")
        
        if dados.empty:
            st.error("Dados não encontrados para este ativo.")
        else:
            # Gráfico de Candlestick
            fig = go.Figure(data=[go.Candlestick(
                x=dados.index,
                open=dados['Open'].values.flatten(),
                high=dados['High'].values.flatten(),
                low=dados['Low'].values.flatten(),
                close=dados['Close'].values.flatten()
            )])
            
            fig.update_layout(xaxis_rangeslider_visible=False, title=f"Gráfico 5min: {ticker}")
            st.plotly_chart(fig, use_container_width=True)
            
            # Pega o último preço de forma segura para evitar o erro de formatação
            ultimo_preco = float(dados['Close'].iloc[-1])
            st.success(f"Preço Atual: {ultimo_preco:.2f}")
            
    except Exception as e:
        st.error(f"Erro ao processar: {e}")
