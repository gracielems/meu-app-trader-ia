import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="IA Trader PRO", layout="wide")

st.title("🎯 IA DayTrade Personal")

# ENTRADA DE DADOS
ticker = st.text_input("Digite o Ativo (Ex: PETR4.SA, ITUB4.SA, BTC-USD)", "PETR4.SA")

if st.button("Analisar Agora"):
    try:
        # Busca dados de hoje em intervalos de 5 minutos
        dados = yf.download(ticker, period="1d", interval="5m")
        
        if dados.empty:
            st.error("Dados não encontrados. Verifique o código do ativo.")
        else:
            # Cria o Gráfico
            fig = go.Figure(data=[go.Candlestick(
                x=dados.index,
                open=dados['Open'],
                high=dados['High'],
                low=dados['Low'],
                close=dados['Close']
            )])
            
            fig.update_layout(title=f"Gráfico de 5 min: {ticker}", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Mostra o Preço Atual
            preco_atual = dados['Close'].iloc[-1]
            st.success(f"Preço Atual: R$ {preco_atual:.2f}")
            
    except Exception as e:
        st.error(f"Erro ao buscar dados: {e}")
