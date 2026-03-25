import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="IA Trader Pro", layout="wide")
st.title("🎯 IA DayTrade Personal")

ticker = st.text_input("Digite o Ativo (Ex: PETR4.SA, BTC-USD)", "PETR4.SA")
intervalo = st.selectbox("Tempo Gráfico", ["1m", "5m", "15m"], index=1)

if st.button("🚀 Analisar Agora"):
    try:
        dados = yf.download(ticker, period="1d", interval=intervalo)
        
        if not dados.empty:
            if dados.columns.nlevels > 1:
                dados.columns = dados.columns.get_level_values(0)

            # Cálculos Técnicos
            dados['MA8'] = dados['Close'].rolling(window=8).mean()
            dados['MA20'] = dados['Close'].rolling(window=20).mean()
            
            # Cálculo do RSI (Força do Mercado)
            delta = dados['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            dados['RSI'] = 100 - (100 / (1 + rs))

            # Captura os últimos valores
            ma8 = dados['MA8'].iloc[-1]
            ma20 = dados['MA20'].iloc[-1]
            rsi = dados['RSI'].iloc[-1]
            preco = dados['Close'].iloc[-1]

            # --- LÓGICA DA IA ---
            # COMPRA: Média curta cruza acima da longa e mercado não está "caro" (RSI < 70)
            if ma8 > ma20 and rsi < 70:
                st.success(f"✅ SINAL: COMPRA (Preço: R$ {preco:.2f} | RSI: {rsi:.2f})")
                st.write("👉 Motivo: Médias em alta e ainda há espaço para subir.")
            
            # VENDA: Média curta cruza abaixo da longa e mercado não está "barato" (RSI > 30)
            elif ma8 < ma20 and rsi > 30:
                st.error(f"🚨 SINAL: VENDA (Preço: R$ {preco:.2f} | RSI: {rsi:.2f})")
                st.write("👉 Motivo: Médias em queda e força vendedora aumentando.")
            
            else:
                st.warning(f"⚪ AGUARDAR (Preço: R$ {preco:.2f} | RSI: {rsi:.2f})")
                st.write("👉 Motivo: Mercado sem direção clara ou muito esticado.")

            # Gráfico
            fig = go.Figure(data=[go.Candlestick(
                x=dados.index, open=dados['Open'], high=dados['High'],
                low=dados['Low'], close=dados['Close'], name='Preço'
            )])
            fig.add_trace(go.Scatter(x=dados.index, y=dados['MA8'], name='Média 8', line=dict(color='cyan')))
            fig.add_trace(go.Scatter(x=dados.index, y=dados['MA20'], name='Média 20', line=dict(color='yellow')))
            
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Erro: {e}")
