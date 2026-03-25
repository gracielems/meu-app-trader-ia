import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="IA Trader Sniper", layout="wide")
st.title("🎯 IA DayTrade Sniper - Alta Precisão")

ticker = st.text_input("Ativo", "PETR4.SA")
intervalo = st.selectbox("Tempo Gráfico", ["1m", "5m", "15m"], index=1)

if st.button("🚀 Gerar Sinal Sniper"):
    try:
        # Busca mais dados para calcular a média de 200
        dados = yf.download(ticker, period="5d", interval=intervalo)
        
        if not dados.empty:
            if dados.columns.nlevels > 1:
                dados.columns = dados.columns.get_level_values(0)

            # Médias Móveis
            dados['MA8'] = dados['Close'].rolling(window=8).mean()
            dados['MA20'] = dados['Close'].rolling(window=20).mean()
            dados['MA200'] = dados['Close'].rolling(window=200).mean()
            
            # RSI
            delta = dados['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            dados['RSI'] = 100 - (100 / (1 + (gain / loss)))
            
            # Volume
            dados['Vol_Media'] = dados['Volume'].rolling(window=10).mean()

            # Valores Atuais
            c = dados.iloc[-1]
            p = dados.iloc[-2] # Penúltimo para ver o cruzamento

            # --- LÓGICA SNIPER (MAIS ASSERTIVA) ---
            compra = (c['MA8'] > c['MA20']) and (c['RSI'] > 50) and (c['Close'] > c['MA200']) and (c['Volume'] > c['Vol_Media'])
            venda = (c['MA8'] < c['MA20']) and (c['RSI'] < 50) and (c['Close'] < c['MA200']) and (c['Volume'] > c['Vol_Media'])

            if compra:
                st.success(f"🔥 SINAL FORTE DE COMPRA! (R$ {c['Close']:.2f})")
                st.info("✅ Motivo: Tendência de alta confirmada pela MA200 + Volume acima da média.")
            elif venda:
                st.error(f"💀 SINAL FORTE DE VENDA! (R$ {c['Close']:.2f})")
                st.info("✅ Motivo: Tendência de baixa confirmada pela MA200 + Pressão vendedora.")
            else:
                st.warning("⚖️ AGUARDAR: Filtros de segurança não confirmados.")
                st.write("Dica: O mercado pode estar sem volume ou contra a tendência principal.")

            # Gráfico com a MA200 (Linha Branca)
            fig = go.Figure(data=[go.Candlestick(x=dados.index, open=dados['Open'], high=dados['High'], low=dados['Low'], close=dados['Close'], name='Preço')])
            fig.add_trace(go.Scatter(x=dados.index, y=dados['MA8'], name='Média 8', line=dict(color='cyan')))
            fig.add_trace(go.Scatter(x=dados.index, y=dados['MA20'], name='Média 20', line=dict(color='yellow')))
            fig.add_trace(go.Scatter(x=dados.index, y=dados['MA200'], name='Média 200 (Tendência)', line=dict(color='white', width=3)))
            
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Erro: {e}")
