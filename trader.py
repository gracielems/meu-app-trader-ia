import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

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

            # Médias Móveis calculadas sem bibliotecas extras
            dados['MA8'] = dados['Close'].rolling(window=8).mean()
            dados['MA20'] = dados['Close'].rolling(window=20).mean()

            ultimo_fechamento = dados['Close'].iloc[-1]
            ma8_atual = dados['MA8'].iloc[-1]
            ma20_atual = dados['MA20'].iloc[-1]

            if ma8_atual > ma20_atual:
                st.success(f"✅ TENDÊNCIA DE ALTA!")
            else:
                st.error(f"🚨 TENDÊNCIA DE BAIXA!")

            fig = go.Figure(data=[go.Candlestick(
                x=dados.index, open=dados['Open'], high=dados['High'],
                low=dados['Low'], close=dados['Close'], name='Candles'
            )])
            fig.add_trace(go.Scatter(x=dados.index, y=dados['MA8'], name='Média 8', line=dict(color='cyan')))
            fig.add_trace(go.Scatter(x=dados.index, y=dados['MA20'], name='Média 20', line=dict(color='yellow')))
            
            st.plotly_chart(fig, use_container_width=True)
            st.write(f"**Preço Atual:** R$ {ultimo_fechamento:.2f}")

    except Exception as e:
        st.error(f"Erro: {e}")
