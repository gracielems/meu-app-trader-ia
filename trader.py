import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="IA Trader Institucional", layout="wide")
st.title("🏦 IA Trader - Nível Institucional")

with st.sidebar:
    st.header("Gestão de Capital")
    capital = st.number_input("Seu Capital (R$)", value=1000.0)
    risco_por_op = st.slider("Risco por Operação (%)", 0.5, 3.0, 1.0)
    ticker = st.text_input("Ativo", "PETR4.SA")
    intervalo = st.selectbox("Tempo Gráfico", ["1m", "5m", "15m"], index=1)

if st.button("🔍 Análise Completa de Mercado"):
    try:
        dados = yf.download(ticker, period="5d", interval=intervalo)
        
        if not dados.empty:
            if dados.columns.nlevels > 1:
                dados.columns = dados.columns.get_level_values(0)

            # Indicadores de Elite
            dados['MA8'] = dados['Close'].rolling(window=8).mean()
            dados['MA20'] = dados['Close'].rolling(window=20).mean()
            dados['MA200'] = dados['Close'].rolling(window=200).mean()
            
            # Suporte e Resistência (Máximas e Mínimas dos últimos 50 candles)
            resistencia = dados['High'].rolling(window=50).max().iloc[-1]
            suporte = dados['Low'].rolling(window=50).min().iloc[-1]

            c = dados.iloc[-1]
            p = dados.iloc[-2]

            # Lógica Institucional
            trend_alta = c['Close'] > c['MA200']
            cruzamento_alta = c['MA8'] > c['MA20']
            distancia_resistencia = resistencia - c['Close']

            st.subheader("📋 Veredito da IA")
            col1, col2, col3 = st.columns(3)

            if trend_alta and cruzamento_alta and distancia_resistencia > 0.10:
                col1.success("SINAL: COMPRA FORTE")
                # Cálculo de Gestão de Risco
                stop = c['Low'] * 0.995 # 0.5% abaixo da mínima
                alvo = c['Close'] + (c['Close'] - stop) * 2 # Relação 2 para 1
                col2.metric("Stop Loss", f"R$ {stop:.2f}")
                col3.metric("Alvo (Take Profit)", f"R$ {alvo:.2f}")
                
                # Cálculo de Lote
                perda_financeira = capital * (risco_por_op / 100)
                lote = int(perda_financeira / (c['Close'] - stop))
                st.info(f"💡 Sugestão de Manejo: Compre **{lote}** ações para arriscar apenas R$ {perda_financeira:.2f}")
            
            elif not trend_alta and not cruzamento_alta:
                col1.error("SINAL: VENDA/SHORT")
                stop = c['High'] * 1.005
                alvo = c['Close'] - (stop - c['Close']) * 2
                col2.metric("Stop Loss", f"R$ {stop:.2f}")
                col3.metric("Alvo", f"R$ {alvo:.2f}")
            else:
                col1.warning("SINAL: AGUARDAR")
                st.write("Mercado perigoso: Preço muito próximo de resistência ou sem tendência definida.")

            # Gráfico com Suporte e Resistência
            fig = go.Figure(data=[go.Candlestick(x=dados.index, open=dados['Open'], high=dados['High'], low=dados['Low'], close=dados['Close'], name='Preço')])
            fig.add_trace(go.Scatter(x=dados.index, y=dados['MA200'], name='Tendência (200)', line=dict(color='white', width=2)))
            
            # Linhas de Suporte e Resistência
            fig.add_hline(y=resistencia, line_dash="dash", line_color="red", annotation_text="Resistência")
            fig.add_hline(y=suporte, line_dash="dash", line_color="green", annotation_text="Suporte")
            
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Erro: {e}")
