import os
import subprocess
import sys

# Comando para instalar a biblioteca de sinais caso ela não esteja lá
try:
    import pandas_ta as ta
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas-ta"])
    import pandas_ta as ta

import streamlit as st
import yfinance as yf
# ... o restante do código continua igual
import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta # Biblioteca para cálculos técnicos
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="IA DayTrade PRO", layout="wide")

st.title("🎯 IA DayTrade Personal - Sinais de Entrada")

# ENTRADA DE DADOS
with st.sidebar:
    st.header("Configurações")
    ticker = st.text_input("Digite o Ativo (Ex: PETR4.SA, BTC-USD)", "BTC-USD")
    intervalo = st.selectbox("Intervalo de Tempo (Mins)", ["1m", "5m", "15m", "30m"], index=1)
    periodo_media_rapida = st.number_input("Média Móvel Rápida", 8, 50, 8)
    periodo_media_lenta = st.number_input("Média Móvel Lenta", 20, 100, 20)

if st.button("🚀 Analisar e Gerar Sinal"):
    try:
        # Busca dados de hoje
        dados = yf.download(ticker, period="1d", interval=intervalo)
        
        if dados.empty:
            st.error("Dados não encontrados para este ativo.")
        else:
            # --- CÁLCULO DOS INDICADORES TÉCNICOS ---
            
            # Remove qualquer coluna multi-index se houver
            if isinstance(dados.columns, pd.MultiIndex):
                dados.columns = dados.columns.get_level_values(0)

            # Calcula Médias Móveis (Usando pandas_ta para precisão)
            dados['MA_Fast'] = ta.ema(dados['Close'], length=periodo_media_rapida)
            dados['MA_Slow'] = ta.ema(dados['Close'], length=periodo_media_lenta)
            
            # Calcula RSI (14 períodos padrão)
            dados['RSI'] = ta.rsi(dados['Close'], length=14)

            # --- ESTRATÉGIA DA IA (SINAIS) ---
            
            # Pega o último registro completo (o candle que acabou de fechar)
            ultimo_candle = dados.iloc[-1]
            penultimo_candle = dados.iloc[-2]
            
            # Regras de Cruzamento de Média e RSI
            cruzamento_alta = (penultimo_candle['MA_Fast'] <= penultimo_candle['MA_Slow']) and (ultimo_candle['MA_Fast'] > ultimo_candle['MA_Slow'])
            cruzamento_baixa = (penultimo_candle['MA_Fast'] >= penultimo_candle['MA_Slow']) and (ultimo_candle['MA_Fast'] < ultimo_candle['MA_Slow'])
            
            rsi_sobrevendido = ultimo_candle['RSI'] < 35 # Indica "Barato"
            rsi_sobrecomprado = ultimo_candle['RSI'] > 65 # Indica "Caro"
            
            # Define o Sinal Final
            if cruzamento_alta and rsi_sobrevendido:
                sinal_ia = "✅ COMPRA CONFIRMADA (Entrada Curta)"
                cor_sinal = "green"
            elif cruzamento_baixa and rsi_sobrecomprado:
                sinal_ia = "🚨 VENDA CONFIRMADA (Entrada Curta)"
                cor_sinal = "red"
            else:
                sinal_ia = "⚪ AGUARDAR - Tendência Não Confirmada"
                cor_sinal = "gray"

            # --- EXIBIÇÃO ---
            
            # 1. Sinal da IA (Destaque)
            st.markdown(f"<h1 style='text-align: center; color: {cor_sinal};'>{sinal_ia}</h1>", unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                # Gráfico com Médias
                fig = go.Figure(data=[go.Candlestick(
                    x=dados.index,
                    open=dados['Open'], high=dados['High'],
                    low=dados['Low'], close=dados['Close'], name='Preço'
                )])
                # Adiciona Médias ao Gráfico
                fig.add_trace(go.Scatter(x=dados.index, y=dados['MA_Fast'], name='Média Rápida', line=dict(color='yellow', width=1)))
                fig.add_trace(go.Scatter(x=dados.index, y=dados['MA_Slow'], name='Média Lenta', line=dict(color='orange', width=2)))
                
                fig.update_layout(xaxis_rangeslider_visible=False, title=f"DayTrade 5min: {ticker}")
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                # Painel de Resumo
                st.metric(label="Último Preço", value=f"{ultimo_candle['Close']:.2f}")
                st.metric(label="RSI (Força)", value=f"{ultimo_candle['RSI']:.2f}")
                
    except Exception as e:
        st.error(f"Erro ao analisar: {e}")


