import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# Configuração da página para parecer um terminal profissional
st.set_page_config(page_title="IA PRO TRADER", layout="wide", initial_sidebar_state="collapsed")

# Estilização CSS para modo Dark e Fluidez
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    .status-card { padding: 20px; border-radius: 10px; margin-bottom: 10px; border-left: 5px solid #00ffcc; background: #161b22; }
    </style>
    """, unsafe_allow_html=True)

st.title("⚡ IA PRO TRADER: Inteligência de Lucro")

# --- INTELIGÊNCIA DE MERCADO (FUNÇÕES) ---

@st.cache_data(ttl=60) # Atualiza a cada 1 minuto para não ser bloqueado
def buscar_dados(ticker):
    try:
        df = yf.download(ticker, period="2d", interval="5m", progress=False)
        if df.empty or len(df) < 50: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df
    except: return None

def analisar_estrategia(df):
    # Médias Móveis Profissionais
    df['MA9'] = df['Close'].rolling(window=9).mean()
    df['MA21'] = df['Close'].rolling(window=21).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # RSI (Força Relativa)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    ultimo = df.iloc[-1]
    penultimo = df.iloc[-2]
    
    # PONTUAÇÃO DE COMPRA (0 a 100)
    score = 0
    if ultimo['Close'] > ultimo['MA200']: score += 30 # Tendência de Longo Prazo
    if ultimo['MA9'] > ultimo['MA21']: score += 30    # Cruzamento de Médias
    if 30 < ultimo['RSI'] < 60: score += 20          # Espaço para subir
    if ultimo['Volume'] > df['Volume'].mean(): score += 20 # Volume Confirmando
    
    return {
        "score": score,
        "preco": ultimo['Close'],
        "rsi": ultimo['RSI'],
        "alvo": ultimo['Close'] * 1.015, # Alvo de 1.5% para Day Trade
        "stop": ultimo['Close'] * 0.992, # Stop de 0.8% (Gestão de Risco)
        "df": df
    }

# --- PAINEL PRINCIPAL ---

col_m1, col_m2, col_m3 = st.columns(3)

# 1. SENTIMENTO GLOBAL
with col_m1:
    sp500 = yf.Ticker("^GSPC").history(period="1d")
    var = ((sp500['Close'].iloc[-1] / sp500['Open'].iloc[-1]) - 1) * 100
    st.metric("S&P 500 (Mundo)", f"{var:.2f}%", delta="GLOBAL" if var > 0 else "CAUTELA")

# 2. CÂMBIO
with col_m2:
    dolar = yf.Ticker("USDBRL=X").history(period="1d")
    st.metric("Dólar / Real", f"R$ {dolar['Close'].iloc[-1]:.2f}")

# 3. FILTRO DE OPORTUNIDADES
with col_m3:
    st.write("⏱️ Próxima Varredura em 60s")

st.divider()

# LISTA DE VARREDURA (As mais líquidas da B3)
LISTA_B3 = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'MGLU3.SA', 'ABEV3.SA', 'RENT3.SA', 'B3SA3.SA']

if st.button("🚀 EXECUTAR ANÁLISE DE ELITE"):
    oportunidades = []
    
    with st.spinner('Varrendo mercado e analisando notícias...'):
        for ativo in LISTA_B3:
            dados = buscar_dados(ativo)
            if dados is not None:
                analise = analisar_estrategia(dados)
                oportunidades.append({
                    "ticker": ativo,
                    "score": analise['score'],
                    "preco": analise['preco'],
                    "alvo": analise['alvo'],
                    "stop": analise['stop'],
                    "df": analise['df']
                })
    
    # ORDENAR PELAS 3 MELHORES (MAIOR SCORE)
    top_3 = sorted(oportunidades, key=lambda x: x['score'], reverse=True)[:3]
    
    st.subheader("🎯 As 3 Melhores Oportunidades Agora")
    
    cols = st.columns(3)
    for i, op in enumerate(top_3):
        with cols[i]:
            status = "🔥 COMPRA FORTE" if op['score'] >= 70 else "⚖️ AGUARDAR"
            st.markdown(f"""
                <div class="status-card">
                    <h3>{op['ticker']}</h3>
                    <h2 style="color: #00ffcc;">{status}</h2>
                    <p><b>Preço de Entrada:</b> R$ {op['preco']:.2f}</p>
                    <p style="color: #00ff00;">🎯 <b>Vender em (ALVO):</b> R$ {op['alvo']:.2f}</p>
                    <p style="color: #ff4b4b;">🛑 <b>Vender em (STOP):</b> R$ {op['stop']:.2f}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Mini Gráfico do Sinal
            fig = go.Figure(data=[go.Candlestick(x=op['df'].index[-20:],
                open=op['df']['Open'][-20:], high=op['df']['High'][-20:],
                low=op['df']['Low'][-20:], close=op['df']['Close'][-20:])])
            fig.update_layout(xaxis_rangeslider_visible=False, height=200, margin=dict(l=0,r=0,t=0,b=0), template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

st.divider()

# RADAR DE NOTÍCIAS SIMPLIFICADO
st.subheader("📰 Radar de Notícias Impactantes")
try:
    ticker_news = yf.Ticker("PETR4.SA")
    for n in ticker_news.news[:3]:
        col_n1, col_n2 = st.columns([1, 4])
        with col_n2:
            st.markdown(f"**{n.get('title')}**")
            st.caption(f"Fonte: {n.get('publisher')} | [Ler Notícia]({n.get('link')})")
except:
    st.info("Aguardando novas atualizações de notícias...")
