import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="IA Elite Trader", layout="wide")

st.markdown("<h1 style='text-align: center; color: #00FFCC;'>💎 IA Elite Trader - Intelligence</h1>", unsafe_allow_html=True)

# Lista simplificada para evitar bloqueios (Rate Limit)
ATIVOS = {
    "Ações": ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA'],
    "Câmbio": ['WDO=F', 'USDBRL=X']
}

def analisar_ativo(ticker):
    try:
        df = yf.download(ticker, period="5d", interval="5m", progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        # Indicadores Básicos
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()
        
        c = df.iloc[-1]
        tendencia = "ALTA" if c['Close'] > c['MA200'] else "BAIXA"
        sinal = "COMPRA" if c['Close'] > c['MA20'] and tendencia == "ALTA" else "AGUARDAR"
        
        return {"Ticker": ticker, "Preço": c['Close'], "Sinal": sinal, "Tendencia": tendencia}
    except:
        return None

tab1, tab2 = st.tabs(["🚀 Scanner", "📰 Notícias Globais"])

with tab1:
    if st.button("🔍 Escanear Mercado"):
        for cat, lista in ATIVOS.items():
            st.write(f"**{cat}**")
            for t in lista:
                res = analisar_ativo(t)
                if res:
                    cor = "green" if res['Sinal'] == "COMPRA" else "yellow"
                    st.success(f"{res['Ticker']}: R$ {res['Preço']:.2f} | Sinal: {res['Sinal']}")

with tab2:
    st.subheader("Radar de Notícias Profissional")
    try:
        foco = st.selectbox("Ver notícias de:", ['PETR4.SA', 'VALE3.SA', 'USDBRL=X'])
        ticker_obj = yf.Ticker(foco)
        news = ticker_obj.news
        if news:
            for n in news[:5]:
                # Tratamento de erro para o KeyError que apareceu no seu print
                title = n.get('title', 'Notícia sem título')
                link = n.get('link', '#')
                st.markdown(f"🔗 [{title}]({link})")
        else:
            st.warning("Nenhuma notícia encontrada para este ativo no momento.")
    except Exception as e:
        st.error("Ocorreu um erro ao buscar notícias. Tente novamente em instantes.")

# Gráfico de Apoio
st.divider()
ativo_grafico = st.selectbox("Gráfico Detalhado:", ['PETR4.SA', 'VALE3.SA', 'WDO=F'])
df_plot = yf.download(ativo_grafico, period="1d", interval="5m", progress=False)
if not df_plot.empty:
    if isinstance(df_plot.columns, pd.MultiIndex): df_plot.columns = df_plot.columns.get_level_values(0)
    fig = go.Figure(data=[go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'])])
    fig.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig, use_container_width=True)
