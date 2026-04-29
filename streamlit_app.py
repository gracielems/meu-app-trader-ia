import streamlit as st
import pandas as pd
import requests
import time

# 1. CONFIGURAÇÕES (Suas chaves seguras)
TOKEN_TELEGRAM = "8512230023:AAFm28JQEYr-2PrvwD0kKG6tVaviGjF9aoQ"
ID_TELEGRAM = "7453152256"
TOKEN_BRAPI = "u4ufLNdYFG1Qo3xwHmuoem"

st.set_page_config(page_title="Robô Elite B3", layout="wide")

def buscar_dados(ticker):
    # Garante o formato .SA
    t = ticker.strip().upper()
    if not t.endswith(".SA"): t = f"{t}.SA"
    
    url = f"https://brapi.dev/api/quote/{t}?range=5d&interval=5m&token={TOKEN_BRAPI}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data['results'][0]['historicalData'])
            return df
        return pd.DataFrame()
    except:
        return pd.DataFrame()

# INTERFACE SIMPLES
st.title("🚀 Analista de Elite B3")
st.write("Status: Aguardando abertura do mercado (10:00)")

acao = st.text_input("Digite a ação (ex: PETR4, VALE3)", "BBAS3")

if st.button("Ligar Monitor"):
    st.info(f"Monitorando {acao}...")
    df = buscar_dados(acao)
    
    if not df.empty:
        st.success("Dados recebidos com sucesso!")
        st.write(df.tail())
    else:
        st.warning("Mercado Fechado ou Erro na API. Tente após as 10:00.")
