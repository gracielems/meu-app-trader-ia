import time
import requests
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st

# =========================================================
# CONFIGURAÇÕES BÁSICAS
# =========================================================
st.set_page_config(page_title="Robô Elite B3", page_icon="🚀", layout="wide")

TOKEN_TELEGRAM = "8512230023:AAGyZ0QwPmnWqnZeQKfS7KCkwcVf1fBckCY"
ID_TELEGRAM = "7453152256"
TOKEN_BRAPI = "u4ufLNdYFG1Qo3xwHmuoem"

def enviar_telegram(dados):
    mensagem = f"🚀 **SINAL: {dados['ticker']}**\nScore: {dados['score']}\nPreço: R$ {dados['entry']:.2f}"
    url = f"https://api.telegram.org/bot{TOKEN_TELEGRAM}/sendMessage"
    try: requests.post(url, data={"chat_id": ID_TELEGRAM, "text": mensagem}, timeout=5)
    except: pass

def load_data_brapi(ticker, interval="5m"):
    t = ticker.strip().upper().replace(".SA", "")
    # Mudança estratégica: Pedimos o range de 5 dias para garantir que a API tenha dados para calcular as médias
    url = f"https://brapi.dev/api/quote/{t}?range=5d&interval={interval}&token={TOKEN_BRAPI}"
    
    try:
        response = requests.get(url, timeout=15).json()
        if 'results' in response:
            data_res = response['results'][0]
            if 'historicalData' in data_res:
                df = pd.DataFrame(data_res['historicalData'])
                df['date'] = pd.to_datetime(df['date'], unit='s')
                df = df.rename(columns={'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low', 'volume': 'Volume'})
                return df.set_index('date').sort_index()
        return pd.DataFrame()
    except:
        return pd.DataFrame()

def calculate_indicators(df):
    if len(df) < 22: return pd.DataFrame() # Precisa de dados suficientes para as médias
    data = df.copy()
    data["EMA9"] = data["Close"].ewm(span=9, adjust=False).mean()
    data["EMA21"] = data["Close"].ewm(span=21, adjust=False).mean()
    data["VolMA"] = data["Volume"].rolling(20).mean()
    data["RVOL"] = data["Volume"] / data["VolMA"]
    return data.dropna()

def evaluate_asset(ticker, df):
    t = df.iloc[-1]
    entry = float(t["Close"])
    
    score = 0
    if t["RVOL"] > 1.1: score += 50
    if t["Close"] > t["EMA9"]: score += 50
    
    decision = "OPERAR AGORA" if score >= 100 else ("OBSERVAR" if score >= 50 else "AGUARDAR")
    return {"ticker": ticker, "score": score, "decision": decision, "entry": entry}

# =========================================================
# INTERFACE
# =========================================================

def main():
    st.title("📈 Monitor de Elite B3 - Real Time")
    
    tickers_input = st.sidebar.text_input("Ações", "PETR4, VALE3, BBAS3, ITUB4")
    intervalo = st.sidebar.selectbox("Gráfico", ["1m", "5m", "15m", "30m", "1h"], index=1) # Padrão 5m
    
    if st.sidebar.button("🚀 INICIAR MONITORAMENTO"):
        tickers = [t.strip().upper() for t in tickers_input.split(",")]
        status = st.empty()
        tabela = st.empty()
        
        while True:
            # Horário de Brasília
            hora_br = (datetime.utcnow() - timedelta(hours=3)).strftime('%H:%M:%S')
            status.info(f"🔄 Varredura Ativa | Horário B3: {hora_br}")
            
            resultados = []
            for t in tickers:
                df = load_data_brapi(t, intervalo)
                if not df.empty:
                    df_ind = calculate_indicators(df)
                    if not df_ind.empty:
                        res = evaluate_asset(t, df_ind)
                        resultados.append(res)
            
            if resultados:
                df_res = pd.DataFrame(resultados).sort_values(by="score", ascending=False)
                tabela.table(df_res)
            else:
                tabela.error("⚠️ Erro de Conexão: A API não está enviando dados. Tente mudar o tempo gráfico ou as ações.")

            time.sleep(60)
            st.rerun()

if __name__ == "__main__":
    main()
