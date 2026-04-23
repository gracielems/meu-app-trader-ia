import time
import requests
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st

# =========================================================
# CONFIGURAÇÕES
# =========================================================
st.set_page_config(page_title="Robô Elite B3", page_icon="🚀", layout="wide")

TOKEN_TELEGRAM = "8512230023:AAGyZ0QwPmnWqnZeQKfS7KCkwcVf1fBckCY"
ID_TELEGRAM = "7453152256"
TOKEN_BRAPI = "u4ufLNdYFG1Qo3xwHmuoem"

def enviar_telegram(dados):
    mensagem = (f"🚀 **SINAL: {dados['ticker']}**\nScore: {dados['score']}\nEntrada: {dados['entry']:.2f}")
    url = f"https://api.telegram.org/bot{TOKEN_TELEGRAM}/sendMessage"
    try: requests.post(url, data={"chat_id": ID_TELEGRAM, "text": mensagem}, timeout=5)
    except: pass

def load_data_brapi(ticker, interval="5m"):
    # Limpa o nome da ação para o padrão Brapi
    t = ticker.replace(".SA", "").strip().upper()
    # Para o intraday (5m), pedimos o range de 1 dia
    url = f"https://brapi.dev/api/quote/{t}?range=1d&interval={interval}&token={TOKEN_BRAPI}"
    try:
        response = requests.get(url, timeout=15).json()
        if 'results' in response:
            hist = response['results'][0].get('historicalData', [])
            if hist:
                df = pd.DataFrame(hist)
                df['date'] = pd.to_datetime(df['date'], unit='s')
                df = df.rename(columns={'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low', 'volume': 'Volume'})
                return df.set_index('date')
        return pd.DataFrame()
    except:
        return pd.DataFrame()

def calculate_indicators(df):
    data = df.copy()
    data["EMA9"] = data["Close"].ewm(span=9, adjust=False).mean()
    data["EMA21"] = data["Close"].ewm(span=21, adjust=False).mean()
    data["ATR"] = (data["High"] - data["Low"]).rolling(min(14, len(df))).mean()
    data["VolMA"] = data["Volume"].rolling(min(20, len(df))).mean()
    data["RVOL"] = data["Volume"] / data["VolMA"]
    return data.dropna()

def evaluate_asset(ticker, df):
    t = df.iloc[-1]
    entry = float(t["Close"])
    # Cálculo de Stop Seguro
    atr_val = float(t["ATR"]) if float(t["ATR"]) > 0 else entry * 0.01
    stop = entry - (1.5 * atr_val)
    target = entry + ((entry - stop) * 2.5)
    
    score = 0
    if t["RVOL"] > 1.1: score += 50
    if t["Close"] > t["EMA9"]: score += 50
    
    decision = "OPERAR AGORA" if score >= 100 else ("OBSERVAR" if score >= 50 else "AGUARDAR")
    return {"ticker": ticker, "score": score, "decision": decision, "entry": entry, "stop": stop, "target": target}

# =========================================================
# INTERFACE
# =========================================================

def main():
    st.title("📈 Monitor de Elite B3")
    
    tickers_input = st.sidebar.text_input("Ações (Sem .SA)", "PETR4, VALE3, BBAS3, ITUB4")
    intervalo = st.sidebar.selectbox("Gráfico", ["1m", "2m", "5m", "15m", "30m"], index=2)
    
    if st.sidebar.button("🚀 LIGAR ROBÔ"):
        tickers = [t.strip().upper() for t in tickers_input.split(",")]
        status = st.empty()
        tabela = st.empty()
        
        while True:
            # Ajuste de horário para Brasília (UTC-3)
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
                else:
                    st.toast(f"Sem dados para {t} ainda...", icon="⚠️")

            if resultados:
                df_res = pd.DataFrame(resultados)
                tabela.table(df_res)
            else:
                tabela.warning("O mercado está em leilão ou a API ainda não liberou os dados de hoje. Aguarde...")

            time.sleep(60)
            st.rerun()

if __name__ == "__main__":
    main()
