import time
import requests
from datetime import datetime
import pandas as pd
import streamlit as st

# =========================================================
# CONFIGURAÇÕES DE ACESSO (CONECTADO)
# =========================================================
st.set_page_config(page_title="Robô Elite B3 - Tempo Real", page_icon="🚀", layout="wide")

TOKEN_TELEGRAM = "8512230023:AAGyZ0QwPmnWqnZeQKfS7KCkwcVf1fBckCY"
ID_TELEGRAM = "7453152256"
TOKEN_BRAPI = "u4ufLNdYFG1Qo3xwHmuoem"

# =========================================================
# FUNÇÕES DE COMUNICAÇÃO E DADOS
# =========================================================

def enviar_telegram(dados):
    """Envia o alerta formatado para o Telegram"""
    mensagem = (
        f"🚀 **SINAL DE ELITE: {dados['ticker']}**\n\n"
        f"📊 **Score:** {dados['score']}/100\n"
        f"💰 **Entrada:** R$ {dados['entry']:.2f}\n"
        f"🛑 **Stop:** R$ {dados['stop']:.2f}\n"
        f"🏁 **Alvo:** R$ {dados['target']:.2f}\n"
        f"📈 **R:R:** {dados['risk_reward']:.2f}\n\n"
        f"💪 **Destaques:** {dados['strengths']}\n"
        f"⚠️ **Filtro:** {dados['alerts']}"
    )
    url = f"https://api.telegram.org/bot{TOKEN_TELEGRAM}/sendMessage"
    payload = {"chat_id": ID_TELEGRAM, "text": mensagem, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=payload, timeout=10)
    except:
        pass

def load_data_brapi(ticker, interval="15m"):
    """Busca dados em tempo real via Brapi"""
    # Formata o ticker para o padrão da Brapi (sem o .SA se necessário)
    t = ticker.replace(".SA", "")
    range_val = "1d" if interval in ["1m", "5m", "15m"] else "5d"
    
    url = f"https://brapi.dev/api/quote/{t}?range={range_val}&interval={interval}&token={TOKEN_BRAPI}"
    try:
        response = requests.get(url).json()
        results = response['results'][0]['historicalData']
        df = pd.DataFrame(results)
        df['date'] = pd.to_datetime(df['date'], unit='s')
        df = df.rename(columns={'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low', 'volume': 'Volume'})
        df = df.set_index('date')
        return df
    except Exception as e:
        return pd.DataFrame()

# =========================================================
# INTELIGÊNCIA E ESTRATÉGIA
# =========================================================

def calculate_indicators(df):
    data = df.copy()
    data["EMA9"] = data["Close"].ewm(span=9, adjust=False).mean()
    data["EMA21"] = data["Close"].ewm(span=21, adjust=False).mean()
    data["ATR"] = (data["High"] - data["Low"]).rolling(14).mean()
    data["VolMA"] = data["Volume"].rolling(20).mean()
    data["RVOL"] = data["Volume"] / data["VolMA"]
    return data.dropna()

def evaluate_asset(ticker, df):
    t = df.iloc[-1]
    
    entry = float(t["Close"])
    stop = entry - (1.5 * float(t["ATR"]))
    target = entry + ((entry - stop) * 2.5)
    rr = (target - entry) / (entry - stop) if (entry - stop) > 0 else 0
    
    score = 0
    strengths = []
    blocks = []
    
    # 1. Volume (O rastro do dinheiro institucional)
    if t["RVOL"] > 1.2:
        score += 50
        strengths.append("Fluxo Institucional")
    else:
        blocks.append("Volume Normal")

    # 2. Tendência (Médias Alinhadas)
    if t["Close"] > t["EMA9"] > t["EMA21"]:
        score += 50
        strengths.append("Tendência de Alta")
    else:
        blocks.append("Sem Tendência Clara")

    decision = "OPERAR AGORA" if score >= 90 else ("OBSERVAR" if score >= 50 else "NÃO OPERAR")

    return {
        "ticker": ticker, "score": score, "decision": decision,
        "entry": entry, "stop": stop, "target": target, "risk_reward": rr,
        "strengths": " | ".join(strengths), "alerts": " | ".join(blocks)
    }

# =========================================================
# INTERFACE
# =========================================================

def main():
    st.title("📈 Robô Elite B3 - Tempo Real (Brapi)")
    st.sidebar.header("Configurações")
    tickers_raw = st.sidebar.text_input("Ações", "PETR4, VALE3, BBAS3, ITUB4")
    intervalo = st.sidebar.selectbox("Tempo Gráfico", ["5m", "15m", "30m", "1h"], index=1)
    
    if "alertas" not in st.session_state:
        st.session_state.alertas = set()

    if st.sidebar.button("🚀 INICIAR MONITORAMENTO REAL-TIME"):
        tickers = [t.strip().upper() for t in tickers_raw.split(",")]
        status = st.empty()
        tabela = st.empty()
        
      while True:
            # Pega o horário atual para mostrar que está vivo
            agora = datetime.now().strftime('%H:%M:%S')
            status.info(f"🔄 **Monitoramento Ativo em Tempo Real** | Última varredura: {agora}")
            
            # ... (resto do código de análise igual) ...
            
            # Em vez de apenas o rerun no final, vamos garantir o loop infinito
            time.sleep(60)
            # O Streamlit rerun atualiza a página inteira
            st.rerun()
            st.rerun()

if __name__ == "__main__":
    main()
