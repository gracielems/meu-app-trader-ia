import math
import time
import requests
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# =========================================================
# CONFIGURAÇÕES INICIAIS E TELEGRAM (PRONTO PARA USO)
# =========================================================
st.set_page_config(
    page_title="Robô Investidor de Elite B3",
    page_icon="🚀",
    layout="wide",
)

# --- DADOS CONECTADOS ---
TOKEN_TELEGRAM = "8512230023:AAGyZ0QwPmnWqnZeQKfS7KCkwcVf1fBckCY"
ID_TELEGRAM = "7453152256" 

def enviar_telegram(dados):
    """Envia o alerta de trade formatado para o seu Telegram"""
    mensagem = (
        f"🚀 **SINAL DE ELITE: {dados['ticker']}**\n\n"
        f"📊 **Score:** {dados['score']}/100\n"
        f"🎯 **Setup:** {dados['setup_type'].upper()}\n"
        f"💰 **Entrada:** R$ {dados['entry']:.2f}\n"
        f"🛑 **Stop:** R$ {dados['stop']:.2f}\n"
        f"🏁 **Alvo:** R$ {dados['target']:.2f}\n\n"
        f"📈 **R:R:** {dados['risk_reward']:.2f}\n"
        f"💪 **Destaques:** {dados['strengths']}\n"
        f"⚠️ **Atenção:** {dados['alerts']}"
    )
    
    url = f"https://api.telegram.org/bot{TOKEN_TELEGRAM}/sendMessage"
    payload = {"chat_id": ID_TELEGRAM, "text": mensagem, "parse_mode": "Markdown"}
    
    try:
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        st.error(f"Erro ao disparar Telegram: {e}")

# =========================================================
# FUNÇÕES DE ANÁLISE TÉCNICA
# =========================================================
def normalize_ticker(raw: str) -> str:
    ticker = raw.strip().upper()
    if not ticker: return ""
    if "." not in ticker and ticker[-1].isdigit(): ticker += ".SA"
    return ticker

@st.cache_data(ttl=60) # Atualiza dados a cada 1 minuto
def load_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=str.title)
        return df.dropna()
    except:
        return pd.DataFrame()

def calculate_indicators(df: pd.DataFrame):
    data = df.copy()
    # Médias de Investidor Grande (Exponenciais)
    data["EMA9"] = data["Close"].ewm(span=9, adjust=False).mean()
    data["EMA21"] = data["Close"].ewm(span=21, adjust=False).mean()
    
    # Volatilidade e Volume (Rastro do Dinheiro)
    data["ATR14"] = (data["High"] - data["Low"]).rolling(14).mean()
    data["VolumeMA20"] = data["Volume"].rolling(20).mean()
    data["RVOL"] = data["Volume"] / data["VolumeMA20"]
    return data.dropna()

def evaluate_asset(ticker, df_daily, df_trigger):
    t = df_trigger.iloc[-1]
    d = df_daily.iloc[-1]
    
    # Cálculo de Gerenciamento de Risco
    entry = float(t["Close"])
    stop = entry - (1.5 * float(t["ATR14"]))
    risk = entry - stop
    target = entry + (risk * 2.5) # Busca lucro 2.5x maior que o risco
    rr = (target - entry) / risk if risk > 0 else 0
    
    score = 0
    strengths = []
    blocks = []
    
    # 1. VOLUME: Tem que ter "peixe grande" entrando (Volume 30% acima da média)
    if t["RVOL"] > 1.3:
        score += 40
        strengths.append("Volume Institucional")
    else:
        blocks.append("Volume Baixo (Sem Tubarão)")
    
    # 2. TENDÊNCIA INTRADAY: Preço acima das médias curtas
    if t["Close"] > t["EMA9"] > t["EMA21"]:
        score += 30
        strengths.append("Tendência Alinhada")
    
    # 3. FILTRO MACRO: No gráfico diário, a ação tem que estar subindo
    if d["Close"] > d["EMA21"]:
        score += 30
        strengths.append("Tendência Diária de Alta")
    else:
        blocks.append("Contra Tendência Diária")

    # Decisão Final
    decision = "NÃO OPERAR"
    if not blocks and score >= 85:
        decision = "OPERAR AGORA"
    elif score >= 65:
        decision = "OBSERVAR"

    return {
        "ticker": ticker, "score": score, "decision": decision,
        "setup_type": "Day Trade (Smart Money)", "entry": entry, "stop": stop,
        "target": target, "risk_reward": rr, "strengths": " | ".join(strengths),
        "alerts": " | ".join(blocks) if blocks else "Cenário Limpo",
    }

# =========================================================
# INTERFACE DO USUÁRIO (STREAMLIT)
# =========================================================
def main():
    st.title("📈 Robô Investidor de Elite - Uso Pessoal")
    st.subheader("Foco: Day Trade com Volume e Lucro Real")
    st.markdown("---")
    
    st.sidebar.header("⚙️ Painel de Controle")
    tickers_input = st.sidebar.text_input("Ações (Ex: PETR4, VALE3, ITUB4)", "PETR4, VALE3, BBAS3, ITUB4, WEGE3")
    intervalo = st.sidebar.selectbox("Tempo do Gráfico", ["5m", "15m", "30m", "60m"], index=1)
    
    if "alertas_enviados" not in st.session_state:
        st.session_state.alertas_enviados = set()

    if st.sidebar.button("🚀 LIGAR ROBÔ E MONITORAR"):
        tickers_list = [normalize_ticker(t) for t in tickers_input.split(",")]
        status_msg = st.empty()
        tabela_resumo = st.empty()
        
        while True:
            agora = datetime.now().strftime('%H:%M:%S')
            status_msg.info(f"🔄 **Varredura Ativa...** Última checagem às {agora}")
            resultados = []
            
            for ticker in tickers_list:
                # Carrega Diário para o "Macro" e Intraday para o "Gatilho"
                df_daily = load_data(ticker, "1y", "1d")
                df_trigger = load_data(ticker, "1mo", intervalo)
                
                if not df_daily.empty and not df_trigger.empty:
                    df_daily = calculate_indicators(df_daily)
                    df_trigger = calculate_indicators(df_trigger)
                    
                    res = evaluate_asset(ticker, df_daily, df_trigger)
                    resultados.append(res)
                    
                    # DISPARO PARA O TELEGRAM (Somente se for "OPERAR AGORA")
                    if res["decision"] == "OPERAR AGORA":
                        # Garante que só envia 1 vez por hora para não travar o celular
                        chave = f"{ticker}_{datetime.now().strftime('%H')}"
                        if chave not in st.session_state.alertas_enviados:
                            enviar_telegram(res)
                            st.session_state.alertas_enviados.add(chave)
            
            if resultados:
                df_final = pd.DataFrame(resultados).sort_values(by="score", ascending=False)
                tabela_resumo.table(df_final[['ticker', 'score', 'decision', 'entry', 'stop', 'target', 'alerts']])
            
            time.sleep(60) # Espera 60 segundos e reinicia
            st.rerun()

if __name__ == "__main__":
    main()
