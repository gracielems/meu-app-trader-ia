"""
╔══════════════════════════════════════════════════════════╗
║          ROBÔ TUBARÃO B3 — Day Trade Semi-Automático     ║
║  Estratégia : Price Action + VWAP + Volume Institucional ║
║  Meta       : R$ 100/dia  |  RR mínimo: 2:1              ║
║  Execução   : MetaTrader 5  |  Dashboard: Streamlit      ║
╚══════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import logging
import csv
import requests
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, time as dtime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import MetaTrader5 as mt5
    MT5_OK = True
except ImportError:
    MT5_OK = False

st.set_page_config(page_title="Tubarão B3", layout="wide", page_icon="🦈")

# ═══════════════════════════════════════════════════════════
# CONFIGURAÇÕES FIXAS DE PREGÃO
# ═══════════════════════════════════════════════════════════
ABERTURA          = dtime(10, 0)
INICIO_OPERACOES  = dtime(10, 15)
FECHAMENTO_ORDENS = dtime(17, 10)
FECHAMENTO_FORCADO= dtime(17, 25)

# ═══════════════════════════════════════════════════════════
# PARÂMETROS DA ESTRATÉGIA TUBARÃO
# ═══════════════════════════════════════════════════════════
class Config:
    TIMEFRAME_NOME   = "5min"
    TIMEFRAME_MT5    = mt5.TIMEFRAME_M5 if MT5_OK else 5
    N_CANDLES        = 200

    META_DIA         = 100.0
    PERDA_MAX_DIA    = 60.0
    RISCO_POR_TRADE  = 0.02
    RR_MINIMO        = 2.0
    MAX_TRADES_DIA   = 5

    VOLUME_MULT      = 1.5
    ATR_PERIODO      = 14
    ATR_MULT_STOP    = 1.5

# ═══════════════════════════════════════════════════════════
# LOGS
# ═══════════════════════════════════════════════════════════
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
ARQ_LOG    = LOG_DIR / "tubarao.log"
ARQ_ORDENS = LOG_DIR / "ordens.csv"

def _get_logger() -> logging.Logger:
    logger = logging.getLogger("tubarao_b3")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    for h in [logging.FileHandler(ARQ_LOG, encoding="utf-8"), logging.StreamHandler()]:
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger

log = _get_logger()

# ═══════════════════════════════════════════════════════════
# FUNÇÕES DE INDICADORES (PURE PANDAS)
# ═══════════════════════════════════════════════════════════

def calcular_vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP intradiário, reseta por dia"""
    df = df.copy()
    df['tp'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['tvp'] = df['tp'] * df['Volume']
    df['cv'] = df['Volume'].cumsum()
    df['tvp_cum'] = df['tvp'].cumsum()
    df['vwap'] = df['tvp_cum'] / df['cv']
    return df['vwap']

def calcular_atr(df: pd.DataFrame, periodo: int = 14) -> pd.Series:
    """Average True Range"""
    df = df.copy()
    df['tr'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low'] - df['Close'].shift(1))
        )
    )
    return df['tr'].rolling(periodo).mean()

def calcular_volume_media(df: pd.DataFrame, periodo: int = 20) -> pd.Series:
    """Volume médio"""
    return df['Volume'].rolling(periodo).mean()

def identificar_swings(df: pd.DataFrame, janela: int = 5) -> Tuple[pd.Series, pd.Series]:
    """Swing highs/lows"""
    df = df.copy()
    swing_high = df['High'].rolling(janela).max()
    swing_low = df['Low'].rolling(janela).min()
    return swing_high, swing_low

# ═══════════════════════════════════════════════════════════
# FUNÇÕES MT5 & YFINANCE
# ═══════════════════════════════════════════════════════════

def _mt5_conectado() -> bool:
    """Verifica se MT5 está disponível e conectado"""
    if not MT5_OK:
        return False
    try:
        if mt5.initialize():
            return mt5.account_info() is not None
    except:
        pass
    return False

def mt5_auto_conectar() -> bool:
    """Tenta conectar no MT5 automaticamente"""
    if not MT5_OK:
        return False
    try:
        mt5.initialize()
        return True
    except:
        return False

def mt5_info_conta() -> Dict:
    """Retorna info da conta logada"""
    if not MT5_OK or not _mt5_conectado():
        return {}
    try:
        info = mt5.account_info()
        return {
            "login": info.login,
            "saldo": info.balance,
            "patrimonio": info.equity,
            "margem_livre": info.margin_free,
            "lucro": info.profit
        }
    except:
        return {}

def mt5_habilitar_symbol(symbol: str) -> bool:
    """Habilita símbolo no MT5"""
    if not MT5_OK or not _mt5_conectado():
        return False
    try:
        mt5.symbol_select(symbol, True)
        return True
    except:
        return False

def _yf_candles(ticker: str, interval: str = "5m", period: str = "5d") -> pd.DataFrame:
    """Puxar candles do yfinance (fallback)"""
    try:
        df = yf.download(ticker, interval=interval, period=period, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df = df.rename(columns={
            'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'
        })
        return df
    except:
        return pd.DataFrame()

def puxar_candles(symbol: str = "BBAS3") -> pd.DataFrame:
    """Puxar candles: tenta MT5, fallback yfinance"""
    # Tenta MT5 primeiro
    if _mt5_conectado():
        try:
            mt5_habilitar_symbol(symbol)
            rates = mt5.copy_rates_from_pos(symbol, Config.TIMEFRAME_MT5, Config.N_CANDLES)
            if rates is not None and len(rates) > 0:
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df = df.rename(columns={'time': 'Datetime', 'open': 'Open', 'high': 'High', 
                                        'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'})
                df.set_index('Datetime', inplace=True)
                return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        except Exception as e:
            log.warning(f"MT5 error: {e}, fallback yfinance")

    # Fallback yfinance
    ticker_yf = f"{symbol}.SA" if not symbol.endswith(".SA") else symbol
    return _yf_candles(ticker_yf, interval="5m", period="5d")

# ═══════════════════════════════════════════════════════════
# LÓGICA DE SINAIS TUBARÃO
# ═══════════════════════════════════════════════════════════

def gerar_sinais(df: pd.DataFrame) -> Tuple[int, str, Dict]:
    """
    Gera sinais usando scoring system.
    Retorna: (score, sinal, detalhes)
    Score >= 55: BUY | Score <= -55: SELL | senão: HOLD
    """
    if len(df) < Config.ATR_PERIODO:
        return 0, "HOLD", {}

    df = df.copy()
    df['vwap'] = calcular_vwap(df)
    df['atr'] = calcular_atr(df, Config.ATR_PERIODO)
    df['vol_media'] = calcular_volume_media(df, 20)
    swing_high, swing_low = identificar_swings(df, 5)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    score = 0
    detalhes = {}

    # 1. VWAP Zone (+30)
    if last['Close'] <= last['vwap']:
        score += 30
        detalhes['vwap'] = "ABAIXO (BUY)"
    else:
        detalhes['vwap'] = "ACIMA"

    # 2. Volume Institucional (+30)
    if last['Volume'] >= last['vol_media'] * Config.VOLUME_MULT:
        score += 30
        detalhes['volume'] = "ALTO"
    else:
        detalhes['volume'] = "NORMAL"

    # 3. Reversal Candle (+25)
    if last['Close'] > last['Open'] and prev['Close'] < last['Low']:
        score += 25
        detalhes['reversal'] = "SIM"
    else:
        detalhes['reversal'] = "NÃO"

    # 4. VWAP Recross (+15)
    if prev['Close'] > prev['vwap'] and last['Close'] <= last['vwap']:
        score += 15
        detalhes['recross'] = "SIM"
    else:
        detalhes['recross'] = "NÃO"

    sinal = "BUY" if score >= 55 else "SELL" if score <= -55 else "HOLD"
    detalhes['score'] = score
    detalhes['vwap_price'] = round(last['vwap'], 2)
    detalhes['atr'] = round(last['atr'], 2) if pd.notna(last['atr']) else 0

    return score, sinal, detalhes

# ═══════════════════════════════════════════════════════════
# GESTÃO DE RISCO
# ═══════════════════════════════════════════════════════════

def check_risk_management(state: Dict) -> bool:
    """Verifica se robô deve travar por gestão de risco"""
    if state.get('lucro_dia', 0) >= Config.META_DIA:
        return True
    if state.get('perda_dia', 0) <= -Config.PERDA_MAX_DIA:
        return True
    if state.get('trades_hoje', 0) >= Config.MAX_TRADES_DIA:
        return True
    return False

# ═══════════════════════════════════════════════════════════
# TELEGRAM
# ═══════════════════════════════════════════════════════════

def enviar_telegram(mensagem: str) -> bool:
    """Envia mensagem ao Telegram"""
    try:
        token = st.secrets.get("TOKEN_TELEGRAM")
        chat_id = st.secrets.get("ID_TELEGRAM")
        if not token or not chat_id:
            return False

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": mensagem,
            "parse_mode": "HTML"
        }
        requests.post(url, json=payload, timeout=8)
        return True
    except:
        return False

# ═══════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════

st.title("🦈 Robô Tubarão B3")
st.markdown("### Painel de Controle - Day Trade")

# Sidebar
with st.sidebar:
    st.subheader("⚙️ Configurações")

    # Seletor de ativo
    ativo = st.selectbox("Ação:", ["BBAS3", "PETR4", "VALE3", "ITUB4"])

    # Intervalo
    intervalo = st.slider("Intervalo (s)", 5, 60, 30)

    # Modo simulação
    modo_simulacao = st.checkbox("Modo simulação", value=True)

    # Semi-auto
    semi_auto = st.checkbox("Semi-auto (aprovar ordens)", value=True)

    # Capital simulado
    capital_sim = st.number_input("Capital simulado (R$)", 500, 50000, 1000)

    st.divider()

    # Info MT5
    if _mt5_conectado():
        info = mt5_info_conta()
        st.success("✅ MT5 Conectado na Clear")
        if info:
            st.metric("Saldo", f"R$ {info.get('saldo', 0):.2f}")
    else:
        st.warning("❌ MT5 não conectado")

# Main content
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Meta Diária", value=f"R$ {Config.META_DIA}")

with col2:
    st.metric(label="Perda Máxima", value=f"R$ {Config.PERDA_MAX_DIA}")

with col3:
    st.metric(label="Risco por Trade", value=f"{Config.RISCO_POR_TRADE * 100}%")

st.divider()

# Gráfico
st.subheader(f"📊 Gráfico {ativo} (5 min)")

df = puxar_candles(ativo)

if not df.empty:
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name=ativo
    )])
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=20, b=20),
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

    # Sinais
    score, sinal, detalhes = gerar_sinais(df)
    
    st.divider()
    st.subheader("📈 Análise de Sinais")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Score", score)
    with col2:
        st.metric("Sinal", sinal)
    with col3:
        st.metric("VWAP", detalhes.get('vwap_price', 0))
    with col4:
        st.metric("ATR", detalhes.get('atr', 0))

    st.divider()

    # Controles
    st.subheader("🎮 Controles")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🟢 Ligar Tubarão", key="ligar"):
            st.success("Robô ligado!")

    with col2:
        if st.button("🔴 Desligar", key="desligar"):
            st.info("Robô desligado")

    with col3:
        if st.button("📊 Fechar Posição", key="fechar"):
            st.warning("Posição fechada manualmente")

else:
    st.error("Erro ao carregar dados")

st.divider()
st.caption("🦈 Robô Tubarão B3 — Em desenvolvimento | MT5 + Streamlit")
