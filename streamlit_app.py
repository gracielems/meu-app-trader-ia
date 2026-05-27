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
import requests as _requests
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, time as dtime, date
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
# PARÂMETROS DA ESTRATÉGIA
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
# INDICADORES
# ═══════════════════════════════════════════════════════════

def calcular_vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP intraday."""
    if df.empty:
        return pd.Series(index=df.index, dtype=float)
    
    df_vwap = df.copy()
    df_vwap['tp'] = (df_vwap['high'] + df_vwap['low'] + df_vwap['close']) / 3
    df_vwap['cum_tp_vol'] = (df_vwap['tp'] * df_vwap['volume']).cumsum()
    df_vwap['cum_vol'] = df_vwap['volume'].cumsum()
    return df_vwap['cum_tp_vol'] / df_vwap['cum_vol']


def calcular_atr(df: pd.DataFrame, periodo: int = 14) -> pd.Series:
    """ATR."""
    df_atr = df.copy()
    df_atr['tr'] = np.maximum(
        df_atr['high'] - df_atr['low'],
        np.maximum(
            np.abs(df_atr['high'] - df_atr['close'].shift()),
            np.abs(df_atr['low'] - df_atr['close'].shift())
        )
    )
    return df_atr['tr'].rolling(window=periodo).mean()


def calcular_volume_media(df: pd.DataFrame, periodo: int = 20) -> pd.Series:
    """Volume médio."""
    return df['volume'].rolling(window=periodo).mean()


# ═══════════════════════════════════════════════════════════
# MT5
# ═══════════════════════════════════════════════════════════

def mt5_conectado() -> bool:
    """Verifica MT5."""
    if not MT5_OK:
        return False
    try:
        info = mt5.account_info()
        return info is not None
    except:
        return False


def mt5_habilitar_symbol(symbol: str) -> bool:
    """Habilita símbolo."""
    if not MT5_OK or not mt5_conectado():
        return False
    try:
        return mt5.symbol_select(symbol, True)
    except:
        return False


def mt5_info_conta() -> Dict:
    """Info da conta."""
    if not MT5_OK or not mt5_conectado():
        return {}
    try:
        info = mt5.account_info()
        return {
            'login': info.login,
            'saldo': info.balance,
            'patrimonio': info.equity,
            'margem_livre': info.margin_free,
            'lucro': info.profit
        }
    except:
        return {}


def _yf_candles(symbol: str, interval: str = "5m", period: str = "5d") -> pd.DataFrame:
    """Fallback yfinance."""
    try:
        df = yf.download(symbol, interval=interval, period=period, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df = df.rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Volume': 'volume'
        })
        return df.dropna()
    except:
        return pd.DataFrame()


def obter_candles(symbol: str = "BBAS3", n: int = 200) -> Optional[pd.DataFrame]:
    """Obtém candles."""
    df = None
    
    if MT5_OK and mt5_conectado():
        try:
            mt5_habilitar_symbol(symbol)
            candles = mt5.copy_rates_from_pos(symbol, Config.TIMEFRAME_MT5, 0, n)
            if candles is not None and len(candles) > 0:
                df = pd.DataFrame(candles)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df = df.set_index('time')
                df = df[['open', 'high', 'low', 'close', 'tick_volume']]
                df = df.rename(columns={'tick_volume': 'volume'})
                return df
        except:
            pass
    
    return _yf_candles(symbol + ".SA", interval="5m", period="5d")


# ═══════════════════════════════════════════════════════════
# TELEGRAM
# ═══════════════════════════════════════════════════════════

def tg_enviar(texto: str) -> bool:
    """Envia Telegram."""
    try:
        token = st.secrets.get("TOKEN_TELEGRAM", "")
        chat_id = st.secrets.get("ID_TELEGRAM", "")
        if not token or not chat_id:
            return False
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = {"chat_id": chat_id, "text": texto, "parse_mode": "HTML"}
        _requests.post(url, json=data, timeout=8)
        return True
    except:
        return False


def tg_resultado(symbol: str, resultado: float, meta_dia: float):
    """Alerta resultado."""
    pct = (resultado / meta_dia) * 100 if meta_dia > 0 else 0
    barra = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
    texto = f"""
📊 <b>RESULTADO - {symbol}</b>
P&L: R$ {resultado:.2f}
Progress: [{barra}] {pct:.0f}%
    """
    tg_enviar(texto)


def tg_alerta_risco():
    """Alerta risco."""
    texto = "⚠️ ROBÔ TRAVADO - Limite de risco atingido!"
    tg_enviar(texto)


# ═══════════════════════════════════════════════════════════
# RISCO
# ═══════════════════════════════════════════════════════════

def check_risk_management() -> Tuple[bool, str]:
    """Verifica risco."""
    if 'pl_dia' not in st.session_state:
        return True, "OK"
    
    pl = st.session_state['pl_dia']
    trades = st.session_state.get('trades_hoje', 0)
    
    if pl >= Config.META_DIA:
        return False, "✅ Meta atingida"
    if pl <= -Config.PERDA_MAX_DIA:
        tg_alerta_risco()
        return False, "❌ Perda máxima"
    if trades >= Config.MAX_TRADES_DIA:
        return False, f"❌ Máx {Config.MAX_TRADES_DIA} trades"
    
    return True, "✓ Operando"


# ═══════════════════════════════════════════════════════════
# INTERFACE
# ═══════════════════════════════════════════════════════════

st.title("🦈 Robô Tubarão B3")
st.markdown("### Painel de Controle - Day Trade")

if 'pl_dia' not in st.session_state:
    st.session_state.pl_dia = 0.0
if 'trades_hoje' not in st.session_state:
    st.session_state.trades_hoje = 0
if 'rodando' not in st.session_state:
    st.session_state.rodando = False
if 'modo_sim' not in st.session_state:
    st.session_state.modo_sim = True

st.divider()

# Sidebar
with st.sidebar:
    st.subheader("⚙️ Configurações")
    
    mt5_status = "✅ Conectado" if mt5_conectado() else "❌ Desconectado"
    st.write(f"**MT5:** {mt5_status}")
    
    if mt5_conectado():
        info = mt5_info_conta()
        st.write(f"**Saldo:** R$ {info.get('saldo', 0):.2f}")
        st.write(f"**P&L:** R$ {info.get('lucro', 0):.2f}")
    
    st.divider()
    
    symbol = st.selectbox("Ativo", ["BBAS3", "PETR4", "VALE3", "ITUB4"])
    intervalo = st.slider("Intervalo (s)", 5, 60, 30)
    
    st.divider()
    
    st.subheader("🎮 Modo")
    st.session_state.modo_sim = st.checkbox("🔵 Simulação", value=True)
    st.session_state.rodando = st.checkbox("Ativo", value=False)
    
    st.divider()
    
    st.metric("P&L Hoje", f"R$ {st.session_state.pl_dia:.2f}")
    st.metric("Trades", st.session_state.trades_hoje)

# Dashboard
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Meta", f"R$ {Config.META_DIA}")

with col2:
    st.metric("Stop", f"R$ {Config.PERDA_MAX_DIA}")

with col3:
    st.metric("Risco", f"{Config.RISCO_POR_TRADE * 100}%")

st.divider()

st.subheader(f"📊 {symbol} (5 min)")

df = obter_candles(symbol, Config.N_CANDLES)

if df is not None and not df.empty:
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name=symbol
    )])
    
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

st.divider()

pode_operar, motivo = check_risk_management()
status_cor = "🟢" if pode_operar else "🔴"
st.write(f"**Status:** {status_cor} {motivo}")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("▶️ Ligar", use_container_width=True):
        st.session_state.rodando = True

with col2:
    if st.button("⏸️ Desligar", use_container_width=True):
        st.session_state.rodando = False

with col3:
    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.pl_dia = 0.0
        st.session_state.trades_hoje = 0

st.divider()

def rerun_seguro(delay: float = 0.5) -> None:
    if st.session_state.get("_rerun_bloqueado"):
if st.session_state.rodando:
    agora = datetime.now().time()
    
    if agora < ABERTURA or agora > FECHAMENTO_FORCADO:
        st.warning("⏰ Mercado fechado")
        time.sleep(60)
        st.rerun()
        # O 'return' foi removido daqui
        
    if agora >= INICIO_OPERACOES and agora <= FECHAMENTO_ORDENS:
        st.info(f"🟢 Monitorando... ({intervalo}s)")
        rerun_seguro(intervalo)
        
    if agora > FECHAMENTO_ORDENS:
        st.warning("🔔 Fechando...")
        rerun_seguro(1)

log.info("Tubarão online")
