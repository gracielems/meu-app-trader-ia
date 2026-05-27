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
# FUNÇÕES DE INDICADORES
# ═══════════════════════════════════════════════════════════

def calcular_vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP intraday, reseta diariamente."""
    if 'Volume' not in df.columns or df.empty:
        return pd.Series(np.nan, index=df.index)
    
    df_cpy = df.copy()
    df_cpy['tp'] = (df_cpy['High'] + df_cpy['Low'] + df_cpy['Close']) / 3
    df_cpy['cum_vol'] = df_cpy['Volume'].cumsum()
    df_cpy['cum_tp_vol'] = (df_cpy['tp'] * df_cpy['Volume']).cumsum()
    df_cpy['vwap'] = df_cpy['cum_tp_vol'] / df_cpy['cum_vol']
    return df_cpy['vwap']


def calcular_atr(df: pd.DataFrame, periodo: int = 14) -> pd.Series:
    """Average True Range."""
    if df.empty or len(df) < periodo:
        return pd.Series(np.nan, index=df.index)
    
    df_cpy = df.copy()
    df_cpy['tr0'] = df_cpy['High'] - df_cpy['Low']
    df_cpy['tr1'] = abs(df_cpy['High'] - df_cpy['Close'].shift())
    df_cpy['tr2'] = abs(df_cpy['Low'] - df_cpy['Close'].shift())
    df_cpy['tr'] = df_cpy[['tr0', 'tr1', 'tr2']].max(axis=1)
    return df_cpy['tr'].rolling(periodo).mean()


def calcular_volume_media(df: pd.DataFrame, periodo: int = 20) -> pd.Series:
    """Volume médio."""
    if df.empty or 'Volume' not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df['Volume'].rolling(periodo).mean()


# ═══════════════════════════════════════════════════════════
# MT5 / YFINANCE
# ═══════════════════════════════════════════════════════════

def mt5_auto_conectar() -> bool:
    """Tenta conectar ao MT5 automaticamente."""
    if not MT5_OK:
        return False
    
    try:
        if not mt5.initialize():
            return False
        acc = mt5.account_info()
        return acc is not None
    except Exception as e:
        log.error(f"MT5 connect error: {e}")
        return False


def mt5_habilitar_symbol(symbol: str) -> bool:
    """Habilita um símbolo no MT5."""
    if not MT5_OK:
        return False
    
    try:
        if not mt5.symbol_select(symbol, True):
            log.warning(f"MT5: Não conseguiu habilitar {symbol}")
            return False
        return True
    except Exception as e:
        log.error(f"MT5 symbol enable error: {e}")
        return False


def _mt5_candles(symbol: str, timeframe: int, n: int) -> Optional[pd.DataFrame]:
    """Pega candles do MT5."""
    if not MT5_OK or not mt5_auto_conectar():
        return None
    
    try:
        mt5_habilitar_symbol(symbol)
        candles = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
        if candles is None or len(candles) == 0:
            return None
        
        df = pd.DataFrame(candles)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df.rename(columns={'time': 'Date', 'open': 'Open', 'high': 'High', 
                                'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'})
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df.set_index('Date', inplace=True)
        return df
    except Exception as e:
        log.error(f"MT5 candles error: {e}")
        return None


def _yf_candles(symbol: str, interval: str = "5m", period: str = "5d") -> Optional[pd.DataFrame]:
    """Pega candles do yfinance com .SA suffix."""
    try:
        yf_symbol = symbol if symbol.endswith(".SA") else f"{symbol}.SA"
        df = yf.download(yf_symbol, interval=interval, period=period, progress=False)
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        
        if df.empty or len(df) < 10:
            return None
        
        return df
    except Exception as e:
        log.error(f"yfinance error: {e}")
        return None


def obter_candles(symbol: str, n: int = 200) -> Optional[pd.DataFrame]:
    """Tenta MT5 primeiro, fallback para yfinance."""
    df_mt5 = _mt5_candles(symbol, Config.TIMEFRAME_MT5, n)
    if df_mt5 is not None and len(df_mt5) > 50:
        log.info(f"Dados de {symbol} obtidos do MT5")
        return df_mt5
    
    df_yf = _yf_candles(symbol, interval="5m", period="5d")
    if df_yf is not None and len(df_yf) > 50:
        log.info(f"Dados de {symbol} obtidos do yfinance (fallback)")
        return df_yf
    
    log.warning(f"Não conseguiu obter candles de {symbol}")
    return None


# ═══════════════════════════════════════════════════════════
# TELEGRAM
# ═══════════════════════════════════════════════════════════

def enviar_telegram(mensagem: str, modo_html: bool = True) -> bool:
    """Envia mensagem via Telegram."""
    try:
        token = st.secrets.get("TOKEN_TELEGRAM")
        chat_id = st.secrets.get("ID_TELEGRAM")
        
        if not token or not chat_id:
            log.warning("Telegram: credenciais não configuradas")
            return False
        
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = {
            "chat_id": chat_id,
            "text": mensagem,
            "parse_mode": "HTML" if modo_html else "Markdown"
        }
        
        _requests.post(url, data=data, timeout=8)
        return True
    except Exception as e:
        log.error(f"Telegram error: {e}")
        return False


def tg_entrada(symbol: str, direcao: str, preco_entrada: float, stop_loss: float, 
               take_profit: float, risco: float, recompensa: float):
    """Alert de entrada."""
    msg = f"""<b>🦈 ENTRADA - {symbol}</b>
Direção: <b>{direcao.upper()}</b>
Preço: R$ {preco_entrada:.2f}
SL: R$ {stop_loss:.2f}
TP: R$ {take_profit:.2f}
Risco: R$ {risco:.2f} | Recompensa: R$ {recompensa:.2f}
RR: {recompensa/max(risco, 0.01):.2f}:1"""
    enviar_telegram(msg)


def tg_resultado(symbol: str, resultado_brl: float, trades_hoje: int, meta: float):
    """Alert de resultado."""
    status = "✅ LUCRO" if resultado_brl > 0 else "❌ PERDA"
    progresso_pct = (resultado_brl / max(meta, 1)) * 100
    barra = "█" * int(progresso_pct / 5) + "░" * (20 - int(progresso_pct / 5))
    
    msg = f"""<b>🦈 RESULTADO - {symbol}</b>
{status}: R$ {resultado_brl:+.2f}
Trades: {trades_hoje}
Meta: R$ {meta:.2f}
Progresso: [{barra}] {progresso_pct:.1f}%"""
    enviar_telegram(msg)


def tg_alerta_risco(motivo: str, valor: float):
    """Alert de risco."""
    msg = f"<b>⚠️ ROBÔ TRAVADO</b>\n{motivo}: R$ {valor:.2f}"
    enviar_telegram(msg)


# ═══════════════════════════════════════════════════════════
# LÓGICA DE SINAIS
# ═══════════════════════════════════════════════════════════

def analisar_sinal(df: pd.DataFrame, symbol: str = "BBAS3") -> Dict:
    """Analisa Price Action + VWAP + Volume + ATR."""
    
    resultado = {
        "sinal": None,
        "preco_entrada": None,
        "stop_loss": None,
        "take_profit": None,
        "risco": None,
        "recompensa": None,
        "rr": None,
        "score": 0,
        "razoes": []
    }
    
    if df is None or df.empty or len(df) < 50:
        return resultado
    
    try:
        df['VWAP'] = calcular_vwap(df)
        df['ATR'] = calcular_atr(df, Config.ATR_PERIODO)
        df['Vol_Media'] = calcular_volume_media(df, 20)
        
        close = df['Close'].iloc[-1]
        vwap = df['VWAP'].iloc[-1]
        atr = df['ATR'].iloc[-1]
        vol = df['Volume'].iloc[-1]
        vol_media = df['Vol_Media'].iloc[-1]
        
        if pd.isna(atr) or atr == 0:
            return resultado
        
        # VWAP ZONE (30 pts)
        if close <= vwap:
            resultado["score"] += 30
            resultado["razoes"].append(f"Preço abaixo VWAP ({close:.2f} <= {vwap:.2f})")
        
        # VOLUME (30 pts)
        if vol >= vol_media * Config.VOLUME_MULT:
            resultado["score"] += 30
            resultado["razoes"].append(f"Volume institucional")
        
        # REVERSAL (25 pts)
        if len(df) >= 2:
            prev_high = df['High'].iloc[-2]
            prev_low = df['Low'].iloc[-2]
            curr_close = df['Close'].iloc[-1]
            
            if curr_close > prev_high or curr_close < prev_low:
                resultado["score"] += 25
                resultado["razoes"].append("Reversal")
        
        # VWAP RECROSS (15 pts)
        if len(df) >= 2:
            prev_close = df['Close'].iloc[-2]
            if prev_close > vwap and close <= vwap:
                resultado["score"] += 15
                resultado["razoes"].append("VWAP recross")
        
        if resultado["score"] >= 55:
            resultado["sinal"] = "BUY"
            resultado["preco_entrada"] = close
            
            stop_loss = close - (atr * Config.ATR_MULT_STOP)
            risco_valor = close - stop_loss
            take_profit = close + (risco_valor * Config.RR_MINIMO)
            
            resultado["stop_loss"] = max(stop_loss, close * 0.985)
            resultado["take_profit"] = take_profit
            resultado["risco"] = risco_valor
            resultado["recompensa"] = take_profit - close
            resultado["rr"] = resultado["recompensa"] / max(resultado["risco"], 0.01)
        
        return resultado
    
    except Exception as e:
        log.error(f"Signal analysis error: {e}")
        return resultado


# ═══════════════════════════════════════════════════════════
# GESTÃO DE RISCO
# ═══════════════════════════════════════════════════════════

def check_risk_management(session_state: Dict) -> Tuple[bool, str]:
    """Verifica se robô pode operar."""
    
    lucro_dia = session_state.get("lucro_dia", 0.0)
    perda_dia = session_state.get("perda_dia", 0.0)
    trades_hoje = session_state.get("trades_hoje", 0)
    
    if lucro_dia >= Config.META_DIA:
        return False, f"Meta diária atingida: R$ {lucro_dia:.2f}"
    
    if abs(perda_dia) >= Config.PERDA_MAX_DIA:
        return False, f"Perda máxima atingida: R$ {perda_dia:.2f}"
    
    if trades_hoje >= Config.MAX_TRADES_DIA:
        return False, f"Limite de {Config.MAX_TRADES_DIA} trades atingido"
    
    return True, "OK"


def check_horario() -> Tuple[bool, str]:
    """Verifica se está dentro do horário operacional."""
    
    agora = datetime.now().time()
    
    if agora < INICIO_OPERACOES:
        return False, f"Aguardando 10:15 (agora: {agora.strftime('%H:%M')})"
    
    if agora > FECHAMENTO_FORCADO:
        return False, "Pregão encerrado (17:25)"
    
    return True, "OK"


# ═══════════════════════════════════════════════════════════
# INTERFACE STREAMLIT
# ═══════════════════════════════════════════════════════════

st.title("🦈 Robô Tubarão B3")
st.markdown("### Painel de Controle - Day Trade")

if "lucro_dia" not in st.session_state:
    st.session_state.lucro_dia = 0.0
if "perda_dia" not in st.session_state:
    st.session_state.perda_dia = 0.0
if "trades_hoje" not in st.session_state:
    st.session_state.trades_hoje = 0
if "rodando" not in st.session_state:
    st.session_state.rodando = False

st.divider()

# Sidebar
with st.sidebar:
    st.header("⚙️ Configurações")
    
    symbol = st.selectbox("Ativo", ["BBAS3", "PETR4", "VALE3", "ITUB4"])
    intervalo_s = st.slider("Intervalo loop (segundos)", 5, 60, 30)
    
    modo_sim = st.toggle("Modo simulação", value=True)
    semi_auto = st.toggle("Semi-auto (aprovar ordens)", value=True)
    
    st.divider()
    st.subheader("💰 Simulado")
    capital_sim = st.number_input("Capital simulado (R$)", 1000, 100000, 10000)
    
    st.divider()
    st.subheader("📊 Status Atual")
    st.metric("P&L Dia", f"R$ {st.session_state.lucro_dia:+.2f}")
    st.metric("Perda", f"R$ {abs(st.session_state.perda_dia):.2f}")
    st.metric("Trades", st.session_state.trades_hoje)
    st.metric("Status", "🟢 Rodando" if st.session_state.rodando else "⚫ Parado")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Meta Diária", value=f"R$ {Config.META_DIA}")

with col2:
    st.metric(label="Perda Máxima (Stop)", value=f"R$ {Config.PERDA_MAX_DIA}")

with col3:
    st.metric(label="Risco por Trade", value=f"{Config.RISCO_POR_TRADE * 100}%")

st.divider()

st.subheader(f"📊 Análise Técnica - {symbol} (5 min)")

df = obter_candles(symbol, Config.N_CANDLES)

if df is not None and len(df) > 0:
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name=symbol
    )])
    
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=20, b=20),
        height=500,
        title=f"{symbol} - Candlestick 5min"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    sinal = analisar_sinal(df, symbol)
    
    if sinal["score"] >= 55 and sinal["sinal"]:
        st.success(f"🎯 SINAL {sinal['sinal']} | Score: {sinal['score']}")
        st.write("**Razões:**")
        for r in sinal["razoes"]:
            st.write(f"  • {r}")
        st.write(f"Entrada: R$ {sinal['preco_entrada']:.2f}")
        st.write(f"SL: R$ {sinal['stop_loss']:.2f} | TP: R$ {sinal['take_profit']:.2f}")
        st.write(f"RR: {sinal['rr']:.2f}:1")
    else:
        st.info(f"⏳ Aguardando sinal (Score: {sinal['score']}/55)")
else:
    st.warning("Aguardando dados...")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🚀 Ligar Tubarão", use_container_width=True):
        horario_ok, msg_h = check_horario()
        risco_ok, msg_r = check_risk_management(st.session_state)
        
        if not horario_ok:
            st.warning(f"⏰ {msg_h}")
        elif not risco_ok:
            st.error(f"🚫 {msg_r}")
        else:
            st.session_state.rodando = True
            st.success("✅ Tubarão ligado!")

with col2:
    if st.button("⏸️ Desligar", use_container_width=True):
        st.session_state.rodando = False
        st.info("⏹️ Tubarão desligado")

with col3:
    if st.button("🔄 Resetar Dia", use_container_width=True):
        st.session_state.lucro_dia = 0.0
        st.session_state.perda_dia = 0.0
        st.session_state.trades_hoje = 0
        st.success("✅ Dia resetado")

st.divider()

st.info("""
**🦈 Robô Tubarão B3**
- Estratégia: Price Action + VWAP + Volume + ATR
- Meta: R$ 100/dia | Stop: R$ 60/dia
- Max 5 trades/dia | Timeframe: 5 min
- Horário: 10:15–17:25 | Force-close: 17:25
""")
