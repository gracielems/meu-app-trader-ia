"""
╔══════════════════════════════════════════════════════════╗
║         ROBÔ TUBARÃO B3 — Day Trade Semi-Automático      ║
║  Estratégia : Price Action + VWAP + Volume Institucional ║
║  Meta       : R$ 100/dia  |  RR mínimo: 2:1             ║
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
INICIO_OPERACOES  = dtime(10, 15)   # aguarda estabilizar após abertura
FECHAMENTO_ORDENS = dtime(17, 10)   # para de abrir novas posições
FECHAMENTO_FORCADO= dtime(17, 25)   # fecha tudo — day trade não vira swing


# ═══════════════════════════════════════════════════════════
# PARÂMETROS DA ESTRATÉGIA TUBARÃO
# ═══════════════════════════════════════════════════════════
class Config:
    # Timeframe principal
    TIMEFRAME_NOME   = "5min"
    TIMEFRAME_MT5    = mt5.TIMEFRAME_M5 if MT5_OK else 5
    N_CANDLES        = 200

    # Gestão de risco DIÁRIA — inegociável
    META_DIA         = 100.0    # R$ — para ao atingir
    PERDA_MAX_DIA    = 60.0     # R$ — trava o robô se perder isso no dia
    RISCO_POR_TRADE  = 0.02     # 2% do capital por operação
    RR_MINIMO        = 2.0      # Take Profit = 2x o Stop Loss (RR 2:1)
    MAX_TRADES_DIA   = 5        # limite de operações por dia

    # Volume institucional
    VOLUME_MULT      = 1.5      # candle com volume > 1.5x a média = institucional

    # ATR para stop dinâmico
    ATR_PERIODO      = 14
    ATR_MULT_STOP    = 1.5      # stop = 1.5x ATR


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


def salvar_ordem(tipo: str, ticker: str, preco: float, qtd: int,
                 motivo: str, simulacao: bool, resultado: float = 0.0) -> None:
    modo = "SIM" if simulacao else "REAL"
    novo = not ARQ_ORDENS.exists()
    with open(ARQ_ORDENS, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if novo:
            w.writerow(["data", "hora", "modo", "tipo", "ticker",
                        "preco", "qtd", "total", "resultado", "motivo"])
        w.writerow([date.today().isoformat(), datetime.now().strftime("%H:%M:%S"),
                    modo, tipo, ticker, f"{preco:.2f}", qtd,
                    f"{preco*qtd:.2f}", f"{resultado:.2f}", motivo])
    log.info(f"[{modo}] {tipo} {qtd}x {ticker} @ R${preco:.2f} | {motivo}")


# ═══════════════════════════════════════════════════════════
# TELEGRAM
# ═══════════════════════════════════════════════════════════
def tg_send(msg: str, token: str, chat_id: str) -> None:
    if not token:
        return
    try:
        _requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": msg, "parse_mode": "HTML"},
            timeout=8,
        )
    except Exception as e:
        log.warning(f"Telegram: {e}")


def tg_entrada(ticker: str, tipo: str, preco: float, qtd: int,
               sl: float, tp: float, motivo: str, sim: bool) -> str:
    emoji = "🟢" if tipo == "COMPRA" else "🔴"
    modo  = "🧪 SIM" if sim else "💰 REAL"
    rr    = round(abs(tp - preco) / abs(preco - sl), 1) if preco != sl else 0
    return (
        f"{modo} · {emoji} <b>{tipo}</b>\n\n"
        f"📌 <code>{ticker}</code> · R$ {preco:.2f}\n"
        f"🎯 TP: R$ {tp:.2f}  |  🛑 SL: R$ {sl:.2f}\n"
        f"⚖️ RR: 1:{rr}  |  📦 {qtd} ações\n"
        f"💡 {motivo}\n"
        f"🕐 {datetime.now().strftime('%H:%M:%S')}"
    )


def tg_resultado(ticker: str, pl_trade: float, pl_dia: float,
                 n_trades: int, motivo: str, sim: bool) -> str:
    emoji = "💰" if pl_trade >= 0 else "🔻"
    sinal = "+" if pl_trade >= 0 else ""
    modo  = "🧪 SIM" if sim else "💰 REAL"
    barra_dia = "▓" * int(min(pl_dia / Config.META_DIA * 10, 10)) + \
                "░" * max(0, 10 - int(pl_dia / Config.META_DIA * 10))
    return (
        f"{modo} · {emoji} <b>FECHAMENTO</b>\n\n"
        f"📌 <code>{ticker}</code>\n"
        f"Trade: <b>R$ {sinal}{pl_trade:.2f}</b>\n"
        f"Dia  : <b>R$ {pl_dia:+.2f}</b> / meta R$ {Config.META_DIA:.0f}\n"
        f"[{barra_dia}] {pl_dia/Config.META_DIA*100:.0f}%\n"
        f"🔁 {n_trades} trades · {motivo}\n"
        f"🕐 {datetime.now().strftime('%H:%M:%S')}"
    )


def tg_alerta_risco(motivo: str, pl_dia: float, sim: bool) -> str:
    modo = "🧪 SIM" if sim else "💰 REAL"
    return (
        f"{modo} · ⚠️ <b>ROBÔ TRAVADO</b>\n\n"
        f"🛑 {motivo}\n"
        f"💵 P&L do dia: R$ {pl_dia:+.2f}\n"
        f"🕐 {datetime.now().strftime('%H:%M:%S')}"
    )


# ═══════════════════════════════════════════════════════════
# GESTÃO DE RISCO DIÁRIA — INEGOCIÁVEL
# ═══════════════════════════════════════════════════════════
def check_risk_management() -> Tuple[bool, str]:
    """Verifica se o robô pode operar com base nas regras diárias.

    Returns:
        (pode_operar: bool, motivo: str)
    """
    pl_dia   = st.session_state.get("pl_dia", 0.0)
    n_trades = st.session_state.get("n_trades_dia", 0)

    # Meta atingida — protege o lucro
    if pl_dia >= Config.META_DIA:
        return False, f"🎯 Meta diária atingida (R$ {pl_dia:.2f}). Parabéns! Robô encerrado."

    # Perda máxima — protege o capital
    if pl_dia <= -Config.PERDA_MAX_DIA:
        return False, f"🛑 Perda máxima diária atingida (R$ {pl_dia:.2f}). Capital protegido."

    # Limite de trades
    if n_trades >= Config.MAX_TRADES_DIA:
        return False, f"🔁 Limite de {Config.MAX_TRADES_DIA} trades/dia atingido."

    # Horário
    agora = datetime.now().time()
    if agora < INICIO_OPERACOES:
        return False, f"⏳ Aguardando estabilização do mercado (início: {INICIO_OPERACOES.strftime('%H:%M')})"
    if agora >= FECHAMENTO_ORDENS:
        return False, f"🔒 Sem novas entradas após {FECHAMENTO_ORDENS.strftime('%H:%M')}"

    return True, "OK"


# ═══════════════════════════════════════════════════════════
# MT5 — CONEXÃO E EXECUÇÃO (Clear Corretora)
# ═══════════════════════════════════════════════════════════
# Clear usa ORDER_FILLING_RETURN — obrigatório para ações B3
FILLING_MODE = mt5.ORDER_FILLING_RETURN if MT5_OK else 0


def mt5_conectar() -> bool:
    """Conecta ao MT5. O app deve estar aberto e logado na Clear."""
    if not MT5_OK:
        return False
    if not mt5.initialize():
        log.error(f"MT5 init falhou: {mt5.last_error()}")
        return False
    info = mt5.account_info()
    if not info:
        log.error("MT5: abra o MetaTrader 5 e faça login na Clear.")
        return False
    log.info(f"MT5 OK | Login:{info.login} | {info.company} | "
             f"Saldo:R${info.balance:.2f} | Patrimônio:R${info.equity:.2f}")
    return True


def mt5_auto_conectar() -> bool:
    """Tenta conectar automaticamente ao iniciar o app."""
    if not MT5_OK or st.session_state.get("mt5_ok"):
        return st.session_state.get("mt5_ok", False)
    ok = mt5_conectar()
    if ok:
        st.session_state.mt5_ok = True
    return ok


def mt5_info_conta() -> Optional[Dict]:
    """Retorna saldo, patrimônio e lucro da conta MT5."""
    if not MT5_OK:
        return None
    info = mt5.account_info()
    if not info:
        return None
    return {
        "login":        info.login,
        "corretora":    info.company,
        "saldo":        info.balance,
        "patrimonio":   info.equity,
        "margem_livre": info.margin_free,
        "lucro":        info.profit,
    }


def mt5_habilitar_symbol(ticker: str) -> bool:
    """Garante que o símbolo está visível no Market Watch do MT5."""
    if not MT5_OK:
        return True
    info = mt5.symbol_info(ticker)
    if info is None:
        log.error(f"Símbolo {ticker} não encontrado. Verifique no MT5.")
        return False
    if not info.visible:
        if not mt5.symbol_select(ticker, True):
            log.error(f"Falha ao habilitar {ticker} no MT5.")
            return False
    return True


def mt5_saldo() -> float:
    if not MT5_OK or not st.session_state.get("mt5_ok"):
        return float(st.session_state.get("capital_sim", 1000.0))
    info = mt5.account_info()
    return info.balance if info else 0.0


def _yf_candles(ticker: str, n: int) -> pd.DataFrame:
    """Busca candles de 5min via yfinance (fallback quando MT5 não está disponível).

    yfinance disponibiliza dados intraday dos últimos 60 dias com interval='5m'.
    """
    import yfinance as yf
    t = ticker.strip().upper()
    if not t.endswith(".SA"):
        t = f"{t}.SA"
    try:
        df = yf.download(t, period="5d", interval="5m",
                         progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        # Flatten MultiIndex se necessário
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index().rename(columns={"Datetime": "time", "index": "time"})
        if "time" not in df.columns and "Date" in df.columns:
            df = df.rename(columns={"Date": "time"})
        df["time"] = pd.to_datetime(df["time"])
        # Garante que Volume existe
        if "Volume" not in df.columns:
            df["Volume"] = 1
        return df[["time", "Open", "High", "Low", "Close", "Volume"]].tail(n).reset_index(drop=True)
    except Exception as e:
        log.error(f"yfinance erro para {t}: {e}")
        return pd.DataFrame()


def mt5_candles(ticker: str, n: int) -> pd.DataFrame:
    """Busca candles do MT5 (real) ou yfinance (simulação no Streamlit Cloud)."""
    if not MT5_OK:
        return _yf_candles(ticker, n)
    mt5_habilitar_symbol(ticker)
    rates = mt5.copy_rates_from_pos(ticker, Config.TIMEFRAME_MT5, 0, n)
    if rates is None:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df.rename(columns={"close": "Close", "open": "Open",
                               "high": "High", "low": "Low",
                               "tick_volume": "Volume"})


def mt5_preco_ask(ticker: str) -> float:
    """Retorna o preço ask atual. No modo sim, usa o último Close do yfinance."""
    if not MT5_OK:
        df = _yf_candles(ticker, 1)
        return float(df["Close"].iloc[-1]) if not df.empty else 0.0
    tick = mt5.symbol_info_tick(ticker)
    return tick.ask if tick else 0.0


def mt5_preco_bid(ticker: str) -> float:
    """Retorna o preço bid atual. No modo sim, usa o último Close do yfinance."""
    if not MT5_OK:
        df = _yf_candles(ticker, 1)
        return float(df["Close"].iloc[-1]) if not df.empty else 0.0
    tick = mt5.symbol_info_tick(ticker)
    return tick.bid if tick else 0.0


def mt5_abrir_compra(ticker: str, qtd: int, sl: float, tp: float,
                     simulacao: bool) -> Optional[float]:
    if simulacao:
        p = mt5_preco_ask(ticker) or st.session_state.get("preco_sim", 10.0)
        log.info(f"[SIM] COMPRA {qtd}x {ticker} @ R${p:.2f} SL:{sl:.2f} TP:{tp:.2f}")
        return p
    if not mt5_habilitar_symbol(ticker):
        log.error(f"Símbolo {ticker} não habilitado — ordem cancelada")
        return None
    preco = mt5_preco_ask(ticker)
    req = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       ticker,
        "volume":       float(qtd),
        "type":         mt5.ORDER_TYPE_BUY,
        "price":        preco,
        "sl":           sl,
        "tp":           tp,
        "deviation":    10,
        "magic":        20250101,
        "comment":      "Tubarao B3",
        "type_time":    mt5.ORDER_TIME_DAY,
        "type_filling": FILLING_MODE,
    }
    res = mt5.order_send(req)
    if res.retcode != mt5.TRADE_RETCODE_DONE:
        log.error(f"COMPRA falhou: {res.retcode} {res.comment}")
        return None
    return res.price


def mt5_fechar_posicao(ticker: str, qtd: int, simulacao: bool,
                       motivo: str = "") -> Optional[float]:
    if simulacao:
        p = mt5_preco_bid(ticker) or st.session_state.get("preco_sim", 10.0)
        log.info(f"[SIM] VENDA {qtd}x {ticker} @ R${p:.2f} | {motivo}")
        return p
    positions = mt5.positions_get(symbol=ticker)
    if not positions:
        log.warning(f"Sem posição aberta em {ticker}")
        return None
    pos   = positions[0]
    preco = mt5_preco_bid(ticker)
    req = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       ticker,
        "volume":       float(pos.volume),
        "type":         mt5.ORDER_TYPE_SELL,
        "position":     pos.ticket,
        "price":        preco,
        "deviation":    10,
        "magic":        20250101,
        "comment":      motivo or "Tubarao B3",
        "type_time":    mt5.ORDER_TIME_DAY,
        "type_filling": FILLING_MODE,
    }
    res = mt5.order_send(req)
    if res.retcode != mt5.TRADE_RETCODE_DONE:
        log.error(f"VENDA falhou: {res.retcode} {res.comment}")
        return None
    return res.price


# ═══════════════════════════════════════════════════════════
# INDICADORES TUBARÃO (pandas puro)
# ═══════════════════════════════════════════════════════════
def calcular_vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP intradiário — reseta a cada dia.

    Preço justo do dia. Comprar abaixo = comprar barato (como institucional).
    Vender acima = realizar com lucro.
    """
    df = df.copy()
    preco_tipico = (df["High"] + df["Low"] + df["Close"]) / 3
    df["_data"]  = pd.to_datetime(df["time"]).dt.date if "time" in df.columns else date.today()

    vwap_values = []
    for _, grupo in df.groupby("_data"):
        vol_cum = grupo["Volume"].cumsum()
        pvol    = (preco_tipico.loc[grupo.index] * grupo["Volume"]).cumsum()
        vwap    = pvol / vol_cum.replace(0, np.nan)
        vwap_values.append(vwap)

    return pd.concat(vwap_values).reindex(df.index)


def calcular_atr(df: pd.DataFrame, periodo: int = 14) -> pd.Series:
    """Average True Range — mede a volatilidade real do mercado."""
    hl  = df["High"] - df["Low"]
    hc  = (df["High"] - df["Close"].shift()).abs()
    lc  = (df["Low"]  - df["Close"].shift()).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(span=periodo, adjust=False).mean()


def calcular_volume_media(df: pd.DataFrame, periodo: int = 20) -> pd.Series:
    """Média móvel do volume — base para identificar volume institucional."""
    return df["Volume"].rolling(periodo).mean()


def identificar_swings(df: pd.DataFrame, janela: int = 5) -> pd.DataFrame:
    """Identifica swing highs e lows (zonas de liquidez).

    Grandes players deixam ordens paradas nessas regiões.
    """
    df = df.copy()
    df["swing_high"] = df["High"][(df["High"] == df["High"].rolling(janela, center=True).max())]
    df["swing_low"]  = df["Low"][(df["Low"]  == df["Low"].rolling(janela, center=True).min())]
    return df


def calcular_todos_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula todos os indicadores da estratégia Tubarão."""
    if df.empty or len(df) < 50:
        return df
    df = df.copy()
    df["vwap"]        = calcular_vwap(df)
    df["atr"]         = calcular_atr(df, Config.ATR_PERIODO)
    df["vol_media"]   = calcular_volume_media(df)
    df["vol_rel"]     = df["Volume"] / df["vol_media"]          # > 1.5 = institucional
    df["acima_vwap"]  = df["Close"] > df["vwap"]
    df = identificar_swings(df)

    # Corpo e direção do candle
    df["corpo"]       = df["Close"] - df["Open"]
    df["corpo_pct"]   = df["corpo"] / df["Open"] * 100
    df["tocha_alta"]  = (df["Close"] > df["Open"])              # candle de alta
    df["tocha_baixa"] = (df["Close"] < df["Open"])              # candle de baixa

    return df


# ═══════════════════════════════════════════════════════════
# ENGINE DE SINAL — ESTRATÉGIA TUBARÃO
# ═══════════════════════════════════════════════════════════
def analisar_sinal_tubarao(df: pd.DataFrame, capital: float) -> Dict:
    """Analisa o fluxo institucional e retorna sinal de entrada.

    Lógica Tubarão:
    - COMPRA: preço recua ao VWAP/suporte + toque em swing low +
              volume institucional + candle de reversão de alta
    - VENDA : preço chega a resistência/swing high + volume institucional
              + candle de reversão de baixa + acima do VWAP
    """
    if df.empty or len(df) < 3:
        return {"tipo": "NEUTRO", "forca": 0, "motivo": "Dados insuficientes"}

    for col in ["vwap", "atr", "vol_rel"]:
        if col not in df.columns or df[col].isna().all():
            return {"tipo": "NEUTRO", "forca": 0, "motivo": "Aguardando indicadores"}

    u  = df.iloc[-1]   # candle atual
    p  = df.iloc[-2]   # candle anterior
    p2 = df.iloc[-3]   # 2 candles atrás

    preco     = float(u["Close"])
    vwap      = float(u["vwap"])
    atr       = float(u["atr"])
    vol_rel   = float(u["vol_rel"])

    if pd.isna(vwap) or pd.isna(atr) or atr == 0:
        return {"tipo": "NEUTRO", "forca": 0, "motivo": "VWAP/ATR indisponíveis"}

    # Stop e alvo calculados pelo ATR (dinâmico)
    stop_dist  = atr * Config.ATR_MULT_STOP
    take_dist  = stop_dist * Config.RR_MINIMO

    # Quantidade de ações pelo risco por trade
    risco_r    = capital * Config.RISCO_POR_TRADE
    qtd        = max(1, int(risco_r / stop_dist))

    # Variáveis de contexto
    vol_institucional = vol_rel >= Config.VOLUME_MULT
    perto_vwap        = abs(preco - vwap) / vwap < 0.005   # dentro de 0.5% do VWAP
    abaixo_vwap       = preco < vwap
    acima_vwap        = preco > vwap

    # Recuo ao VWAP (zone de interesse dos institucionais)
    recuo_vwap_compra = abaixo_vwap and p["acima_vwap"]   # cruzou abaixo
    recuo_vwap_venda  = acima_vwap  and not p["acima_vwap"]  # cruzou acima

    # Candle de reversão
    reversao_alta  = (u["tocha_alta"] and not p["tocha_alta"]   # virou verde
                      and float(u["Close"]) > float(p["High"]))   # rompeu a máxima anterior
    reversao_baixa = (u["tocha_baixa"] and not p["tocha_baixa"]  # virou vermelho
                      and float(u["Close"]) < float(p["Low"]))    # rompeu a mínima anterior

    # ── SINAL DE COMPRA ────────────────────────────────────────────────
    pontos_compra = 0
    motivos_compra = []

    if abaixo_vwap or perto_vwap:
        pontos_compra += 30
        motivos_compra.append("Preço no VWAP (zona institucional)")
    if vol_institucional:
        pontos_compra += 30
        motivos_compra.append(f"Volume institucional ({vol_rel:.1f}x a média)")
    if reversao_alta:
        pontos_compra += 25
        motivos_compra.append("Candle de reversão de alta")
    if recuo_vwap_compra:
        pontos_compra += 15
        motivos_compra.append("Recuo ao VWAP")

    # ── SINAL DE VENDA ─────────────────────────────────────────────────
    pontos_venda = 0
    motivos_venda = []

    if acima_vwap or perto_vwap:
        pontos_venda += 30
        motivos_venda.append("Preço acima do VWAP (resistência)")
    if vol_institucional:
        pontos_venda += 30
        motivos_venda.append(f"Volume institucional ({vol_rel:.1f}x a média)")
    if reversao_baixa:
        pontos_venda += 25
        motivos_venda.append("Candle de reversão de baixa")
    if recuo_vwap_venda:
        pontos_venda += 15
        motivos_venda.append("Toque acima do VWAP")

    # Retorna o sinal mais forte (mínimo 55 pontos para entrar)
    LIMIAR = 55

    if pontos_compra >= LIMIAR and pontos_compra > pontos_venda:
        sl = round(preco - stop_dist, 2)
        tp = round(preco + take_dist, 2)
        return {
            "tipo":   "COMPRA",
            "forca":  min(100, pontos_compra),
            "preco":  preco,
            "sl":     sl,
            "tp":     tp,
            "qtd":    qtd,
            "vwap":   round(vwap, 2),
            "atr":    round(atr, 2),
            "vol_rel": round(vol_rel, 2),
            "motivo": " + ".join(motivos_compra),
        }

    if pontos_venda >= LIMIAR and pontos_venda > pontos_compra:
        sl = round(preco + stop_dist, 2)
        tp = round(preco - take_dist, 2)
        return {
            "tipo":   "VENDA",
            "forca":  min(100, pontos_venda),
            "preco":  preco,
            "sl":     sl,
            "tp":     tp,
            "qtd":    qtd,
            "vwap":   round(vwap, 2),
            "atr":    round(atr, 2),
            "vol_rel": round(vol_rel, 2),
            "motivo": " + ".join(motivos_venda),
        }

    return {
        "tipo":    "NEUTRO",
        "forca":   0,
        "preco":   preco,
        "vwap":    round(vwap, 2),
        "atr":     round(atr, 2),
        "vol_rel": round(vol_rel, 2),
        "motivo":  f"Aguardando confluência (compra:{pontos_compra}pts venda:{pontos_venda}pts)",
    }


# ═══════════════════════════════════════════════════════════
# INTERFACE — COMPONENTES
# ═══════════════════════════════════════════════════════════
def _barra_meta(pl: float) -> str:
    """Barra ASCII de progresso da meta diária."""
    pct   = max(-100, min(100, pl / Config.META_DIA * 100))
    total = 20
    cheio = int(abs(pct) / 100 * total)
    if pl >= 0:
        return f"[{'█' * cheio}{'░' * (total - cheio)}] {pl:+.2f} / R${Config.META_DIA:.0f}"
    return f"[{'▓' * cheio}{'░' * (total - cheio)}] {pl:+.2f} / -R${Config.PERDA_MAX_DIA:.0f}"


def exibir_painel_risco() -> None:
    pl_dia   = st.session_state.get("pl_dia", 0.0)
    n_trades = st.session_state.get("n_trades_dia", 0)
    pode, _  = check_risk_management()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("P&L do dia",    f"R$ {pl_dia:+.2f}",
              delta_color="normal")
    c2.metric("Meta diária",   f"R$ {Config.META_DIA:.0f}")
    c3.metric("Perda máx.",    f"R$ {Config.PERDA_MAX_DIA:.0f}")
    c4.metric("Trades hoje",   f"{n_trades}/{Config.MAX_TRADES_DIA}")
    c5.metric("Status",        "✅ Operando" if pode else "🔒 Travado")

    # Barra de progresso
    cor = "green" if pl_dia >= 0 else "red"
    st.markdown(
        f"<p style='font-family:monospace; color:{cor}; font-size:13px;'>{_barra_meta(pl_dia)}</p>",
        unsafe_allow_html=True,
    )


def exibir_sinal_tubarao(sinal: Dict) -> None:
    tipo  = sinal.get("tipo", "NEUTRO")
    forca = sinal.get("forca", 0)
    preco = sinal.get("preco", 0)
    vwap  = sinal.get("vwap", 0)
    atr   = sinal.get("atr", 0)
    vr    = sinal.get("vol_rel", 0)

    if tipo == "COMPRA":
        st.success(f"### 🟢 {tipo}  ·  Força: {forca}%")
    elif tipo == "VENDA":
        st.error(f"### 🔴 {tipo}  ·  Força: {forca}%")
    else:
        st.info(f"### ⚪ Aguardando sinal  ({forca}pts)")

    if forca:
        st.progress(forca / 100)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Preço",  f"R$ {preco:.2f}")
    c2.metric("VWAP",   f"R$ {vwap:.2f}")
    c3.metric("ATR",    f"R$ {atr:.2f}")
    c4.metric("Volume", f"{vr:.1f}x {'🐳' if vr >= Config.VOLUME_MULT else ''}")

    if tipo in ("COMPRA", "VENDA"):
        sl   = sinal.get("sl", 0)
        tp   = sinal.get("tp", 0)
        qtd  = sinal.get("qtd", 0)
        rr   = abs(tp - preco) / abs(preco - sl) if preco != sl else 0
        st.write(f"**🎯 TP:** R$ {tp:.2f} · **🛑 SL:** R$ {sl:.2f} · "
                 f"**⚖️ RR:** 1:{rr:.1f} · **📦 Qtd:** {qtd} ações")

    st.caption(f"💡 {sinal.get('motivo', '')}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main() -> None:
    # ── Session state ──────────────────────────────────────
    defaults = {
        "rodando":        False,
        "em_posicao":     False,
        "preco_entrada":  0.0,
        "sl_entrada":     0.0,
        "tp_entrada":     0.0,
        "qtd_posicao":    0,
        "pl_dia":         0.0,
        "n_trades_dia":   0,
        "ultimo_sinal":   None,
        "sinal_pendente": None,
        "ciclos":         0,
        "mt5_ok":         False,
        "capital_sim":    1000.0,
        "travado":        False,
        "motivo_trava":   "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Guard: evita st.rerun() duplo na mesma execução (causa erro de DOM no Streamlit)
    if st.session_state.get("_rerun_bloqueado"):
        st.session_state["_rerun_bloqueado"] = False
        return

    def rerun_seguro(delay: float = 0.5) -> None:
        """Aguarda um tempo mínimo antes de rerun para evitar conflito de DOM."""
        st.session_state["_rerun_bloqueado"] = True
        time.sleep(delay)
        st.rerun()

    # ── Credenciais ────────────────────────────────────────
    try:
        tg_token = st.secrets["TOKEN_TELEGRAM"]
        tg_id    = st.secrets["ID_TELEGRAM"]
    except Exception:
        tg_token = tg_id = ""

    # ── Cabeçalho ──────────────────────────────────────────
    st.title("🦈 Robô Tubarão B3 — Day Trade")
    agora = datetime.now()
    hora  = agora.time()

    if hora < ABERTURA:
        st.warning(f"🕙 Mercado abre às {ABERTURA.strftime('%H:%M')}")
    elif hora >= FECHAMENTO_FORCADO:
        st.info("🔒 Pregão encerrado")
    else:
        st.success(f"🟢 Pregão aberto · {agora.strftime('%H:%M:%S')}")

    st.divider()
    exibir_painel_risco()
    st.divider()

    # ── Sidebar ────────────────────────────────────────────
    with st.sidebar:
        st.header("🦈 Configurações")

        if not MT5_OK:
            st.error("MetaTrader5 não instalado:\n```\npip install MetaTrader5\n```")
        else:
            if st.button("🔌 Conectar MT5", use_container_width=True,
                         disabled=st.session_state.mt5_ok):
                if mt5_conectar():
                    st.session_state.mt5_ok = True
                    pass  # Streamlit já faz rerun ao clicar em botão
            if st.session_state.mt5_ok:
                st.success(f"✅ MT5 · R$ {mt5_saldo():,.2f}")

        st.divider()
        ticker      = st.text_input("Ação", "BBAS3")
        intervalo   = st.slider("Intervalo (s)", 15, 120, 30, 15)
        modo_sim    = st.toggle("Modo simulação", True)
        modo_semi   = st.toggle("Semi-auto (aprovar ordens)", True)

        if modo_sim:
            st.session_state.capital_sim = st.number_input(
                "Capital simulado (R$)", 500, 100000,
                int(st.session_state.capital_sim), 500)
            st.success("🧪 Paper trade")
        else:
            st.warning("⚠️ ORDENS REAIS")

        st.divider()
        st.caption(f"🎯 Meta: R$ {Config.META_DIA:.0f}/dia")
        st.caption(f"🛑 Perda máx: R$ {Config.PERDA_MAX_DIA:.0f}/dia")
        st.caption(f"⚖️ RR mínimo: 1:{Config.RR_MINIMO:.0f}")
        st.caption(f"📊 Risco/trade: {Config.RISCO_POR_TRADE*100:.0f}%")
        st.caption(f"🔁 Ciclos: {st.session_state.ciclos}")

    # ── Botões principais ──────────────────────────────────
    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("▶️ Ligar Tubarão", type="primary", use_container_width=True,
                     disabled=st.session_state.rodando or st.session_state.travado):
            if not modo_sim and not st.session_state.mt5_ok:
                st.error("Conecte o MT5 antes de rodar no modo real.")
            else:
                st.session_state.rodando      = True
                st.session_state.pl_dia       = 0.0
                st.session_state.n_trades_dia = 0
                st.session_state.travado      = False
                log.info(f"Tubarão LIGADO | {ticker} | sim={modo_sim}")
                tg_send(
                    f"🦈 <b>Tubarão LIGADO</b>\n📌 <code>{ticker}</code>\n"
                    f"🎯 Meta: R${Config.META_DIA:.0f} · 🛑 Perda max: R${Config.PERDA_MAX_DIA:.0f}\n"
                    f"{'🧪 Simulação' if modo_sim else '💰 REAL'}",
                    tg_token, tg_id,
                )
                pass  # Streamlit já faz rerun ao clicar em botão

    with c2:
        if st.button("⏹️ Desligar", use_container_width=True,
                     disabled=not st.session_state.rodando):
            st.session_state.rodando = False
            log.info("Tubarão DESLIGADO")
            pass  # Streamlit já faz rerun ao clicar em botão

    with c3:
        if st.button("🔒 Fechar Agora", use_container_width=True,
                     disabled=not st.session_state.em_posicao):
            p = mt5_fechar_posicao(ticker.upper(), st.session_state.qtd_posicao,
                                   modo_sim, "Fechamento manual")
            if p:
                res = (p - st.session_state.preco_entrada) * st.session_state.qtd_posicao
                st.session_state.pl_dia       += res
                st.session_state.n_trades_dia += 1
                st.session_state.em_posicao    = False
                salvar_ordem("VENDA", ticker.upper(), p, st.session_state.qtd_posicao,
                             "Fechamento manual", modo_sim, res)
                tg_send(tg_resultado(ticker.upper(), res, st.session_state.pl_dia,
                                    st.session_state.n_trades_dia, "Manual", modo_sim),
                        tg_token, tg_id)
            pass  # Streamlit já faz rerun ao clicar em botão

    # ── Aprovação manual ────────────────────────────────────
    if modo_semi and st.session_state.sinal_pendente:
        sinal = st.session_state.sinal_pendente
        st.divider()
        st.subheader("🔔 Aguardando sua aprovação")
        exibir_sinal_tubarao(sinal)

        ca, cr = st.columns(2)
        with ca:
            if st.button("✅ Executar ordem", type="primary", use_container_width=True):
                tipo = sinal["tipo"]
                sym  = ticker.upper()
                capital = mt5_saldo() if st.session_state.mt5_ok else float(st.session_state.capital_sim)
                qtd  = sinal.get("qtd", 1)
                sl   = sinal["sl"]
                tp   = sinal["tp"]

                if tipo == "COMPRA" and not st.session_state.em_posicao:
                    p = mt5_abrir_compra(sym, qtd, sl, tp, modo_sim)
                    if p:
                        st.session_state.em_posicao    = True
                        st.session_state.preco_entrada = p
                        st.session_state.sl_entrada    = sl
                        st.session_state.tp_entrada    = tp
                        st.session_state.qtd_posicao   = qtd
                        salvar_ordem("COMPRA", sym, p, qtd, sinal["motivo"], modo_sim)
                        tg_send(tg_entrada(sym, "COMPRA", p, qtd, sl, tp,
                                          sinal["motivo"], modo_sim), tg_token, tg_id)

                elif tipo == "VENDA" and st.session_state.em_posicao:
                    p = mt5_fechar_posicao(sym, st.session_state.qtd_posicao,
                                          modo_sim, sinal["motivo"])
                    if p:
                        res = (p - st.session_state.preco_entrada) * st.session_state.qtd_posicao
                        st.session_state.pl_dia       += res
                        st.session_state.n_trades_dia += 1
                        st.session_state.em_posicao    = False
                        salvar_ordem("VENDA", sym, p, st.session_state.qtd_posicao,
                                    sinal["motivo"], modo_sim, res)
                        tg_send(tg_resultado(sym, res, st.session_state.pl_dia,
                                            st.session_state.n_trades_dia,
                                            sinal["motivo"], modo_sim), tg_token, tg_id)

                st.session_state.sinal_pendente = None
                pass  # Streamlit já faz rerun ao clicar em botão

        with cr:
            if st.button("❌ Recusar", use_container_width=True):
                log.info(f"Sinal {sinal['tipo']} recusado")
                st.session_state.sinal_pendente = None
                pass  # Streamlit já faz rerun ao clicar em botão

    # ── Último sinal ────────────────────────────────────────
    st.divider()
    st.subheader("📡 Leitura do mercado")
    if st.session_state.ultimo_sinal:
        exibir_sinal_tubarao(st.session_state.ultimo_sinal)
    else:
        st.info("Aguardando primeiro ciclo de análise…")

    # ── Ordens e log ───────────────────────────────────────
    tab_ordens, tab_log = st.tabs(["📋 Ordens do dia", "🗒️ Log"])
    with tab_ordens:
        if ARQ_ORDENS.exists():
            df_o = pd.read_csv(ARQ_ORDENS)
            hoje = date.today().isoformat()
            dh   = df_o[df_o["data"] == hoje] if not df_o.empty else pd.DataFrame()
            if not dh.empty:
                st.dataframe(dh, use_container_width=True, hide_index=True)
            else:
                st.info("Nenhuma ordem hoje.")
    with tab_log:
        if ARQ_LOG.exists():
            linhas = ARQ_LOG.read_text(encoding="utf-8").strip().splitlines()
            st.code("\n".join(linhas[-40:]), language="text")

    # ═══════════════════════════════════════════════════════
    # LOOP PRINCIPAL
    # ═══════════════════════════════════════════════════════
    if not st.session_state.rodando:
        return

    hora  = datetime.now().time()
    sym   = ticker.upper()
    capital = mt5_saldo() if st.session_state.mt5_ok else float(st.session_state.capital_sim)

    # ── Fechamento forçado 17:25 ───────────────────────────
    if hora >= FECHAMENTO_FORCADO:
        if st.session_state.em_posicao:
            p = mt5_fechar_posicao(sym, st.session_state.qtd_posicao,
                                   modo_sim, "Fechamento forçado 17:25")
            if p:
                res = (p - st.session_state.preco_entrada) * st.session_state.qtd_posicao
                st.session_state.pl_dia       += res
                st.session_state.n_trades_dia += 1
                st.session_state.em_posicao    = False
                salvar_ordem("VENDA", sym, p, st.session_state.qtd_posicao,
                            "Fechamento forçado 17:25", modo_sim, res)
                tg_send(tg_resultado(sym, res, st.session_state.pl_dia,
                                    st.session_state.n_trades_dia,
                                    "Fechamento forçado 17:25", modo_sim), tg_token, tg_id)
        st.session_state.rodando = False
        log.info("Robô encerrado — fim do pregão")
        rerun_seguro(1)
        return

    # Fora do pregão
    if hora < ABERTURA:
        time.sleep(60); st.rerun(); return

    # ── Check de risco — INEGOCIÁVEL ───────────────────────
    pode, motivo_risco = check_risk_management()
    if not pode:
        if not st.session_state.travado:
            st.session_state.travado      = True
            st.session_state.motivo_trava = motivo_risco
            log.warning(f"TRAVA: {motivo_risco}")
            tg_send(tg_alerta_risco(motivo_risco, st.session_state.pl_dia, modo_sim),
                    tg_token, tg_id)
        st.warning(f"🔒 {motivo_risco}")
        # Fecha posição aberta se o motivo for perda máxima
        if st.session_state.em_posicao and "Perda máxima" in motivo_risco:
            p = mt5_fechar_posicao(sym, st.session_state.qtd_posicao,
                                   modo_sim, "Stop diário atingido")
            if p:
                res = (p - st.session_state.preco_entrada) * st.session_state.qtd_posicao
                st.session_state.pl_dia      += res
                st.session_state.em_posicao   = False
                salvar_ordem("VENDA", sym, p, st.session_state.qtd_posicao,
                            "Stop diário", modo_sim, res)
        st.session_state.rodando = False
        rerun_seguro(1)
        return

    # ── Ciclo de análise ───────────────────────────────────
    with st.spinner(f"🦈 Analisando {sym}…"):
        df = mt5_candles(sym, Config.N_CANDLES)

    if df.empty:
        log.warning(f"Sem candles para {sym}")
        rerun_seguro(intervalo); return

    df    = calcular_todos_indicadores(df)
    sinal = analisar_sinal_tubarao(df, capital)

    st.session_state.ultimo_sinal = sinal
    st.session_state.ciclos      += 1

    # ── Verifica stop/tp da posição aberta ─────────────────
    if st.session_state.em_posicao:
        p_atual = float(sinal.get("preco", st.session_state.preco_entrada))
        sl      = st.session_state.sl_entrada
        tp      = st.session_state.tp_entrada

        fechou, motivo_fechamento = False, ""
        if p_atual <= sl:
            fechou, motivo_fechamento = True, "Stop loss atingido"
        elif p_atual >= tp:
            fechou, motivo_fechamento = True, "Take profit atingido"
        elif sinal["tipo"] == "VENDA":
            if modo_semi:
                st.session_state.sinal_pendente = sinal
            else:
                fechou, motivo_fechamento = True, sinal["motivo"]

        if fechou and not (modo_semi and sinal["tipo"] == "VENDA"):
            p = mt5_fechar_posicao(sym, st.session_state.qtd_posicao,
                                   modo_sim, motivo_fechamento)
            if p:
                res = (p - st.session_state.preco_entrada) * st.session_state.qtd_posicao
                st.session_state.pl_dia       += res
                st.session_state.n_trades_dia += 1
                st.session_state.em_posicao    = False
                salvar_ordem("VENDA", sym, p, st.session_state.qtd_posicao,
                            motivo_fechamento, modo_sim, res)
                tg_send(tg_resultado(sym, res, st.session_state.pl_dia,
                                    st.session_state.n_trades_dia,
                                    motivo_fechamento, modo_sim), tg_token, tg_id)
            rerun_seguro(1)
            return

    # ── Abre nova posição (se não estiver em posição) ──────
    elif sinal["tipo"] == "COMPRA":
        if modo_semi:
            st.session_state.sinal_pendente = sinal
        else:
            p = mt5_abrir_compra(sym, sinal["qtd"], sinal["sl"], sinal["tp"], modo_sim)
            if p:
                st.session_state.em_posicao    = True
                st.session_state.preco_entrada = p
                st.session_state.sl_entrada    = sinal["sl"]
                st.session_state.tp_entrada    = sinal["tp"]
                st.session_state.qtd_posicao   = sinal["qtd"]
                salvar_ordem("COMPRA", sym, p, sinal["qtd"], sinal["motivo"], modo_sim)
                tg_send(tg_entrada(sym, "COMPRA", p, sinal["qtd"],
                                  sinal["sl"], sinal["tp"], sinal["motivo"], modo_sim),
                        tg_token, tg_id)

    rerun_seguro(max(intervalo, 1))


if __name__ == "__main__":
    main()
