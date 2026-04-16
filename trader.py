import math
from datetime import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf


st.set_page_config(
    page_title="Robô Pessoal de Análise de Ações — V6",
    page_icon="📈",
    layout="wide",
)

DEFAULT_TICKERS = "PETR4, VALE3, BBAS3, ITUB4, WEGE3, BBDC4, ABEV3, RENT3"

TRIGGER_PERIOD_OPTIONS = {
    "5 dias": "5d",
    "1 mês": "1mo",
    "3 meses": "3mo",
}

TRIGGER_INTERVAL_OPTIONS = {
    "5 minutos": "5m",
    "15 minutos": "15m",
    "30 minutos": "30m",
    "60 minutos": "60m",
    "Diário": "1d",
}

TREND_PERIOD_OPTIONS = {
    "6 meses": "6mo",
    "1 ano": "1y",
    "2 anos": "2y",
}

BACKTEST_PERIOD_OPTIONS = {
    "1 mês": "1mo",
    "3 meses": "3mo",
    "6 meses": "6mo",
    "1 ano": "1y",
}

PERIOD_ORDER = ["5d", "1mo", "3mo", "6mo", "1y", "2y"]
TIME_CHOICES = [f"{h:02d}:{m:02d}" for h in range(9, 19) for m in range(0, 60, 5)]

SIGNAL_COLORS = {
    "OPERAR AGORA": "🟢",
    "OBSERVAR": "🟡",
    "NÃO OPERAR": "🔴",
}

REGIME_COLORS = {
    "COMPRADOR": "🟢",
    "LATERAL": "🟡",
    "DEFENSIVO": "🔴",
}


# =========================================================
# UTILIDADES
# =========================================================
def normalize_ticker(raw: str) -> str:
    ticker = raw.strip().upper().replace(" ", "")
    if not ticker:
        return ""
    if ticker.startswith("^"):
        return ticker
    if "." not in ticker and ticker[-1].isdigit():
        ticker += ".SA"
    return ticker


def safe_float(value: str) -> Optional[float]:
    text = value.strip().replace(",", ".")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def format_pct(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"{value * 100:.2f}%"


def format_money(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"R$ {value:,.2f}"


def at_least_period(requested: str, minimum: str) -> str:
    req_idx = PERIOD_ORDER.index(requested) if requested in PERIOD_ORDER else 0
    min_idx = PERIOD_ORDER.index(minimum) if minimum in PERIOD_ORDER else 0
    return PERIOD_ORDER[max(req_idx, min_idx)]


def get_safe_backtest_period(backtest_period: str, trigger_interval: str) -> str:
    if trigger_interval == "1d":
        return backtest_period

    safe_caps = {
        "5m": "1mo",
        "15m": "1mo",
        "30m": "1mo",
        "60m": "3mo",
    }
    cap = safe_caps.get(trigger_interval, "1mo")
    req_idx = PERIOD_ORDER.index(backtest_period) if backtest_period in PERIOD_ORDER else 1
    cap_idx = PERIOD_ORDER.index(cap)
    return PERIOD_ORDER[min(req_idx, cap_idx)]


def parse_hhmm(value: str) -> time:
    h, m = value.split(":")
    return time(int(h), int(m))


def build_allowed_windows(
    use_time_filter: bool,
    morning_window: Tuple[str, str],
    afternoon_window: Tuple[str, str],
    use_afternoon: bool,
) -> List[Tuple[time, time]]:
    if not use_time_filter:
        return []

    windows = [(parse_hhmm(morning_window[0]), parse_hhmm(morning_window[1]))]
    if use_afternoon:
        windows.append((parse_hhmm(afternoon_window[0]), parse_hhmm(afternoon_window[1])))
    return windows


def within_any_window(ts, windows: List[Tuple[time, time]]) -> bool:
    if not windows:
        return True

    current = pd.Timestamp(ts)
    if current.tzinfo is not None:
        current = current.tz_localize(None)
    current_t = current.time()

    for start, end in windows:
        if start <= current_t <= end:
            return True
    return False


# =========================================================
# DADOS
# =========================================================
def _prepare_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=str.title).copy()

    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    if not required_cols.issubset(df.columns):
        return pd.DataFrame()

    df.index = pd.to_datetime(df.index, errors="coerce")
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)

    df = df[~df.index.isna()].sort_index().copy()
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    return df


@st.cache_data(ttl=900, show_spinner=False)
def load_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    tentativas = []

    try:
        df1 = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        tentativas.append(df1)
    except Exception:
        tentativas.append(pd.DataFrame())

    try:
        df2 = yf.Ticker(ticker).history(
            period=period,
            interval=interval,
            auto_adjust=False,
        )
        tentativas.append(df2)
    except Exception:
        tentativas.append(pd.DataFrame())

    for df in tentativas:
        prepared = _prepare_ohlcv(df)
        if not prepared.empty:
            return prepared

    return pd.DataFrame()


# =========================================================
# INDICADORES
# =========================================================
def calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def add_session_vwap(data: pd.DataFrame) -> pd.Series:
    idx = pd.to_datetime(data.index)
    session_key = pd.Series(idx.date, index=data.index)
    typical_price = (data["High"] + data["Low"] + data["Close"]) / 3
    tpv = typical_price * data["Volume"]
    cum_tpv = tpv.groupby(session_key).cumsum()
    cum_vol = data["Volume"].groupby(session_key).cumsum().replace(0, np.nan)
    return cum_tpv / cum_vol


def add_intraday_rvol_clock(data: pd.DataFrame) -> pd.Series:
    idx = pd.to_datetime(data.index)
    slot = idx.strftime("%H:%M")
    series = data["Volume"].copy()
    same_slot_avg = series.groupby(slot).transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean())
    return series / same_slot_avg.replace(0, np.nan)


def add_indicators(
    df: pd.DataFrame,
    interval: str,
    benchmark_close: Optional[pd.Series] = None,
) -> pd.DataFrame:
    data = df.copy()

    data["EMA9"] = data["Close"].ewm(span=9, adjust=False).mean()
    data["EMA21"] = data["Close"].ewm(span=21, adjust=False).mean()
    data["EMA50"] = data["Close"].ewm(span=50, adjust=False).mean()

    prev_close = data["Close"].shift(1)
    tr = pd.concat(
        [
            data["High"] - data["Low"],
            (data["High"] - prev_close).abs(),
            (data["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    data["ATR14"] = tr.rolling(14).mean()

    data["Retorno"] = data["Close"].pct_change()
    data["Volatilidade20"] = data["Retorno"].rolling(20).std() * np.sqrt(252)
    data["VolumeMA20"] = data["Volume"].rolling(20).mean()
    data["RVOL"] = data["Volume"] / data["VolumeMA20"]
    data["RSI14"] = calculate_rsi(data["Close"], 14)

    data["Max20"] = data["High"].rolling(20).max()
    data["Min20"] = data["Low"].rolling(20).min()
    data["Min10"] = data["Low"].rolling(10).min()
    data["ValorFinanceiroMedio"] = data["Close"] * data["VolumeMA20"]

    body = (data["Close"] - data["Open"]).abs()
    range_ = (data["High"] - data["Low"]).replace(0, np.nan)
    upper_wick = data["High"] - data[["Open", "Close"]].max(axis=1)
    data["BODY_FRAC"] = body / range_
    data["CLOSE_TO_HIGH"] = (data["High"] - data["Close"]) / range_
    data["BULL_CANDLE"] = (data["Close"] > data["Open"]).astype(int)

    if interval != "1d":
        data["VWAP"] = add_session_vwap(data)
        data["RVOL_CLOCK"] = add_intraday_rvol_clock(data)
    else:
        data["VWAP"] = np.nan
        data["RVOL_CLOCK"] = np.nan

    data["ForcaRelativa"] = np.nan
    if benchmark_close is not None and not benchmark_close.empty:
        aligned_bench = benchmark_close.reindex(data.index).ffill().bfill()
        lookback = 20 if len(data) >= 60 else 10 if len(data) >= 30 else 5
        asset_ret = data["Close"].pct_change(lookback)
        bench_ret = aligned_bench.pct_change(lookback)
        data["ForcaRelativa"] = asset_ret - bench_ret

    required = [
        "EMA9",
        "EMA21",
        "EMA50",
        "ATR14",
        "Volatilidade20",
        "VolumeMA20",
        "RVOL",
        "RSI14",
        "Max20",
        "Min20",
        "Min10",
        "ValorFinanceiroMedio",
        "BODY_FRAC",
        "CLOSE_TO_HIGH",
    ]

    if interval != "1d":
        required.append("VWAP")

    data = data.dropna(subset=required).copy()
    return data


# =========================================================
# REGIME DE MERCADO
# =========================================================
def build_regime_table(benchmark_df: pd.DataFrame) -> pd.DataFrame:
    if benchmark_df.empty:
        return pd.DataFrame()

    bench = benchmark_df.copy()
    bench["EMA21"] = bench["Close"].ewm(span=21, adjust=False).mean()
    bench["EMA50"] = bench["Close"].ewm(span=50, adjust=False).mean()
    bench["RSI14"] = calculate_rsi(bench["Close"], 14)
    bench["RET20"] = bench["Close"] .pct_change(20)

    regimes = []
    reasons = []
    for _, row in bench.iterrows():
        close = row["Close"]
        ema21 = row["EMA21"]
        ema50 = row["EMA50"]
        rsi = row["RSI14"]
        ret20 = row["RET20"]

        if pd.isna(ema21) or pd.isna(ema50) or pd.isna(rsi) or pd.isna(ret20):
            regimes.append("LATERAL")
            reasons.append("Índice com poucos dados para confirmar tendência.")
        elif close > ema21 > ema50 and rsi >= 52 and ret20 > 0:
            regimes.append("COMPRADOR")
            reasons.append("Índice acima da EMA21/EMA50, RSI positivo e retorno recente positivo.")
        elif close < ema21 < ema50 and rsi < 48 and ret20 < 0:
            regimes.append("DEFENSIVO")
            reasons.append("Índice abaixo da EMA21/EMA50, RSI fraco e retorno recente negativo.")
        else:
            regimes.append("LATERAL")
            reasons.append("Mercado sem alinhamento claro de tendência.")

    bench["REGIME"] = regimes
    bench["REASON"] = reasons
    return bench


def get_regime_info_at(regime_table: pd.DataFrame, timestamp) -> Dict[str, float]:
    if regime_table.empty:
        return {
            "regime": "LATERAL",
            "reason": "Índice indisponível.",
            "close": np.nan,
            "ema21": np.nan,
            "ema50": np.nan,
            "rsi": np.nan,
            "ret20": np.nan,
        }

    table = regime_table.copy()
    table.index = pd.to_datetime(table.index, errors="coerce")
    table = table[~table.index.isna()].sort_index()

    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    ts = pd.Timestamp(ts.to_datetime64())

    subset = table.loc[table.index <= ts]
    row = table.iloc[0] if subset.empty else subset.iloc[-1]

    return {
        "regime": row.get("REGIME", "LATERAL"),
        "reason": row.get("REASON", "Mercado sem alinhamento claro de tendência."),
        "close": float(row["Close"]) if pd.notna(row.get("Close")) else np.nan,
        "ema21": float(row["EMA21"]) if pd.notna(row.get("EMA21")) else np.nan,
        "ema50": float(row["EMA50"]) if pd.notna(row.get("EMA50")) else np.nan,
        "rsi": float(row["RSI14"]) if pd.notna(row.get("RSI14")) else np.nan,
        "ret20": float(row["RET20"]) if pd.notna(row.get("RET20")) else np.nan,
    }


# =========================================================
# RISCO
# =========================================================
def compute_trade_levels(trigger_data: pd.DataFrame) -> Dict[str, float]:
    last = trigger_data.iloc[-1]
    entry = float(last["Close"])
    atr = float(last["ATR14"])
    recent_support = float(last["Min10"])
    breakout_ref = float(last["Max20"])

    stop = max(recent_support, entry - 1.5 * atr)
    if stop >= entry:
        stop = entry * 0.9875

    risk = max(entry - stop, entry * 0.006)
    target_base = entry + (risk * 2.2)
    target_resistance = max(breakout_ref, entry + 1.2 * atr)
    target = max(target_base, target_resistance)
    partial = entry + risk * 1.0
    rr = (target - entry) / risk if risk > 0 else np.nan

    return {
        "entry": entry,
        "stop": stop,
        "target": target,
        "partial": partial,
        "risk": risk,
        "rr": rr,
    }


def calculate_position_size(
    capital: float,
    risk_per_trade_pct: float,
    entry: float,
    stop: float,
    max_position_pct: float,
) -> Dict[str, float]:
    capital = max(capital, 0)
    risk_amount = capital * risk_per_trade_pct
    risk_per_share = max(entry - stop, 0)

    if capital <= 0 or entry <= 0 or risk_per_share <= 0:
        return {
            "risk_amount": risk_amount,
            "risk_per_share": risk_per_share,
            "qty": 0,
            "position_value": 0,
            "position_pct": 0,
        }

    qty_risco = math.floor(risk_amount / risk_per_share)
    qty_caixa = math.floor(capital / entry)
    qty_exposicao = math.floor((capital * max_position_pct) / entry)
    qty_final = max(min(qty_risco, qty_caixa, qty_exposicao), 0)

    position_value = qty_final * entry
    position_pct = position_value / capital if capital > 0 else 0

    return {
        "risk_amount": risk_amount,
        "risk_per_share": risk_per_share,
        "qty": qty_final,
        "position_value": position_value,
        "position_pct": position_pct,
    }


# =========================================================
# LÓGICA DO SETUP
# =========================================================
def get_structure_settings(trigger_interval: str) -> Tuple[str, str]:
    if trigger_interval == "1d":
        return "3mo", "1d"
    return "1mo", "60m"


def effective_rvol(trigger_row: pd.Series, interval: str) -> float:
    if interval == "1d":
        return float(trigger_row["RVOL"])
    clock = trigger_row.get("RVOL_CLOCK", np.nan)
    if pd.notna(clock):
        return float(clock)
    return float(trigger_row["RVOL"])


def evaluate_asset_from_indicator_frames(
    ticker: str,
    daily: pd.DataFrame,
    structure: pd.DataFrame,
    trigger: pd.DataFrame,
    benchmark_trigger: Optional[pd.DataFrame],
    regime_info: Dict[str, float],
    capital: float,
    risk_per_trade_pct: float,
    min_rr: float,
    max_position_pct: float,
    min_upside_pct: float,
    min_liquidity_million: float,
    entry_min_rvol: float,
    entry_max_rsi: float,
    entry_min_score: float,
    require_breakout: bool,
    require_strong_candle: bool,
    allowed_windows: List[Tuple[time, time]],
    trigger_interval: str,
    require_index_intraday: bool,
) -> Dict[str, float]:
    d = daily.iloc[-1]
    s = structure.iloc[-1]
    t = trigger.iloc[-1]

    levels = compute_trade_levels(trigger)
    position = calculate_position_size(
        capital=capital,
        risk_per_trade_pct=risk_per_trade_pct,
        entry=levels["entry"],
        stop=levels["stop"],
        max_position_pct=max_position_pct,
    )

    score = 0
    strengths: List[str] = []
    alerts: List[str] = []
    hard_blocks: List[str] = []

    close_daily = float(d["Close"])
    close_structure = float(s["Close"])
    close_trigger = float(t["Close"])

    daily_ema21 = float(d["EMA21"])
    daily_ema50 = float(d["EMA50"])
    structure_ema9 = float(s["EMA9"])
    structure_ema21 = float(s["EMA21"])
    trigger_ema9 = float(t["EMA9"])
    trigger_ema21 = float(t["EMA21"])
    trigger_vwap = float(t["VWAP"]) if pd.notna(t["VWAP"]) else np.nan
    trigger_rsi = float(t["RSI14"])
    trigger_rvol = effective_rvol(t, trigger_interval)
    trigger_vol = float(t["Volatilidade20"])
    liquidity_value = float(t["ValorFinanceiroMedio"])
    rel_strength = float(d["ForcaRelativa"]) if pd.notna(d["ForcaRelativa"]) else np.nan
    body_frac = float(t["BODY_FRAC"])
    close_to_high = float(t["CLOSE_TO_HIGH"])
    bull_candle = bool(int(t["BULL_CANDLE"]))

    prev_trigger_high = float(trigger.iloc[-2]["High"]) if len(trigger) >= 2 else close_trigger
    breakout_ok = close_trigger > prev_trigger_high
    strong_candle = bull_candle and body_frac >= 0.55 and close_to_high <= 0.25
    time_ok = True if trigger_interval == "1d" else within_any_window(trigger.index[-1], allowed_windows)

    index_intraday_ok = True
    if require_index_intraday and trigger_interval != "1d" and benchmark_trigger is not None and not benchmark_trigger.empty:
        b = benchmark_trigger.iloc[-1]
        b_close = float(b["Close"])
        b_ema21 = float(b["EMA21"])
        b_vwap = float(b["VWAP"]) if pd.notna(b["VWAP"]) else np.nan
        index_intraday_ok = b_close > b_ema21 and (pd.isna(b_vwap) or b_close > b_vwap)

    stop_pct = (levels["entry"] - levels["stop"]) / levels["entry"] if levels["entry"] > 0 else np.nan
    upside_pct = (levels["target"] - levels["entry"]) / levels["entry"] if levels["entry"] > 0 else np.nan

    if regime_info["regime"] == "COMPRADOR":
        score += 12
        strengths.append("Mercado favorável")
    elif regime_info["regime"] == "LATERAL":
        score -= 5
        alerts.append("Mercado lateral")
    else:
        score -= 20
        alerts.append("Mercado defensivo")

    if require_index_intraday and trigger_interval != "1d":
        if index_intraday_ok:
            score += 8
            strengths.append("Índice intraday alinhado")
        else:
            score -= 12
            alerts.append("Índice intraday fraco")

    if close_daily > daily_ema21 > daily_ema50:
        score += 24
        strengths.append("Tendência diária alinhada")
    elif close_daily > daily_ema21:
        score += 10
    else:
        score -= 20
        alerts.append("Diário abaixo da EMA21")

    if close_structure > structure_ema21 and structure_ema9 > structure_ema21:
        score += 18
        strengths.append("Estrutura intraday positiva")
    elif close_structure > structure_ema21:
        score += 8
    else:
        score -= 12
        alerts.append("Estrutura fraca")

    if close_trigger > trigger_ema21:
        score += 10
        strengths.append("Gatilho acima da EMA21")
    else:
        score -= 10
        alerts.append("Gatilho abaixo da EMA21")

    if pd.notna(trigger_vwap):
        if close_trigger > trigger_vwap:
            score += 10
            strengths.append("Acima da VWAP")
        else:
            score -= 12
            alerts.append("Abaixo da VWAP")

    if trigger_ema9 > trigger_ema21:
        score += 8
        strengths.append("EMA9 acima da EMA21")
    else:
        score -= 4

    if breakout_ok:
        score += 8
        strengths.append("Rompimento confirmado")
    else:
        score -= 4
        alerts.append("Sem rompimento de curto prazo")

    if strong_candle:
        score += 8
        strengths.append("Candle de força")
    else:
        score -= 5
        alerts.append("Candle sem força")

    if 54 <= trigger_rsi <= 68:
        score += 10
        strengths.append("Momentum saudável")
    elif 68 < trigger_rsi <= entry_max_rsi:
        score += 2
        alerts.append("Levemente esticada")
    elif trigger_rsi < 45:
        score -= 8
        alerts.append("RSI fraco")
    elif trigger_rsi > entry_max_rsi:
        score -= 15
        alerts.append("RSI esticado")

    if trigger_rvol >= 1.5:
        score += 12
        strengths.append("Volume forte")
    elif trigger_rvol >= 1.1:
        score += 6
    elif trigger_rvol < entry_min_rvol:
        score -= 18
        alerts.append("RVOL abaixo do mínimo de entrada")
    else:
        score -= 8
        alerts.append("RVOL moderado")

    if pd.notna(rel_strength):
        if rel_strength > 0.03:
            score += 10
            strengths.append("Mais forte que o índice")
        elif rel_strength < -0.02:
            score -= 10
            alerts.append("Mais fraca que o índice")

    if 0.10 <= trigger_vol <= 0.45:
        score += 6
    elif trigger_vol > 0.65:
        score -= 8
        alerts.append("Volatilidade alta")

    if levels["rr"] >= min_rr:
        score += 10
        strengths.append("Risco/retorno favorável")
    else:
        score -= 15
        alerts.append("R:R abaixo do mínimo")

    if pd.notna(upside_pct) and upside_pct >= min_upside_pct:
        score += 8
        strengths.append("Potencial mínimo atendido")
    else:
        score -= 15
        alerts.append("Potencial baixo")

    if pd.notna(stop_pct) and stop_pct <= 0.08:
        score += 6
        strengths.append("Stop controlado")
    elif pd.notna(stop_pct) and stop_pct > 0.15:
        score -= 8
        alerts.append("Stop muito longo")

    liquidity_cut = min_liquidity_million * 1_000_000
    if liquidity_value >= liquidity_cut:
        score += 8
        strengths.append("Boa liquidez")
    else:
        score -= 18
        alerts.append("Liquidez fraca")

    if not time_ok:
        hard_blocks.append("Fora da janela de horário")
    if require_index_intraday and trigger_interval != "1d" and not index_intraday_ok:
        hard_blocks.append("Índice intraday abaixo da EMA21/VWAP")
    if regime_info["regime"] == "DEFENSIVO":
        hard_blocks.append("Mercado defensivo")
    if close_daily <= daily_ema21:
        hard_blocks.append("Tendência diária não alinhada")
    if close_structure <= structure_ema21:
        hard_blocks.append("Estrutura 60m fraca")
    if close_trigger <= trigger_ema21:
        hard_blocks.append("Gatilho abaixo da EMA21")
    if pd.notna(trigger_vwap) and close_trigger <= trigger_vwap:
        hard_blocks.append("Preço abaixo da VWAP")
    if trigger_rvol < entry_min_rvol:
        hard_blocks.append(f"RVOL abaixo de {entry_min_rvol:.2f}")
    if trigger_rsi > entry_max_rsi:
        hard_blocks.append(f"RSI acima de {entry_max_rsi:.0f}")
    if require_breakout and not breakout_ok:
        hard_blocks.append("Sem rompimento confirmado")
    if require_strong_candle and not strong_candle:
        hard_blocks.append("Candle sem força")
    if levels["rr"] < min_rr:
        hard_blocks.append("Risco/retorno abaixo do mínimo")
    if pd.notna(upside_pct) and upside_pct < min_upside_pct:
        hard_blocks.append("Potencial abaixo do mínimo")
    if liquidity_value < liquidity_cut:
        hard_blocks.append("Liquidez abaixo do mínimo")
    if position["qty"] <= 0:
        hard_blocks.append("Quantidade inviável")
    if position["position_value"] > capital:
        hard_blocks.append("Excede o capital disponível")
    if position["position_pct"] > max_position_pct:
        hard_blocks.append("Excede exposição máxima")
    if score < entry_min_score:
        hard_blocks.append(f"Score abaixo de {entry_min_score:.0f}")

    score = max(min(score, 100), 0)

    if hard_blocks:
        decision = "NÃO OPERAR"
    elif score >= 90:
        decision = "OPERAR AGORA"
    elif score >= 70:
        decision = "OBSERVAR"
    else:
        decision = "NÃO OPERAR"

    if score >= 94 and not hard_blocks:
        confidence = "A+"
        confidence_pct = 0.72
    elif score >= 88 and not hard_blocks:
        confidence = "A"
        confidence_pct = 0.64
    elif score >= 82 and not hard_blocks:
        confidence = "B"
        confidence_pct = 0.57
    else:
        confidence = "C"
        confidence_pct = 0.48

    risk_multiplier = 1.0
    if confidence == "A+":
        risk_multiplier = 1.25
    elif confidence == "A":
        risk_multiplier = 1.10
    elif confidence == "B":
        risk_multiplier = 0.90
    else:
        risk_multiplier = 0.75

    adjusted_risk_amount = position["risk_amount"] * risk_multiplier

    return {
        "ticker": ticker.replace(".SA", ""),
        "score": score,
        "decision": decision,
        "regime": regime_info["regime"],
        "close": close_trigger,
        "entry": levels["entry"],
        "stop": levels["stop"],
        "target": levels["target"],
        "partial": levels["partial"],
        "risk_reward": levels["rr"],
        "stop_pct": stop_pct,
        "upside_pct": upside_pct,
        "rsi": trigger_rsi,
        "rvol": trigger_rvol,
        "volatility": trigger_vol,
        "relative_strength": rel_strength,
        "liquidity_value": liquidity_value,
        "qty": position["qty"],
        "position_value": position["position_value"],
        "position_pct": position["position_pct"],
        "risk_amount": adjusted_risk_amount,
        "base_risk_amount": position["risk_amount"],
        "risk_per_share": position["risk_per_share"],
        "strengths": " | ".join(strengths[:7]) if strengths else "Sem destaques",
        "alerts": " | ".join(alerts[:7]) if alerts else "Sem alertas",
        "hard_blocks": " | ".join(hard_blocks) if hard_blocks else "Sem bloqueios",
        "confidence": confidence,
        "confidence_pct": confidence_pct,
        "strong_candle": strong_candle,
        "breakout_ok": breakout_ok,
        "time_ok": time_ok,
    }


def build_summary(
    tickers: List[str],
    trigger_period: str,
    trigger_interval: str,
    trend_period: str,
    benchmark: str,
    capital: float,
    risk_per_trade_pct: float,
    max_position_pct: float,
    min_rr: float,
    min_upside_pct: float,
    min_liquidity_million: float,
    entry_min_rvol: float,
    entry_max_rsi: float,
    entry_min_score: float,
    require_breakout: bool,
    require_strong_candle: bool,
    allowed_windows: List[Tuple[time, time]],
    require_index_intraday: bool,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, float], pd.DataFrame]:
    rows = []
    chart_map: Dict[str, pd.DataFrame] = {}
    debug_rows = []

    benchmark_daily = load_data(benchmark, trend_period, "1d")
    benchmark_trigger_raw = load_data(benchmark, trigger_period, trigger_interval)
    regime_table = build_regime_table(benchmark_daily)
    regime_info = get_regime_info_at(regime_table, benchmark_daily.index.max() if not benchmark_daily.empty else pd.Timestamp.now())
    benchmark_daily_close = benchmark_daily["Close"] if not benchmark_daily.empty else pd.Series(dtype=float)
    benchmark_trigger_ind = add_indicators(benchmark_trigger_raw, trigger_interval, None) if not benchmark_trigger_raw.empty else pd.DataFrame()

    structure_period, structure_interval = get_structure_settings(trigger_interval)

    for ticker in tickers:
        try:
            daily_raw = load_data(ticker, trend_period, "1d")
            structure_raw = load_data(ticker, structure_period, structure_interval)
            trigger_raw = load_data(ticker, trigger_period, trigger_interval)

            if daily_raw.empty or structure_raw.empty or trigger_raw.empty:
                debug_rows.append({"ticker": ticker.replace(".SA", ""), "status": "erro", "motivo": "dados insuficientes"})
                continue

            daily_ind = add_indicators(daily_raw, "1d", benchmark_daily_close)
            structure_ind = add_indicators(structure_raw, structure_interval, None)
            trigger_ind = add_indicators(trigger_raw, trigger_interval, None)

            if daily_ind.empty or structure_ind.empty or trigger_ind.empty:
                debug_rows.append({"ticker": ticker.replace(".SA", ""), "status": "erro", "motivo": "indicadores insuficientes"})
                continue

            evaluated = evaluate_asset_from_indicator_frames(
                ticker=ticker,
                daily=daily_ind,
                structure=structure_ind,
                trigger=trigger_ind,
                benchmark_trigger=benchmark_trigger_ind,
                regime_info=regime_info,
                capital=capital,
                risk_per_trade_pct=risk_per_trade_pct,
                min_rr=min_rr,
                max_position_pct=max_position_pct,
                min_upside_pct=min_upside_pct,
                min_liquidity_million=min_liquidity_million,
                entry_min_rvol=entry_min_rvol,
                entry_max_rsi=entry_max_rsi,
                entry_min_score=entry_min_score,
                require_breakout=require_breakout,
                require_strong_candle=require_strong_candle,
                allowed_windows=allowed_windows,
                trigger_interval=trigger_interval,
                require_index_intraday=require_index_intraday,
            )

            rows.append(evaluated)
            chart_map[ticker] = trigger_ind
            debug_rows.append({"ticker": ticker.replace(".SA", ""), "status": "ok" if evaluated["decision"] != "NÃO OPERAR" else "bloqueado", "motivo": evaluated["hard_blocks"]})

        except Exception as e:
            debug_rows.append({"ticker": ticker.replace(".SA", ""), "status": "erro", "motivo": str(e)})

    debug_df = pd.DataFrame(debug_rows)
    if not rows:
        return pd.DataFrame(), chart_map, regime_info, debug_df

    summary = pd.DataFrame(rows)
    decision_order = {"OPERAR AGORA": 0, "OBSERVAR": 1, "NÃO OPERAR": 2}
    summary["decision_order"] = summary["decision"].map(decision_order).fillna(9)
    summary["ranking"] = (
        summary["score"]
        + summary["upside_pct"].fillna(0) * 100
        - summary["stop_pct"].fillna(0) * 45
        + summary["relative_strength"].fillna(0) * 110
        + summary["rvol"].fillna(0) * 7
        + summary["confidence_pct"].fillna(0) * 25
    )
    summary = summary.sort_values(by=["decision_order", "ranking", "score"], ascending=[True, False, False]).reset_index(drop=True)
    return summary, chart_map, regime_info, debug_df


# =========================================================
# BACKTEST
# =========================================================
def simulate_trade_path(
    trigger_ind: pd.DataFrame,
    start_idx: int,
    entry: float,
    stop: float,
    partial: float,
    target: float,
    qty: int,
    max_hold_bars: int,
    total_cost_pct: float,
    breakeven_r: float,
    partial_size_pct: float,
) -> Tuple[int, float, str, float]:
    end_idx = min(start_idx + max_hold_bars, len(trigger_ind) - 1)
    if end_idx <= start_idx:
        exit_price = float(trigger_ind.iloc[start_idx]["Close"])
        turnover = qty * (entry + exit_price)
        costs = turnover * total_cost_pct
        pnl = (exit_price - entry) * qty - costs
        return start_idx, exit_price, "sem barras futuras", pnl

    risk_per_share = max(entry - stop, 0)
    be_trigger = entry + risk_per_share * breakeven_r
    stop_level = stop
    breakeven_armed = False
    partial_taken = False
    remaining_qty = qty
    partial_qty = 0
    exit_turnover = 0.0
    realized_gross = 0.0

    exit_idx = end_idx
    final_exit_price = float(trigger_ind.iloc[end_idx]["Close"])
    reason = "tempo esgotado"

    for j in range(start_idx + 1, end_idx + 1):
        row = trigger_ind.iloc[j]
        high = float(row["High"])
        low = float(row["Low"])
        close = float(row["Close"])

        if low <= stop_level:
            exit_idx = j
            final_exit_price = stop_level
            reason = "breakeven" if breakeven_armed and stop_level >= entry else "stop"
            break

        if (not partial_taken) and high >= partial and qty > 1:
            partial_qty = max(1, int(qty * partial_size_pct))
            partial_qty = min(partial_qty, remaining_qty - 1) if remaining_qty > 1 else 0
            if partial_qty > 0:
                partial_taken = True
                remaining_qty -= partial_qty
                realized_gross += (partial - entry) * partial_qty
                exit_turnover += partial * partial_qty
                reason = "parcial"
                if breakeven_r <= 1.0:
                    breakeven_armed = True
                    stop_level = entry

        if (not breakeven_armed) and risk_per_share > 0 and high >= be_trigger:
            breakeven_armed = True
            stop_level = entry

        if high >= target:
            exit_idx = j
            final_exit_price = target
            reason = "alvo"
            break

        exit_idx = j
        final_exit_price = close

    exit_turnover += final_exit_price * remaining_qty
    turnover = qty * entry + exit_turnover
    costs = turnover * total_cost_pct
    gross = realized_gross + (final_exit_price - entry) * remaining_qty
    pnl = gross - costs
    return exit_idx, final_exit_price, reason, pnl


@st.cache_data(ttl=1800, show_spinner=False)
def run_backtest(
    tickers: List[str],
    backtest_period: str,
    trigger_interval: str,
    benchmark: str,
    capital: float,
    risk_per_trade_pct: float,
    max_position_pct: float,
    min_rr: float,
    min_upside_pct: float,
    min_liquidity_million: float,
    max_hold_bars: int,
    total_cost_pct: float,
    entry_min_rvol: float,
    entry_max_rsi: float,
    entry_min_score: float,
    require_breakout: bool,
    require_strong_candle: bool,
    allowed_windows: List[Tuple[time, time]],
    breakeven_r: float,
    partial_size_pct: float,
    require_index_intraday: bool,
    max_trades_per_day: int,
    max_consecutive_losses: int,
    one_trade_per_asset_day: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trades: List[Dict] = []
    debug_rows: List[Dict] = []

    fetch_period = get_safe_backtest_period(backtest_period, trigger_interval)
    daily_period = at_least_period(backtest_period, "6mo")

    benchmark_daily = load_data(benchmark, daily_period, "1d")
    benchmark_trigger_raw = load_data(benchmark, fetch_period if trigger_interval != "1d" else backtest_period, trigger_interval)
    benchmark_daily_close = benchmark_daily["Close"] if not benchmark_daily.empty else pd.Series(dtype=float)
    regime_table = build_regime_table(benchmark_daily)
    benchmark_trigger_ind = add_indicators(benchmark_trigger_raw, trigger_interval, None) if not benchmark_trigger_raw.empty else pd.DataFrame()

    structure_period, structure_interval = get_structure_settings(trigger_interval)
    if trigger_interval != "1d":
        structure_period = fetch_period if structure_interval == "60m" else daily_period

    for ticker in tickers:
        try:
            daily_raw = load_data(ticker, daily_period, "1d")
            structure_raw = load_data(ticker, structure_period, structure_interval)
            trigger_raw = load_data(ticker, fetch_period if trigger_interval != "1d" else backtest_period, trigger_interval)

            if daily_raw.empty or structure_raw.empty or trigger_raw.empty:
                debug_rows.append({"ticker": ticker.replace(".SA", ""), "status": "erro", "motivo": "dados insuficientes para backtest"})
                continue

            daily_ind = add_indicators(daily_raw, "1d", benchmark_daily_close)
            structure_ind = add_indicators(structure_raw, structure_interval, None)
            trigger_ind = add_indicators(trigger_raw, trigger_interval, None)

            if daily_ind.empty or structure_ind.empty or trigger_ind.empty:
                debug_rows.append({"ticker": ticker.replace(".SA", ""), "status": "erro", "motivo": "indicadores insuficientes para backtest"})
                continue

            i = 20
            trades_count = 0
            trade_count_by_day: Dict = {}
            loss_streak_by_day: Dict = {}
            traded_days: set = set()
            while i < len(trigger_ind) - 1:
                ts = pd.Timestamp(trigger_ind.index[i])
                day_key = ts.date()
                if one_trade_per_asset_day and day_key in traded_days:
                    i += 1
                    continue
                if trade_count_by_day.get(day_key, 0) >= max_trades_per_day:
                    i += 1
                    continue
                if loss_streak_by_day.get(day_key, 0) >= max_consecutive_losses:
                    i += 1
                    continue

                daily_slice = daily_ind.loc[daily_ind.index <= ts]
                structure_slice = structure_ind.loc[structure_ind.index <= ts]
                trigger_slice = trigger_ind.iloc[: i + 1]
                benchmark_trigger_slice = benchmark_trigger_ind.loc[benchmark_trigger_ind.index <= ts] if not benchmark_trigger_ind.empty else pd.DataFrame()
                regime_info = get_regime_info_at(regime_table, ts)

                if daily_slice.empty or structure_slice.empty or len(trigger_slice) < 20:
                    i += 1
                    continue

                evaluated = evaluate_asset_from_indicator_frames(
                    ticker=ticker,
                    daily=daily_slice,
                    structure=structure_slice,
                    trigger=trigger_slice,
                    benchmark_trigger=benchmark_trigger_slice,
                    regime_info=regime_info,
                    capital=capital,
                    risk_per_trade_pct=risk_per_trade_pct,
                    min_rr=min_rr,
                    max_position_pct=max_position_pct,
                    min_upside_pct=min_upside_pct,
                    min_liquidity_million=min_liquidity_million,
                    entry_min_rvol=entry_min_rvol,
                    entry_max_rsi=entry_max_rsi,
                    entry_min_score=entry_min_score,
                    require_breakout=require_breakout,
                    require_strong_candle=require_strong_candle,
                    allowed_windows=allowed_windows,
                    trigger_interval=trigger_interval,
                    require_index_intraday=require_index_intraday,
                )

                if evaluated["decision"] != "OPERAR AGORA":
                    i += 1
                    continue

                qty = int(evaluated["qty"])
                if qty <= 0:
                    i += 1
                    continue

                entry = float(evaluated["entry"])
                stop = float(evaluated["stop"])
                partial = float(evaluated["partial"])
                target = float(evaluated["target"])
                risk_amount = float(evaluated["risk_amount"])

                exit_idx, exit_price, exit_reason, pnl = simulate_trade_path(
                    trigger_ind=trigger_ind,
                    start_idx=i,
                    entry=entry,
                    stop=stop,
                    partial=partial,
                    target=target,
                    qty=qty,
                    max_hold_bars=max_hold_bars,
                    total_cost_pct=total_cost_pct,
                    breakeven_r=breakeven_r,
                    partial_size_pct=partial_size_pct,
                )

                r_multiple = pnl / risk_amount if risk_amount > 0 else np.nan
                trades.append(
                    {
                        "ticker": ticker.replace(".SA", ""),
                        "entry_time": ts,
                        "exit_time": trigger_ind.index[exit_idx],
                        "entry": entry,
                        "exit": exit_price,
                        "stop": stop,
                        "partial": partial,
                        "target": target,
                        "qty": qty,
                        "pnl": pnl,
                        "ret_capital": pnl / capital if capital > 0 else np.nan,
                        "r_multiple": r_multiple,
                        "reason": exit_reason,
                        "score": evaluated["score"],
                        "regime": regime_info["regime"],
                    }
                )
                trades_count += 1
                trade_count_by_day[day_key] = trade_count_by_day.get(day_key, 0) + 1
                traded_days.add(day_key)
                if pnl <= 0:
                    loss_streak_by_day[day_key] = loss_streak_by_day.get(day_key, 0) + 1
                else:
                    loss_streak_by_day[day_key] = 0
                i = exit_idx + 1

            debug_rows.append({"ticker": ticker.replace(".SA", ""), "status": "ok" if trades_count > 0 else "sem trades", "motivo": f"{trades_count} trades no backtest" if trades_count > 0 else "nenhuma barra passou pelos filtros"})

        except Exception as e:
            debug_rows.append({"ticker": ticker.replace(".SA", ""), "status": "erro", "motivo": str(e)})

    debug_df = pd.DataFrame(debug_rows)
    if not trades:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), debug_df

    trades_df = pd.DataFrame(trades).sort_values("exit_time").reset_index(drop=True)
    trades_df["equity"] = capital + trades_df["pnl"].cumsum()
    trades_df["equity_peak"] = trades_df["equity"].cummax()
    trades_df["drawdown"] = (trades_df["equity"] - trades_df["equity_peak"]) / trades_df["equity_peak"].replace(0, np.nan)

    total_trades = len(trades_df)
    wins = int((trades_df["pnl"] > 0).sum())
    losses = int((trades_df["pnl"] <= 0).sum())
    gross_profit = trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum()
    gross_loss = abs(trades_df.loc[trades_df["pnl"] <= 0, "pnl"].sum())
    avg_win = trades_df.loc[trades_df["pnl"] > 0, "pnl"].mean() if wins > 0 else 0.0
    avg_loss = abs(trades_df.loc[trades_df["pnl"] <= 0, "pnl"].mean()) if losses > 0 else 0.0
    payoff = (avg_win / avg_loss) if avg_loss > 0 else np.nan
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.nan
    net_profit = trades_df["pnl"].sum()
    win_rate = wins / total_trades if total_trades > 0 else np.nan
    expectancy_r = trades_df["r_multiple"].mean() if total_trades > 0 else np.nan
    max_drawdown = trades_df["drawdown"].min() if total_trades > 0 else np.nan
    final_equity = capital + net_profit

    summary_df = pd.DataFrame([
        {
            "trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "net_profit": net_profit,
            "final_equity": final_equity,
            "profit_factor": profit_factor,
            "payoff": payoff,
            "expectancy_r": expectancy_r,
            "max_drawdown": max_drawdown,
        }
    ])

    by_ticker = (
        trades_df.groupby("ticker")
        .agg(
            trades=("ticker", "size"),
            net_profit=("pnl", "sum"),
            win_rate=("pnl", lambda s: (s > 0).mean()),
            avg_r=("r_multiple", "mean"),
            avg_score=("score", "mean"),
            gross_profit=("pnl", lambda s: s[s > 0].sum()),
            gross_loss=("pnl", lambda s: abs(s[s <= 0].sum())),
            winners=("pnl", lambda s: int((s > 0).sum())),
            losers=("pnl", lambda s: int((s <= 0).sum())),
        )
        .reset_index()
    )
    by_ticker["profit_factor"] = by_ticker.apply(
        lambda row: row["gross_profit"] / row["gross_loss"] if row["gross_loss"] > 0 else np.nan,
        axis=1,
    )
    by_ticker = by_ticker.sort_values(["net_profit", "profit_factor", "win_rate"], ascending=False)

    return summary_df, trades_df, by_ticker, debug_df


def select_tickers_from_backtest(
    bt_by_ticker: pd.DataFrame,
    min_trades_asset: int,
    min_win_rate_asset: float,
    min_avg_r_asset: float,
    require_positive_net_profit_asset: bool,
    require_positive_profit_factor_asset: bool,
) -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
    if bt_by_ticker.empty:
        return [], pd.DataFrame(), pd.DataFrame()

    decisions = []
    for _, row in bt_by_ticker.iterrows():
        reasons = []
        if int(row["trades"]) < min_trades_asset:
            reasons.append(f"menos de {min_trades_asset} trades")
        if float(row["win_rate"]) < min_win_rate_asset:
            reasons.append(f"win rate abaixo de {min_win_rate_asset * 100:.0f}%")
        if float(row["avg_r"]) < min_avg_r_asset:
            reasons.append(f"expectância média abaixo de {min_avg_r_asset:.2f}R")
        if require_positive_net_profit_asset and float(row["net_profit"]) <= 0:
            reasons.append("lucro líquido não positivo")
        if require_positive_profit_factor_asset and pd.notna(row["profit_factor"]) and float(row["profit_factor"]) <= 1:
            reasons.append("profit factor abaixo de 1")

        decisions.append(
            {
                "ticker": row["ticker"],
                "aprovado": len(reasons) == 0,
                "motivos": " | ".join(reasons) if reasons else "Aprovado no filtro histórico",
            }
        )

    decision_df = pd.DataFrame(decisions)
    merged = bt_by_ticker.merge(decision_df, on="ticker", how="left")
    approved_df = merged[merged["aprovado"]].copy().sort_values(["net_profit", "profit_factor"], ascending=False)
    blocked_df = merged[~merged["aprovado"]].copy().sort_values(["net_profit", "win_rate"], ascending=[False, False])
    approved_labels = approved_df["ticker"].tolist()
    return approved_labels, approved_df, blocked_df


# =========================================================
# GRÁFICOS
# =========================================================
def build_price_chart(data: pd.DataFrame, title: str, entry: float, stop: float, partial: float, target: float) -> go.Figure:
    plot_df = data.tail(120).copy()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.75, 0.25])

    fig.add_trace(
        go.Candlestick(
            x=plot_df.index,
            open=plot_df["Open"],
            high=plot_df["High"],
            low=plot_df["Low"],
            close=plot_df["Close"],
            name="Preço",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["EMA9"], name="EMA9"), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["EMA21"], name="EMA21"), row=1, col=1)
    if "VWAP" in plot_df.columns and plot_df["VWAP"].notna().any():
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["VWAP"], name="VWAP"), row=1, col=1)
    fig.add_trace(go.Bar(x=plot_df.index, y=plot_df["Volume"], name="Volume"), row=2, col=1)

    fig.add_hrect(y0=stop, y1=entry, fillcolor="rgba(255,0,0,0.12)", line_width=0, row=1, col=1)
    fig.add_hrect(y0=entry, y1=target, fillcolor="rgba(0,255,0,0.12)", line_width=0, row=1, col=1)

    for level, name in [(entry, "Entrada"), (stop, "Stop"), (partial, "Parcial"), (target, "Alvo")]:
        fig.add_hline(y=level, annotation_text=f"{name}: {level:.2f}", row=1, col=1)

    fig.update_layout(
        title=title,
        height=720,
        xaxis_rangeslider_visible=False,
        legend_title="Indicadores",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_yaxes(title_text="Preço", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


def build_equity_chart(trades_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not trades_df.empty:
        fig.add_trace(go.Scatter(x=trades_df["exit_time"], y=trades_df["equity"], mode="lines+markers", name="Capital"))
    fig.update_layout(title="Curva de capital do backtest", xaxis_title="Saída do trade", yaxis_title="Capital", height=360)
    return fig


# =========================================================
# INTERFACE
# =========================================================
st.title("📈 Robô Pessoal de Análise de Ações — V8")
st.caption("Perfil agressivo-controlado: mais ganho potencial, mas só com ativos aprovados no histórico e setups de alta confiança.")

with st.sidebar:
    st.header("Configurações")
    tickers_text = st.text_area("Ativos monitorados (separados por vírgula)", value=DEFAULT_TICKERS)
    trigger_interval_label = st.selectbox("Intervalo de gatilho", list(TRIGGER_INTERVAL_OPTIONS.keys()), index=1)
    trigger_period_label = st.selectbox("Período do gatilho", list(TRIGGER_PERIOD_OPTIONS.keys()), index=0)
    trend_period_label = st.selectbox("Período diário para tendência maior", list(TREND_PERIOD_OPTIONS.keys()), index=0)
    benchmark = st.text_input("Referência", value="^BVSP").strip().upper() or "^BVSP"

    st.divider()
    st.subheader("Gestão de risco")
    capital = st.number_input("Capital total (R$)", min_value=100.0, value=10000.0, step=100.0)
    risk_per_trade_pct = st.slider("Risco por operação (%)", 0.25, 3.0, 1.0, 0.25) / 100
    daily_loss_limit_pct = st.slider("Perda máxima no dia (%)", 1.0, 5.0, 2.0, 0.5) / 100
    max_position_pct = st.slider("Exposição máxima por ativo (%)", 5.0, 50.0, 30.0, 5.0) / 100

    st.divider()
    st.subheader("Filtros duros")
    min_rr = st.slider("Risco/retorno mínimo", 1.5, 4.0, 2.0, 0.1)
    min_upside_pct = st.slider("Potencial mínimo (%)", 0.5, 5.0, 1.2, 0.1) / 100
    min_liquidity_million = st.slider("Liquidez mínima média (R$ mi)", 1.0, 50.0, 10.0, 1.0)
    top_n = st.slider("Quantidade de sugestões", 1, 10, 3)

    st.divider()
    st.subheader("Ajuste fino do setup")
    entry_min_rvol = st.slider("RVOL mínimo para entrada", 0.60, 2.00, 1.00, 0.05)
    entry_max_rsi = st.slider("RSI máximo para entrada", 55, 80, 68, 1)
    entry_min_score = st.slider("Score mínimo para entrada", 70, 100, 85, 1)
    require_breakout = st.checkbox("Exigir rompimento confirmado", value=True)
    require_strong_candle = st.checkbox("Exigir candle de força", value=True)
    require_index_intraday = st.checkbox("Exigir índice intraday alinhado", value=True)

    st.divider()
    st.subheader("Filtro histórico dos ativos")
    use_performance_filter = st.checkbox("Usar só ativos aprovados no backtest", value=True)
    min_trades_asset = st.slider("Mínimo de trades por ativo no backtest", 3, 20, 5)
    min_win_rate_asset = st.slider("Win rate mínimo por ativo (%)", 30, 80, 45) / 100
    min_avg_r_asset = st.slider("Expectância mínima por ativo (R)", -0.20, 0.30, 0.00, 0.01)
    require_positive_net_profit_asset = st.checkbox("Exigir lucro líquido positivo por ativo", value=True)
    require_positive_profit_factor_asset = st.checkbox("Exigir profit factor acima de 1 por ativo", value=True)
    profile_mode = st.selectbox("Perfil operacional", ["Agressivo-controlado", "Conservador"], index=0)

    st.divider()
    st.subheader("Filtro de horário")
    use_time_filter = st.checkbox("Usar janela de horário", value=True)
    morning_window = st.select_slider("Janela da manhã", options=TIME_CHOICES, value=("10:05", "11:30"))
    use_afternoon = st.checkbox("Usar janela da tarde", value=True)
    afternoon_window = st.select_slider("Janela da tarde", options=TIME_CHOICES, value=("14:00", "16:30"))

    st.divider()
    st.subheader("Backtest")
    run_backtest_toggle = st.checkbox("Mostrar backtest", value=True)
    backtest_period_label = st.selectbox("Janela faz backtest", list(BACKTEST_PERIOD_OPTIONS.keys()), index=1)
    max_hold_bars = st.slider("Máximo de velas por comércio", 3, 60, 20)
    total_cost_pct = st.slider("Custo total por transação (%)", 0.0, 1.0, 0.10, 0.01) / 100
    breakeven_r = st.slider("Mover stop para breakeven em (R)", 0.5, 2.0, 1.0, 0.1)
    partial_size_pct = st.slider("Parcial no 1R (% da posição)", 25, 75, 50, 5) / 100
    max_trades_per_day = st.slider("Máximo de trades por dia/ativo", 1, 5, 2)
    max_consecutive_losses = st.slider("Máximo de perdas seguidas por dia/ativo", 1, 3, 2)
    one_trade_per_asset_day = st.checkbox("Só 1 trade por ativo por dia", value=False)

if profile_mode == "Agressivo-controlado":
    entry_min_rvol = min(entry_min_rvol, 0.95)
    entry_min_score = min(entry_min_score, 82)
    max_trades_per_day = max(max_trades_per_day, 3)
    max_consecutive_losses = min(max_consecutive_losses, 2)
    partial_size_pct = 0.40 if partial_size_pct > 0.40 else partial_size_pct
    one_trade_per_asset_day = False

    st.divider()
    auto_refresh = st.checkbox("Atualização automática", value=False)
    refresh_seconds = st.slider("Atualizar a cada (segundos)", 30, 600, 120, 30)
    force_update = st.button("Atualizar agora", use_container_width=True)

    st.divider()
    st.subheader("Alertas manuais")
    alert_above_text = st.text_input("Alertar se subir até", value="")
    alert_below_text = st.text_input("Alertar se cair até", value="")

trigger_period = TRIGGER_PERIOD_OPTIONS[trigger_period_label]
trigger_interval = TRIGGER_INTERVAL_OPTIONS[trigger_interval_label]
trend_period = TREND_PERIOD_OPTIONS[trend_period_label]
backtest_period = BACKTEST_PERIOD_OPTIONS[backtest_period_label]
allowed_windows = build_allowed_windows(use_time_filter, morning_window, afternoon_window, use_afternoon)

tickers = [normalize_ticker(t) for t in tickers_text.split(",")]
tickers = [t for t in tickers if t]

if force_update:
    st.cache_data.clear()

if auto_refresh:
    components.html(
        f"""
        <script>
            setTimeout(function() {{
                window.parent.location.reload();
            }}, {refresh_seconds * 1000});
        </script>
        """,
        height=0,
    )

bt_summary = pd.DataFrame()
bt_trades = pd.DataFrame()
bt_by_ticker = pd.DataFrame()
bt_debug = pd.DataFrame()
approved_bt_df = pd.DataFrame()
blocked_bt_df = pd.DataFrame()
filtered_tickers = tickers.copy()

if use_performance_filter or run_backtest_toggle:
    bt_summary, bt_trades, bt_by_ticker, bt_debug = run_backtest(
        tickers=tickers,
        backtest_period=backtest_period,
        trigger_interval=trigger_interval,
        benchmark=benchmark,
        capital=capital,
        risk_per_trade_pct=risk_per_trade_pct,
        max_position_pct=max_position_pct,
        min_rr=min_rr,
        min_upside_pct=min_upside_pct,
        min_liquidity_million=min_liquidity_million,
        max_hold_bars=max_hold_bars,
        total_cost_pct=total_cost_pct,
        entry_min_rvol=entry_min_rvol,
        entry_max_rsi=entry_max_rsi,
        entry_min_score=entry_min_score,
        require_breakout=require_breakout,
        require_strong_candle=require_strong_candle,
        allowed_windows=allowed_windows,
        breakeven_r=breakeven_r,
        partial_size_pct=partial_size_pct,
        require_index_intraday=require_index_intraday,
        max_trades_per_day=max_trades_per_day,
        max_consecutive_losses=max_consecutive_losses,
        one_trade_per_asset_day=one_trade_per_asset_day,
    )

    if use_performance_filter and not bt_by_ticker.empty:
        label_to_full = {t.replace(".SA", ""): t for t in tickers}
        approved_labels, approved_bt_df, blocked_bt_df = select_tickers_from_backtest(
            bt_by_ticker=bt_by_ticker,
            min_trades_asset=min_trades_asset,
            min_win_rate_asset=min_win_rate_asset,
            min_avg_r_asset=min_avg_r_asset,
            require_positive_net_profit_asset=require_positive_net_profit_asset,
            require_positive_profit_factor_asset=require_positive_profit_factor_asset,
        )
        filtered_tickers = [label_to_full[label] for label in approved_labels if label in label_to_full]

summary, chart_map, regime_info, debug_df = build_summary(
    tickers=filtered_tickers,
    trigger_period=trigger_period,
    trigger_interval=trigger_interval,
    trend_period=trend_period,
    benchmark=benchmark,
    capital=capital,
    risk_per_trade_pct=risk_per_trade_pct,
    max_position_pct=max_position_pct,
    min_rr=min_rr,
    min_upside_pct=min_upside_pct,
    min_liquidity_million=min_liquidity_million,
    entry_min_rvol=entry_min_rvol,
    entry_max_rsi=entry_max_rsi,
    entry_min_score=entry_min_score,
    require_breakout=require_breakout,
    require_strong_candle=require_strong_candle,
    allowed_windows=allowed_windows,
    require_index_intraday=require_index_intraday,
)

if use_performance_filter and len(filtered_tickers) == 0:
    st.error("Nenhum ativo passou no filtro histórico do backtest. Isso é melhor do que forçar operação ruim.")
    if not blocked_bt_df.empty:
        view_blocked = blocked_bt_df[["ticker", "trades", "win_rate", "avg_r", "net_profit", "profit_factor", "motivos"]].copy()
        view_blocked.columns = ["Ticker", "Trades", "Win rate", "Média em R", "Lucro líquido", "Profit factor", "Motivos"]
        view_blocked["Win rate"] = view_blocked["Win rate"].map(format_pct)
        view_blocked["Média em R"] = view_blocked["Média em R"].map(lambda x: f"{x:.2f}")
        view_blocked["Lucro líquido"] = view_blocked["Lucro líquido"].map(format_money)
        view_blocked["Profit factor"] = view_blocked["Profit factor"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
        st.dataframe(view_blocked, use_container_width=True, hide_index=True)
    st.stop()

if summary.empty:
    st.error("O app abriu, mas nenhum ativo retornou dados válidos para montar a análise.")
    if not debug_df.empty:
        st.dataframe(debug_df, use_container_width=True, hide_index=True)
    st.stop()

if use_performance_filter and not bt_by_ticker.empty:
    st.subheader("Filtro histórico dos ativos")
    f1, f2, f3 = st.columns(3)
    f1.metric("Ativos aprovados", len(filtered_tickers))
    f2.metric("Ativos bloqueados", len(blocked_bt_df))
    f3.metric("Universo original", len(tickers))

    if not approved_bt_df.empty:
        with st.expander("✅ Ativos aprovados pelo histórico"):
            view_ok = approved_bt_df[["ticker", "trades", "win_rate", "avg_r", "net_profit", "profit_factor"]].copy()
            view_ok.columns = ["Ticker", "Trades", "Win rate", "Média em R", "Lucro líquido", "Profit factor"]
            view_ok["Win rate"] = view_ok["Win rate"].map(format_pct)
            view_ok["Média em R"] = view_ok["Média em R"].map(lambda x: f"{x:.2f}")
            view_ok["Lucro líquido"] = view_ok["Lucro líquido"].map(format_money)
            view_ok["Profit factor"] = view_ok["Profit factor"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
            st.dataframe(view_ok, use_container_width=True, hide_index=True)

    if not blocked_bt_df.empty:
        with st.expander("🚫 Ativos bloqueados pelo histórico"):
            view_blocked = blocked_bt_df[["ticker", "trades", "win_rate", "avg_r", "net_profit", "profit_factor", "motivos"]].copy()
            view_blocked.columns = ["Ticker", "Trades", "Win rate", "Média em R", "Lucro líquido", "Profit factor", "Motivos"]
            view_blocked["Win rate"] = view_blocked["Win rate"].map(format_pct)
            view_blocked["Média em R"] = view_blocked["Média em R"].map(lambda x: f"{x:.2f}")
            view_blocked["Lucro líquido"] = view_blocked["Lucro líquido"].map(format_money)
            view_blocked["Profit factor"] = view_blocked["Profit factor"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
            st.dataframe(view_blocked, use_container_width=True, hide_index=True)

if not debug_df.empty:
    problemas = debug_df[debug_df["status"] != "ok"]
    if not problemas.empty:
        with st.expander("Ver ativos bloqueados / com problema"):
            st.dataframe(problemas, use_container_width=True, hide_index=True)

daily_loss_limit_value = capital * daily_loss_limit_pct
regime_icon = REGIME_COLORS.get(regime_info["regime"], "•")

st.subheader("Panorama do mercado")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Regime atual", f"{regime_icon} {regime_info['regime']}")
m2.metric("Benchmark", benchmark)
m3.metric("RSI do índice", f"{regime_info['rsi']:.1f}" if pd.notna(regime_info["rsi"]) else "-")
m4.metric("Perda máxima no dia", format_money(daily_loss_limit_value))
st.caption(regime_info["reason"])

best_row = summary.iloc[0]
decision_icon = SIGNAL_COLORS.get(best_row["decision"], "•")
st.markdown(f"### {decision_icon} Decisão final do melhor ativo: **{best_row['ticker']} — {best_row['decision']}**")
st.caption(f"Score {best_row['score']:.0f} | Entrada {best_row['entry']:.2f} | Stop {best_row['stop']:.2f} | Alvo {best_row['target']:.2f}")

st.subheader("Top oportunidades")
approved_now_df = summary[summary["decision"] == "OPERAR AGORA"].copy()
if not approved_now_df.empty:
    st.caption(f"Mostrando {len(approved_now_df)} ativo(s) com sinal OPERAR AGORA dentro do universo aprovado.")
    top_df = approved_now_df.head(top_n).copy()
else:
    st.caption("Nenhum ativo está em OPERAR AGORA agora. O robô está preservando caixa.")
    top_df = summary.head(top_n).copy()
cols = st.columns(len(top_df))
for col, (_, row) in zip(cols, top_df.iterrows()):
    signal_icon = SIGNAL_COLORS.get(row["decision"], "•")
    with col:
        st.markdown(f"#### {signal_icon} {row['ticker']}")
        st.write(f"**Decisão:** {row['decision']}")
        st.write(f"**Confiança:** {row['confidence']} ({row['confidence_pct'] * 100:.0f}%)")
        st.write(f"**Score:** {row['score']:.0f}")
        st.write(f"**Preço atual:** {format_money(row['close'])}")
        st.write(f"**Entrada:** {format_money(row['entry'])}")
        st.write(f"**Stop:** {format_money(row['stop'])}")
        st.write(f"**Parcial:** {format_money(row['partial'])}")
        st.write(f"**Alvo:** {format_money(row['target'])}")
        st.write(f"**Potencial ganho:** {format_pct(row['upside_pct'])}")
        st.write(f"**Risco perda:** {format_pct(row['stop_pct'])}")
        st.write(f"**R:R:** {row['risk_reward']:.2f}")
        st.write(f"**Qtd sugerida:** {int(row['qty'])}")
        st.write(f"**Valor da posição:** {format_money(row['position_value'])}")
        st.write(f"**Fortes:** {row['strengths']}")
        st.write(f"**Alertas:** {row['alerts']}")
        st.write(f"**Bloqueios:** {row['hard_blocks']}")

with st.expander("📋 Ver ranking completo"):
    display_df = summary[[
        "ticker", "decision", "score", "close", "entry", "stop", "partial", "target",
        "risk_reward", "stop_pct", "upside_pct", "qty", "position_value", "position_pct",
        "rsi", "rvol", "volatility", "relative_strength", "liquidity_value", "strengths", "alerts", "hard_blocks"
    ]].copy()
    display_df.columns = [
        "Ticker", "Decisão", "Score", "Preço", "Entrada", "Stop", "Parcial", "Alvo",
        "Risco/Retorno", "Risco %", "Potencial %", "Qtd", "Valor Posição", "Exposição %",
        "RSI", "RVOL efetivo", "Volatilidade", "Força Relativa", "Liquidez Média", "Fortes", "Alertas", "Bloqueios"
    ]
    display_df["Preço"] = display_df["Preço"].map(format_money)
    display_df["Entrada"] = display_df["Entrada"].map(format_money)
    display_df["Stop"] = display_df["Stop"].map(format_money)
    display_df["Parcial"] = display_df["Parcial"].map(format_money)
    display_df["Alvo"] = display_df["Alvo"].map(format_money)
    display_df["Valor Posição"] = display_df["Valor Posição"].map(format_money)
    display_df["Liquidez Média"] = display_df["Liquidez Média"].map(format_money)
    display_df["Risco/Retorno"] = display_df["Risco/Retorno"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
    display_df["Risco %"] = display_df["Risco %"].map(format_pct)
    display_df["Potencial %"] = display_df["Potencial %"].map(format_pct)
    display_df["Exposição %"] = display_df["Exposição %"].map(format_pct)
    display_df["Volatilidade"] = display_df["Volatilidade"].map(format_pct)
    display_df["Força Relativa"] = display_df["Força Relativa"].map(format_pct)
    display_df["RSI"] = display_df["RSI"].map(lambda x: f"{x:.1f}")
    display_df["RVOL efetivo"] = display_df["RVOL efetivo"].map(lambda x: f"{x:.2f}")
    display_df["Qtd"] = display_df["Qtd"].map(lambda x: int(x))
    st.dataframe(display_df, use_container_width=True, hide_index=True)

selected_label = st.selectbox("Ativo para ver o gráfico detalhado", options=summary["ticker"].tolist(), index=0)
selected_ticker = normalize_ticker(selected_label)
selected_row = summary.loc[summary["ticker"] == selected_label].iloc[0]
selected_data = chart_map[selected_ticker]

st.subheader(f"📈 Gráfico detalhado — {selected_label}")
price_fig = build_price_chart(
    selected_data,
    title=f"{selected_label} | {selected_row['decision']}",
    entry=float(selected_row["entry"]),
    stop=float(selected_row["stop"]),
    partial=float(selected_row["partial"]),
    target=float(selected_row["target"]),
)
st.plotly_chart(price_fig, use_container_width=True)

latest_close = float(selected_row["close"])
alert_above = safe_float(alert_above_text)
alert_below = safe_float(alert_below_text)
manual_alerts = []
if alert_above is not None and latest_close >= alert_above:
    manual_alerts.append(f"Preço atingiu ou superou o alerta de alta: {alert_above:.2f}")
if alert_below is not None and latest_close <= alert_below:
    manual_alerts.append(f"Preço atingiu ou perdeu o alerta de baixa: {alert_below:.2f}")

if manual_alerts:
    for msg in manual_alerts:
        st.warning(msg)
else:
    st.info("Nenhum alerta manual acionado no ativo selecionado.")

st.subheader("Resumo operacional")
r1, r2, r3, r4 = st.columns(4)
r1.metric("Decisão", selected_row["decision"])
r2.metric("Preço atual", format_money(selected_row["close"]))
r3.metric("Risco / Retorno", f"{selected_row['risk_reward']:.2f}")
r4.metric("Potencial", format_pct(float(selected_row["upside_pct"])))

r5, r6, r7, r8 = st.columns(4)
r5.metric("RSI 14", f"{selected_row['rsi']:.1f}")
r6.metric("RVOL", f"{selected_row['rvol']:.2f}")
r7.metric("Volatilidade 20", format_pct(float(selected_row["volatility"])))
r8.metric("Força Relativa", format_pct(float(selected_row["relative_strength"])) if pd.notna(selected_row["relative_strength"]) else "-")

r9, r10, r11, r12 = st.columns(4)
r9.metric("Qtd sugerida", int(selected_row["qty"]))
r10.metric("Valor da posição", format_money(selected_row["position_value"]))
r11.metric("Risco por ação", format_money(selected_row["risk_per_share"]))
r12.metric("Risco financeiro", format_money(selected_row["risk_amount"]))

r13, r14 = st.columns(2)
r13.metric("Confiança do setup", f"{selected_row['confidence']} | {selected_row['confidence_pct'] * 100:.0f}%")
r14.metric("Risco base", format_money(selected_row["base_risk_amount"]))

if run_backtest_toggle:
    st.divider()
    st.subheader("🧪 Backtest rápido da estratégia")
    if bt_trades.empty:
        st.warning("O backtest não encontrou trades com as regras atuais.")
        if not bt_debug.empty:
            st.dataframe(bt_debug, use_container_width=True, hide_index=True)
    else:
        s = bt_summary.iloc[0]
        b1, b2, b3, b4, b5 = st.columns(5)
        b1.metric("Trades", int(s["trades"]))
        b2.metric("Win rate", format_pct(float(s["win_rate"])))
        b3.metric("Lucro líquido", format_money(float(s["net_profit"])))
        b4.metric("Profit factor", f"{s['profit_factor']:.2f}" if pd.notna(s["profit_factor"]) else "-")
        b5.metric("Drawdown máx.", format_pct(float(s["max_drawdown"])))

        b6, b7, b8, b9, b10 = st.columns(5)
        b6.metric("Capital final", format_money(float(s["final_equity"])))
        b7.metric("Payoff", f"{s['payoff']:.2f}" if pd.notna(s["payoff"]) else "-")
        b8.metric("Expectância (R)", f"{s['expectancy_r']:.2f}" if pd.notna(s["expectancy_r"]) else "-")
        b9.metric("Vencedores", int(s["wins"]))
        b10.metric("Perdedores", int(s["losses"]))

        st.plotly_chart(build_equity_chart(bt_trades), use_container_width=True)

        with st.expander("📊 Resultado por ativo"):
            view_bt = bt_by_ticker.copy()
            view_bt = view_bt[["ticker", "trades", "net_profit", "win_rate", "avg_r", "profit_factor", "avg_score", "winners", "losers"]]
            view_bt.columns = ["Ticker", "Trades", "Lucro líquido", "Win rate", "Média em R", "Profit factor", "Score médio", "Vencedores", "Perdedores"]
            view_bt["Lucro líquido"] = view_bt["Lucro líquido"].map(format_money)
            view_bt["Win rate"] = view_bt["Win rate"].map(format_pct)
            view_bt["Média em R"] = view_bt["Média em R"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
            view_bt["Profit factor"] = view_bt["Profit factor"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
            view_bt["Score médio"] = view_bt["Score médio"].map(lambda x: f"{x:.1f}")
            st.dataframe(view_bt, use_container_width=True, hide_index=True)

        with st.expander("🧾 Log de trades do backtest"):
            view_trades = bt_trades.copy()
            for col in ["entry", "exit", "stop", "partial", "target", "pnl"]:
                view_trades[col] = view_trades[col].map(format_money)
            view_trades["ret_capital"] = view_trades["ret_capital"].map(format_pct)
            view_trades["r_multiple"] = view_trades["r_multiple"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
            view_trades["score"] = view_trades["score"].map(lambda x: f"{x:.0f}")
            st.dataframe(view_trades, use_container_width=True, hide_index=True)

with st.expander("Como o robô decide"):
    st.markdown(
        """
        - **OPERAR AGORA**: contexto alinhado, horário válido, candle forte e sem bloqueios.
        - **OBSERVAR**: ativo interessante, mas faltou confirmação.
        - **NÃO OPERAR**: o robô recusou o trade por filtro duro.

        **Melhorias da V8 (perfil agressivo-controlado):**
        - continua usando filtro histórico por ativo
        - privilegia ativos com mais força relativa e RVOL
        - adiciona nota de confiança do setup (A+, A, B, C)
        - aumenta o risco financeiro apenas nos setups mais fortes
        - mantém trava de horário, índice intraday e candle de força
        - preserva o foco em operar pouco, mas com mais qualidade

        **Bloqueios duros atuais:**
        - fora da janela de horário
        - mercado defensivo
        - tendência diária fraca
        - estrutura 60m fraca
        - gatilho abaixo da EMA21
        - preço abaixo da VWAP
        - RVOL abaixo do mínimo
        - RSI acima do máximo
        - sem rompimento quando exigido
        - candle sem força quando exigido
        - risco/retorno abaixo do mínimo
        - potencial abaixo do mínimo
        - liquidez abaixo do mínimo
        - score abaixo do mínimo de entrada
        """
    )

st.caption("Uso educacional. Não é garantia de lucro e não substitui gerenciamento de risco, backtest amplo, taxas reais e disciplina operacional.")
st.info("Dica: se o navegador estiver traduzindo a página automaticamente, alguns rótulos podem ficar estranhos. Desative a tradução nesta página para ver os textos corretamente.")
