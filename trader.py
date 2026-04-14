import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf


st.set_page_config(
    page_title="Robô Pessoal de Análise de Ações — V4",
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
        preparado = _prepare_ohlcv(df)
        if not preparado.empty:
            return preparado

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
    if data.empty:
        return pd.Series(dtype=float)

    idx = pd.to_datetime(data.index)
    session_key = pd.Series(idx.date, index=data.index)

    typical_price = (data["High"] + data["Low"] + data["Close"]) / 3
    tpv = typical_price * data["Volume"]

    cum_tpv = tpv.groupby(session_key).cumsum()
    cum_vol = data["Volume"].groupby(session_key).cumsum().replace(0, np.nan)

    return cum_tpv / cum_vol


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

    if interval != "1d":
        data["VWAP"] = add_session_vwap(data)
    else:
        data["VWAP"] = np.nan

    data["ForcaRelativa"] = np.nan
    if benchmark_close is not None and not benchmark_close.empty:
        aligned_bench = benchmark_close.reindex(data.index).ffill().bfill()

        if len(data) >= 60:
            lookback = 20
        elif len(data) >= 30:
            lookback = 10
        else:
            lookback = 5

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
    ]

    if interval != "1d":
        required.append("VWAP")

    data = data.dropna(subset=required).copy()
    return data


# =========================================================
# REGIME DE MERCADO
# =========================================================
def detect_market_regime(benchmark_df: pd.DataFrame) -> Dict[str, float]:
    if benchmark_df.empty or len(benchmark_df) < 60:
        return {
            "regime": "LATERAL",
            "reason": "Índice com poucos dados para confirmar tendência.",
            "close": np.nan,
            "ema21": np.nan,
            "ema50": np.nan,
            "rsi": np.nan,
            "ret20": np.nan,
        }

    bench = benchmark_df.copy()
    bench["EMA21"] = bench["Close"].ewm(span=21, adjust=False).mean()
    bench["EMA50"] = bench["Close"].ewm(span=50, adjust=False).mean()
    bench["RSI14"] = calculate_rsi(bench["Close"], 14)
    bench["RET20"] = bench["Close"].pct_change(20)
    bench = bench.dropna().copy()

    if bench.empty:
        return {
            "regime": "LATERAL",
            "reason": "Índice sem dados válidos após cálculo do regime.",
            "close": np.nan,
            "ema21": np.nan,
            "ema50": np.nan,
            "rsi": np.nan,
            "ret20": np.nan,
        }

    last = bench.iloc[-1]
    close = float(last["Close"])
    ema21 = float(last["EMA21"])
    ema50 = float(last["EMA50"])
    rsi = float(last["RSI14"])
    ret20 = float(last["RET20"])

    if close > ema21 > ema50 and rsi >= 52 and ret20 > 0:
        regime = "COMPRADOR"
        reason = "Índice acima da EMA21/EMA50, RSI positivo e retorno recente positivo."
    elif close < ema21 < ema50 and rsi < 48 and ret20 < 0:
        regime = "DEFENSIVO"
        reason = "Índice abaixo da EMA21/EMA50, RSI fraco e retorno recente negativo."
    else:
        regime = "LATERAL"
        reason = "Mercado sem alinhamento claro de tendência."

    return {
        "regime": regime,
        "reason": reason,
        "close": close,
        "ema21": ema21,
        "ema50": ema50,
        "rsi": rsi,
        "ret20": ret20,
    }


# =========================================================
# RISCO E TAMANHO DE POSIÇÃO
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

    rr = (target - entry) / risk if risk > 0 else np.nan

    return {
        "entry": entry,
        "stop": stop,
        "target": target,
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
            "qty_risco": 0,
            "qty_caixa": 0,
            "qty_exposicao": 0,
            "qty": 0,
            "position_value": 0,
            "position_pct": 0,
        }

    qty_risco = math.floor(risk_amount / risk_per_share)
    qty_caixa = math.floor(capital / entry)
    qty_exposicao = math.floor((capital * max_position_pct) / entry)

    qty_final = min(qty_risco, qty_caixa, qty_exposicao)
    qty_final = max(qty_final, 0)

    position_value = qty_final * entry
    position_pct = position_value / capital if capital > 0 else 0

    return {
        "risk_amount": risk_amount,
        "risk_per_share": risk_per_share,
        "qty_risco": qty_risco,
        "qty_caixa": qty_caixa,
        "qty_exposicao": qty_exposicao,
        "qty": qty_final,
        "position_value": position_value,
        "position_pct": position_pct,
    }


# =========================================================
# LÓGICA PRINCIPAL
# =========================================================
def get_structure_settings(trigger_interval: str) -> Tuple[str, str]:
    if trigger_interval == "1d":
        return "3mo", "1d"
    return "1mo", "60m"


def evaluate_asset(
    ticker: str,
    daily_data: pd.DataFrame,
    structure_data: pd.DataFrame,
    trigger_data: pd.DataFrame,
    benchmark_daily_close: pd.Series,
    regime_info: Dict[str, float],
    capital: float,
    risk_per_trade_pct: float,
    min_rr: float,
    max_position_pct: float,
    min_upside_pct: float,
    min_liquidity_million: float,
) -> Dict[str, float]:
    daily = add_indicators(daily_data, "1d", benchmark_daily_close)
    structure_interval = "1d" if len(structure_data) == len(daily_data) else "60m"
    structure = add_indicators(structure_data, structure_interval, None)
    trigger_interval = "1d" if trigger_data.index.equals(daily_data.index) else "intraday"
    trigger = add_indicators(trigger_data, "1d" if trigger_interval == "1d" else "15m", None)

    if daily.empty:
        raise ValueError("diário sem indicadores")
    if structure.empty:
        raise ValueError("estrutura sem indicadores")
    if trigger.empty:
        raise ValueError("gatilho sem indicadores")

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
    trigger_rvol = float(t["RVOL"])
    trigger_vol = float(t["Volatilidade20"])
    liquidity_value = float(t["ValorFinanceiroMedio"])
    rel_strength = float(d["ForcaRelativa"]) if pd.notna(d["ForcaRelativa"]) else np.nan

    stop_pct = (levels["entry"] - levels["stop"]) / levels["entry"] if levels["entry"] > 0 else np.nan
    upside_pct = (levels["target"] - levels["entry"]) / levels["entry"] if levels["entry"] > 0 else np.nan

    # ---------------------------
    # SCORE
    # ---------------------------
    if regime_info["regime"] == "COMPRADOR":
        score += 12
        strengths.append("Mercado favorável")
    elif regime_info["regime"] == "LATERAL":
        score -= 5
        alerts.append("Mercado lateral")
    else:
        score -= 20
        alerts.append("Mercado defensivo")

    # Tendência diária
    if close_daily > daily_ema21 > daily_ema50:
        score += 24
        strengths.append("Tendência diária alinhada")
    elif close_daily > daily_ema21:
        score += 10
    else:
        score -= 20
        alerts.append("Diário abaixo da EMA21")

    # Estrutura 60m
    if close_structure > structure_ema21 and structure_ema9 > structure_ema21:
        score += 18
        strengths.append("Estrutura intraday positiva")
    elif close_structure > structure_ema21:
        score += 8
    else:
        score -= 12
        alerts.append("Estrutura fraca")

    # Gatilho
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

    if 54 <= trigger_rsi <= 68:
        score += 10
        strengths.append("Momentum saudável")
    elif 68 < trigger_rsi <= 75:
        score += 4
        alerts.append("Levemente esticada")
    elif trigger_rsi < 45:
        score -= 8
        alerts.append("RSI fraco")
    elif trigger_rsi > 80:
        score -= 12
        alerts.append("Muito esticada")

    if trigger_rvol >= 1.5:
        score += 12
        strengths.append("Volume forte")
    elif trigger_rvol >= 1.1:
        score += 6
    elif trigger_rvol < 0.8:
        score -= 15
        alerts.append("RVOL muito fraco")

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

    # Liquidez
    liquidity_cut = min_liquidity_million * 1_000_000
    if liquidity_value >= liquidity_cut:
        score += 8
        strengths.append("Boa liquidez")
    else:
        score -= 18
        alerts.append("Liquidez fraca")

    # ---------------------------
    # BLOQUEIOS DUROS
    # ---------------------------
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

    if trigger_rvol < 0.80:
        hard_blocks.append("RVOL abaixo de 0.80")

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

    score = max(min(score, 100), 0)

    if hard_blocks:
        decision = "NÃO OPERAR"
    elif score >= 80:
        decision = "OPERAR AGORA"
    elif score >= 60:
        decision = "OBSERVAR"
    else:
        decision = "NÃO OPERAR"

    return {
        "ticker": ticker.replace(".SA", ""),
        "score": score,
        "decision": decision,
        "regime": regime_info["regime"],
        "close": close_trigger,
        "entry": levels["entry"],
        "stop": levels["stop"],
        "target": levels["target"],
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
        "risk_amount": position["risk_amount"],
        "risk_per_share": position["risk_per_share"],
        "qty_risco": position["qty_risco"],
        "qty_caixa": position["qty_caixa"],
        "qty_exposicao": position["qty_exposicao"],
        "strengths": " | ".join(strengths[:6]) if strengths else "Sem destaques",
        "alerts": " | ".join(alerts[:6]) if alerts else "Sem alertas",
        "hard_blocks": " | ".join(hard_blocks) if hard_blocks else "Sem bloqueios",
    }


# =========================================================
# RESUMO GERAL
# =========================================================
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
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, float], pd.DataFrame]:
    rows = []
    chart_map: Dict[str, pd.DataFrame] = {}
    debug_rows = []

    benchmark_daily = load_data(benchmark, trend_period, "1d")
    regime_info = detect_market_regime(benchmark_daily)
    benchmark_daily_close = benchmark_daily["Close"] if not benchmark_daily.empty else pd.Series(dtype=float)

    structure_period, structure_interval = get_structure_settings(trigger_interval)

    for ticker in tickers:
        try:
            daily_raw = load_data(ticker, trend_period, "1d")
            structure_raw = load_data(ticker, structure_period, structure_interval)
            trigger_raw = load_data(ticker, trigger_period, trigger_interval)

            if daily_raw.empty:
                debug_rows.append(
                    {"ticker": ticker.replace(".SA", ""), "status": "erro", "motivo": "sem dados no diário"}
                )
                continue

            if structure_raw.empty:
                debug_rows.append(
                    {"ticker": ticker.replace(".SA", ""), "status": "erro", "motivo": "sem dados no timeframe de estrutura"}
                )
                continue

            if trigger_raw.empty:
                debug_rows.append(
                    {"ticker": ticker.replace(".SA", ""), "status": "erro", "motivo": "sem dados no timeframe de gatilho"}
                )
                continue

            evaluated = evaluate_asset(
                ticker=ticker,
                daily_data=daily_raw,
                structure_data=structure_raw,
                trigger_data=trigger_raw,
                benchmark_daily_close=benchmark_daily_close,
                regime_info=regime_info,
                capital=capital,
                risk_per_trade_pct=risk_per_trade_pct,
                min_rr=min_rr,
                max_position_pct=max_position_pct,
                min_upside_pct=min_upside_pct,
                min_liquidity_million=min_liquidity_million,
            )

            rows.append(evaluated)
            chart_map[ticker] = add_indicators(trigger_raw, "1d" if trigger_interval == "1d" else "15m", None)

            debug_rows.append(
                {
                    "ticker": ticker.replace(".SA", ""),
                    "status": "ok" if evaluated["decision"] != "NÃO OPERAR" else "bloqueado",
                    "motivo": evaluated["hard_blocks"],
                }
            )

        except Exception as e:
            debug_rows.append(
                {"ticker": ticker.replace(".SA", ""), "status": "erro", "motivo": str(e)}
            )

    debug_df = pd.DataFrame(debug_rows)

    if not rows:
        return pd.DataFrame(), chart_map, regime_info, debug_df

    summary = pd.DataFrame(rows)

    decision_order = {
        "OPERAR AGORA": 0,
        "OBSERVAR": 1,
        "NÃO OPERAR": 2,
    }
    summary["decision_order"] = summary["decision"].map(decision_order).fillna(9)

    summary["ranking"] = (
        summary["score"]
        + summary["upside_pct"].fillna(0) * 100
        - summary["stop_pct"].fillna(0) * 50
        + summary["relative_strength"].fillna(0) * 100
        + (summary["rvol"].fillna(0) * 4)
    )

    summary = summary.sort_values(
        by=["decision_order", "ranking", "score"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    return summary, chart_map, regime_info, debug_df


# =========================================================
# GRÁFICO
# =========================================================
def build_chart(data: pd.DataFrame, title: str, entry: float, stop: float, target: float) -> go.Figure:
    plot_df = data.tail(120).copy()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.75, 0.25],
    )

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

    fig.add_trace(
        go.Bar(
            x=plot_df.index,
            y=plot_df["Volume"],
            name="Volume",
        ),
        row=2,
        col=1,
    )

    fig.add_hrect(
        y0=stop,
        y1=entry,
        fillcolor="rgba(255,0,0,0.12)",
        line_width=0,
        row=1,
        col=1,
    )

    fig.add_hrect(
        y0=entry,
        y1=target,
        fillcolor="rgba(0,255,0,0.12)",
        line_width=0,
        row=1,
        col=1,
    )

    for level, name in [(entry, "Entrada"), (stop, "Stop"), (target, "Alvo")]:
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


# =========================================================
# INTERFACE
# =========================================================
st.title("📈 Robô Pessoal de Análise de Ações — V4")
st.caption("Foco em recusar operações ruins, controlar risco e priorizar contexto forte.")

with st.sidebar:
    st.header("Configurações")

    tickers_text = st.text_area(
        "Ativos monitorados (separados por vírgula)",
        value=DEFAULT_TICKERS,
        help="Ex.: PETR4, VALE3, BBAS3",
    )

    trigger_interval_label = st.selectbox(
        "Intervalo de gatilho",
        list(TRIGGER_INTERVAL_OPTIONS.keys()),
        index=1,
    )

    trigger_period_label = st.selectbox(
        "Período do gatilho",
        list(TRIGGER_PERIOD_OPTIONS.keys()),
        index=0,
    )

    trend_period_label = st.selectbox(
        "Período diário para tendência maior",
        list(TREND_PERIOD_OPTIONS.keys()),
        index=0,
    )

    benchmark = st.text_input("Benchmark", value="^BVSP").strip().upper() or "^BVSP"

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

summary, chart_map, regime_info, debug_df = build_summary(
    tickers=tickers,
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
)

if summary.empty:
    st.error("O app abriu, mas nenhum ativo retornou dados válidos para montar a análise.")
    if not debug_df.empty:
        st.dataframe(debug_df, use_container_width=True, hide_index=True)
    st.stop()

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

st.markdown(
    f"### {decision_icon} Decisão final do melhor ativo: **{best_row['ticker']} — {best_row['decision']}**"
)
st.caption(
    f"Score {best_row['score']:.0f} | Entrada {best_row['entry']:.2f} | Stop {best_row['stop']:.2f} | Alvo {best_row['target']:.2f}"
)

st.subheader("Top oportunidades")
top_df = summary.head(top_n).copy()
cols = st.columns(len(top_df))

for col, (_, row) in zip(cols, top_df.iterrows()):
    signal_icon = SIGNAL_COLORS.get(row["decision"], "•")
    with col:
        st.markdown(f"#### {signal_icon} {row['ticker']}")
        st.write(f"**Decisão:** {row['decision']}")
        st.write(f"**Score:** {row['score']:.0f}")
        st.write(f"**Preço atual:** {format_money(row['close'])}")
        st.write(f"**Entrada:** {format_money(row['entry'])}")
        st.write(f"**Stop:** {format_money(row['stop'])}")
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
    display_df = summary[
        [
            "ticker",
            "decision",
            "score",
            "close",
            "entry",
            "stop",
            "target",
            "risk_reward",
            "stop_pct",
            "upside_pct",
            "qty",
            "position_value",
            "position_pct",
            "rsi",
            "rvol",
            "volatility",
            "relative_strength",
            "liquidity_value",
            "strengths",
            "alerts",
            "hard_blocks",
        ]
    ].copy()

    display_df.columns = [
        "Ticker",
        "Decisão",
        "Score",
        "Preço",
        "Entrada",
        "Stop",
        "Alvo",
        "Risco/Retorno",
        "Risco %",
        "Potencial %",
        "Qtd",
        "Valor Posição",
        "Exposição %",
        "RSI",
        "RVOL",
        "Volatilidade",
        "Força Relativa",
        "Liquidez Média",
        "Fortes",
        "Alertas",
        "Bloqueios",
    ]

    display_df["Risco/Retorno"] = display_df["Risco/Retorno"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
    display_df["Risco %"] = display_df["Risco %"].map(format_pct)
    display_df["Potencial %"] = display_df["Potencial %"].map(format_pct)
    display_df["Exposição %"] = display_df["Exposição %"].map(format_pct)
    display_df["Volatilidade"] = display_df["Volatilidade"].map(format_pct)
    display_df["Força Relativa"] = display_df["Força Relativa"].map(format_pct)
    display_df["Preço"] = display_df["Preço"].map(format_money)
    display_df["Entrada"] = display_df["Entrada"].map(format_money)
    display_df["Stop"] = display_df["Stop"].map(format_money)
    display_df["Alvo"] = display_df["Alvo"].map(format_money)
    display_df["Valor Posição"] = display_df["Valor Posição"].map(format_money)
    display_df["Liquidez Média"] = display_df["Liquidez Média"].map(format_money)
    display_df["RSI"] = display_df["RSI"].map(lambda x: f"{x:.1f}")
    display_df["RVOL"] = display_df["RVOL"].map(lambda x: f"{x:.2f}")
    display_df["Qtd"] = display_df["Qtd"].map(lambda x: int(x))

    st.dataframe(display_df, use_container_width=True, hide_index=True)

selected_label = st.selectbox(
    "Ativo para ver o gráfico detalhado",
    options=summary["ticker"].tolist(),
    index=0,
)

selected_ticker = normalize_ticker(selected_label)
selected_row = summary.loc[summary["ticker"] == selected_label].iloc[0]
selected_data = chart_map[selected_ticker]

st.subheader(f"📈 Gráfico detalhado — {selected_label}")
fig = build_chart(
    selected_data,
    title=f"{selected_label} | {selected_row['decision']}",
    entry=float(selected_row["entry"]),
    stop=float(selected_row["stop"]),
    target=float(selected_row["target"]),
)
st.plotly_chart(fig, use_container_width=True)

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

with st.expander("Como o robô decide"):
    st.markdown(
        """
        - **OPERAR AGORA**: só aparece quando o contexto está alinhado e não há bloqueios duros.
        - **OBSERVAR**: ativo interessante, mas sem confirmação suficiente.
        - **NÃO OPERAR**: o robô recusou o trade por contexto ruim ou filtro duro.

        **Bloqueios duros atuais:**
        - mercado defensivo
        - tendência diária fraca
        - estrutura 60m fraca
        - gatilho abaixo da EMA21
        - preço abaixo da VWAP
        - RVOL abaixo de 0.80
        - risco/retorno abaixo do mínimo
        - potencial abaixo do mínimo
        - liquidez abaixo do mínimo
        - quantidade inviável
        - posição acima do capital
        - posição acima da exposição máxima
        """
    )

st.caption("Uso educacional. Não é garantia de lucro e não substitui gerenciamento de risco, backtest e disciplina operacional.")
