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
    page_title="Robô de Análise de Ações — V3",
    page_icon="📈",
    layout="wide",
)

DEFAULT_TICKERS = "PETR4, VALE3, BBAS3, ITUB4, WEGE3, BBDC4, ABEV3, RENT3"

PERIOD_OPTIONS = {
    "5 dias": "5d",
    "1 mês": "1mo",
    "3 meses": "3mo",
    "6 meses": "6mo",
    "1 ano": "1y",
}

INTERVAL_OPTIONS = {
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


@st.cache_data(ttl=900, show_spinner=False)
def load_benchmark(period: str, interval: str, benchmark: str) -> pd.DataFrame:
    tentativas = []

    try:
        df1 = yf.download(
            benchmark,
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
        df2 = yf.Ticker(benchmark).history(
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


def calculate_indicators(df: pd.DataFrame, benchmark_close: Optional[pd.Series] = None) -> pd.DataFrame:
    data = df.copy()

    data["EMA9"] = data["Close"].ewm(span=9, adjust=False).mean()
    data["EMA21"] = data["Close"].ewm(span=21, adjust=False).mean()
    data["EMA50"] = data["Close"].ewm(span=50, adjust=False).mean()
    data["EMA200"] = data["Close"].ewm(span=200, adjust=False).mean()

    typical_price = (data["High"] + data["Low"] + data["Close"]) / 3
    vol_cumsum = data["Volume"].replace(0, np.nan).cumsum()
    data["VWAP"] = (typical_price * data["Volume"]).cumsum() / vol_cumsum

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

    # força relativa adaptativa
    data["ForcaRelativa"] = np.nan
    if benchmark_close is not None and not benchmark_close.empty:
        aligned_bench = benchmark_close.reindex(data.index).ffill().bfill()
        if len(data) >= 60:
            lookback = 60
        elif len(data) >= 30:
            lookback = 20
        elif len(data) >= 15:
            lookback = 10
        else:
            lookback = 5

        asset_ret = data["Close"].pct_change(lookback)
        bench_ret = aligned_bench.pct_change(lookback)
        data["ForcaRelativa"] = asset_ret - bench_ret

    subset_cols = [
        "EMA9",
        "EMA21",
        "EMA50",
        "EMA200",
        "VWAP",
        "ATR14",
        "Volatilidade20",
        "VolumeMA20",
        "RVOL",
        "RSI14",
        "Max20",
        "Min20",
        "Min10",
    ]

    data = data.dropna(subset=subset_cols).copy()
    return data


def detect_market_regime(benchmark_df: pd.DataFrame) -> Dict[str, str]:
    if benchmark_df.empty or len(benchmark_df) < 30:
        return {
            "regime": "LATERAL",
            "reason": "Sem dados suficientes do índice.",
            "close": np.nan,
            "ema21": np.nan,
            "ema50": np.nan,
            "rsi": np.nan,
        }

    bench = calculate_indicators(benchmark_df)
    if bench.empty:
        return {
            "regime": "LATERAL",
            "reason": "Índice sem dados válidos após indicadores.",
            "close": np.nan,
            "ema21": np.nan,
            "ema50": np.nan,
            "rsi": np.nan,
        }

    last = bench.iloc[-1]

    close = float(last["Close"])
    ema21 = float(last["EMA21"])
    ema50 = float(last["EMA50"])
    rsi = float(last["RSI14"])

    if close > ema21 > ema50 and rsi >= 52:
        regime = "COMPRADOR"
        reason = "Índice acima das médias e com momentum positivo."
    elif close < ema21 < ema50 and rsi < 48:
        regime = "DEFENSIVO"
        reason = "Índice abaixo das médias e com momentum fraco."
    else:
        regime = "LATERAL"
        reason = "Mercado sem tendência clara."

    return {
        "regime": regime,
        "reason": reason,
        "close": close,
        "ema21": ema21,
        "ema50": ema50,
        "rsi": rsi,
    }


# =========================================================
# TRADE / RISCO
# =========================================================
def compute_trade_levels(data: pd.DataFrame) -> Dict[str, float]:
    last = data.iloc[-1]

    entry = float(last["Close"])
    atr = float(last["ATR14"])
    recent_support = float(last["Min10"])

    stop = max(recent_support, entry - 1.5 * atr)
    if stop >= entry:
        stop = entry * 0.9875

    risk = max(entry - stop, entry * 0.006)
    target = entry + (risk * 2.2)
    rr = (target - entry) / risk if risk > 0 else np.nan

    return {
        "entry": entry,
        "stop": stop,
        "target": target,
        "risk": risk,
        "rr": rr,
    }


def calculate_position_size(capital: float, risk_per_trade_pct: float, entry: float, stop: float) -> Dict[str, float]:
    capital = max(capital, 0)
    risk_amount = capital * risk_per_trade_pct
    risk_per_share = max(entry - stop, 0)

    if risk_per_share <= 0 or entry <= 0:
        return {
            "risk_amount": risk_amount,
            "risk_per_share": 0,
            "qty": 0,
            "position_value": 0,
            "position_pct": 0,
        }

    qty = math.floor(risk_amount / risk_per_share)
    position_value = qty * entry
    position_pct = position_value / capital if capital > 0 else 0

    return {
        "risk_amount": risk_amount,
        "risk_per_share": risk_per_share,
        "qty": max(qty, 0),
        "position_value": position_value,
        "position_pct": position_pct,
    }


def score_asset(
    data: pd.DataFrame,
    regime: str,
    capital: float,
    risk_per_trade_pct: float,
    min_rr: float,
) -> Dict[str, float]:
    last = data.iloc[-1]
    levels = compute_trade_levels(data)
    position = calculate_position_size(capital, risk_per_trade_pct, levels["entry"], levels["stop"])

    score = 0
    fortes: List[str] = []
    alertas: List[str] = []

    close = float(last["Close"])
    ema9 = float(last["EMA9"])
    ema21 = float(last["EMA21"])
    ema50 = float(last["EMA50"])
    ema200 = float(last["EMA200"])
    vwap = float(last["VWAP"])
    rsi = float(last["RSI14"])
    rvol = float(last["RVOL"])
    vol = float(last["Volatilidade20"])

    rs_value = last["ForcaRelativa"]
    rs = float(rs_value) if pd.notna(rs_value) else np.nan

    max20_prev = float(data["Max20"].shift(1).iloc[-1]) if len(data) > 1 else float(last["Max20"])

    if close > ema21 > ema50 > ema200:
        score += 30
        fortes.append("Tendência maior forte")
    elif close > ema21 > ema50:
        score += 22
        fortes.append("Tendência positiva")
    elif close > ema21:
        score += 12
    else:
        score -= 18
        alertas.append("Abaixo das médias")

    if close > vwap:
        score += 10
        fortes.append("Acima da VWAP")
    else:
        score -= 8
        alertas.append("Abaixo da VWAP")

    if ema9 > ema21:
        score += 10
        fortes.append("EMA9 acima da EMA21")
    else:
        score -= 5

    if 52 <= rsi <= 68:
        score += 14
        fortes.append("Momentum saudável")
    elif 68 < rsi <= 75:
        score += 8
        alertas.append("Levemente esticada")
    elif rsi < 45:
        score -= 8
        alertas.append("Momentum fraco")
    elif rsi > 80:
        score -= 15
        alertas.append("Muito esticada")

    if rvol >= 1.5:
        score += 12
        fortes.append("Volume forte")
    elif rvol >= 1.1:
        score += 7
    elif rvol < 0.8:
        score -= 8
        alertas.append("Volume fraco")

    if pd.notna(rs):
        if rs > 0.08:
            score += 12
            fortes.append("Força relativa forte")
        elif rs > 0.02:
            score += 6
        elif rs < -0.03:
            score -= 8
            alertas.append("Mais fraca que o índice")

    if close > max20_prev:
        score += 8
        fortes.append("Rompimento de máxima")
    elif close >= max20_prev * 0.98:
        score += 4

    if 0.12 <= vol <= 0.45:
        score += 6
    elif vol > 0.65:
        score -= 8
        alertas.append("Volatilidade alta")

    stop_pct = (levels["entry"] - levels["stop"]) / levels["entry"]
    upside_pct = (levels["target"] - levels["entry"]) / levels["entry"]

    if levels["rr"] >= min_rr:
        score += 8
        fortes.append("Risco/retorno favorável")
    else:
        score -= 12
        alertas.append("Risco/retorno abaixo do mínimo")

    if stop_pct <= 0.08:
        score += 8
        fortes.append("Risco controlado")
    elif stop_pct > 0.15:
        score -= 8
        alertas.append("Stop muito longo")

    if position["position_pct"] > 0.35:
        score -= 10
        alertas.append("Exigiria posição muito grande")

    if regime == "COMPRADOR":
        score += 8
        fortes.append("Mercado favorável")
    elif regime == "LATERAL":
        score -= 4
        alertas.append("Mercado lateral")
    elif regime == "DEFENSIVO":
        score -= 18
        alertas.append("Mercado defensivo")

    score = max(min(score, 100), 0)

    must_block = any(
        [
            regime == "DEFENSIVO",
            close < ema21,
            close < vwap,
            levels["rr"] < min_rr,
            position["qty"] == 0,
        ]
    )

    if not must_block and score >= 75:
        decision = "OPERAR AGORA"
    elif score >= 55:
        decision = "OBSERVAR"
    else:
        decision = "NÃO OPERAR"

    if must_block:
        decision = "NÃO OPERAR"

    return {
        "score": score,
        "decision": decision,
        "close": close,
        "ema9": ema9,
        "ema21": ema21,
        "ema50": ema50,
        "ema200": ema200,
        "vwap": vwap,
        "rsi": rsi,
        "rvol": rvol,
        "volatility": vol,
        "relative_strength": rs,
        "entry": levels["entry"],
        "stop": levels["stop"],
        "target": levels["target"],
        "risk_reward": levels["rr"],
        "stop_pct": stop_pct,
        "upside_pct": upside_pct,
        "risk_amount": position["risk_amount"],
        "risk_per_share": position["risk_per_share"],
        "qty": position["qty"],
        "position_value": position["position_value"],
        "position_pct": position["position_pct"],
        "strengths": " | ".join(fortes[:5]) if fortes else "Sem destaques",
        "alerts": " | ".join(alertas[:5]) if alertas else "Sem alertas",
    }


# =========================================================
# RANKING + DIAGNÓSTICO
# =========================================================
def build_summary(
    tickers: List[str],
    intraday_period: str,
    intraday_interval: str,
    benchmark: str,
    trend_period: str,
    capital: float,
    risk_per_trade_pct: float,
    min_rr: float,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, str], pd.DataFrame]:
    rows = []
    data_map: Dict[str, pd.DataFrame] = {}
    debug_rows = []

    intraday_bench = load_benchmark(intraday_period, intraday_interval, benchmark)
    intraday_bench_close = intraday_bench["Close"] if not intraday_bench.empty else pd.Series(dtype=float)

    trend_bench = load_benchmark(trend_period, "1d", benchmark)
    regime_info = detect_market_regime(trend_bench)

    for ticker in tickers:
        raw = load_data(ticker, intraday_period, intraday_interval)

        if raw.empty:
            debug_rows.append(
                {
                    "ticker": ticker.replace(".SA", ""),
                    "status": "sem dados",
                    "motivo": "Yahoo não retornou candles",
                }
            )
            continue

        enriched = calculate_indicators(raw, intraday_bench_close)

        if enriched.empty:
            debug_rows.append(
                {
                    "ticker": ticker.replace(".SA", ""),
                    "status": "sem indicadores",
                    "motivo": "dados insuficientes após indicadores",
                }
            )
            continue

        if len(enriched) < 15:
            debug_rows.append(
                {
                    "ticker": ticker.replace(".SA", ""),
                    "status": "poucos candles",
                    "motivo": f"apenas {len(enriched)} candles válidos",
                }
            )
            continue

        scored = score_asset(
            enriched,
            regime=regime_info["regime"],
            capital=capital,
            risk_per_trade_pct=risk_per_trade_pct,
            min_rr=min_rr,
        )

        scored["ticker"] = ticker.replace(".SA", "")
        rows.append(scored)
        data_map[ticker] = enriched

        debug_rows.append(
            {
                "ticker": ticker.replace(".SA", ""),
                "status": "ok",
                "motivo": f"{len(enriched)} candles válidos",
            }
        )

    debug_df = pd.DataFrame(debug_rows)

    if not rows:
        return pd.DataFrame(), data_map, regime_info, debug_df

    summary = pd.DataFrame(rows)
    summary["ranking"] = (
        summary["score"]
        + summary["upside_pct"] * 100
        - summary["stop_pct"] * 60
        - summary["volatility"].fillna(0) * 5
    )
    summary = summary.sort_values(["ranking", "score"], ascending=False).reset_index(drop=True)

    return summary, data_map, regime_info, debug_df


# =========================================================
# ALERTAS / GRÁFICO
# =========================================================
def check_price_alerts(close: float, above: Optional[float], below: Optional[float]) -> List[str]:
    alerts = []
    if above is not None and close >= above:
        alerts.append(f"Preço atingiu ou superou o alerta de alta: {above:.2f}")
    if below is not None and close <= below:
        alerts.append(f"Preço atingiu ou perdeu o alerta de baixa: {below:.2f}")
    return alerts


def build_chart(data: pd.DataFrame, title: str, entry: float, stop: float, target: float) -> go.Figure:
    last_points = data.tail(120).copy()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.75, 0.25],
    )

    fig.add_trace(
        go.Candlestick(
            x=last_points.index,
            open=last_points["Open"],
            high=last_points["High"],
            low=last_points["Low"],
            close=last_points["Close"],
            name="Preço",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(go.Scatter(x=last_points.index, y=last_points["EMA9"], name="EMA9"), row=1, col=1)
    fig.add_trace(go.Scatter(x=last_points.index, y=last_points["EMA21"], name="EMA21"), row=1, col=1)
    fig.add_trace(go.Scatter(x=last_points.index, y=last_points["VWAP"], name="VWAP"), row=1, col=1)

    fig.add_trace(
        go.Bar(
            x=last_points.index,
            y=last_points["Volume"],
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
st.title("📈 Robô Pessoal de Análise de Ações — V3")
st.caption("Foco em tendência, regime de mercado, risco por operação decisão e objetiva.")

with st.sidebar:
    st.header("configurações")

    tickers_text = st.text_area(
        "Ativos monitorados (separados por vírgula)",
        value=DEFAULT_TICKERS,
        help="Ex.: PETR4, VALE3, BBAS3",
    )

    intraday_interval_label = st.selectbox("Intervalo intradia", list(INTERVAL_OPTIONS.keys()), index=1)
    intraday_period_label = st.selectbox("Período intradia", list(PERIOD_OPTIONS.keys()), index=0)
    trend_period_label = st.selectbox("Período diário para tendência maior", list(TREND_PERIOD_OPTIONS.keys()), index=0)

    benchmark = st.text_input("Referência", value="^BVSP").strip().upper() or "^BVSP"

    st.divider()
    st.subheader("Gestão de risco")

    capital = st.number_input("Capital total (R$)", min_value=100.0, value=10000.0, step=100.0)
    risk_per_trade_pct = st.slider("Risco por operação (%)", 0.25, 3.0, 1.0, 0.25) / 100
    daily_loss_limit_pct = st.slider("Perda máxima no dia (%)", 1.0, 5.0, 2.0, 0.5) / 100
    min_rr = st.slider("Risco/retorno mínimo", 1.5, 4.0, 2.0, 0.1)
    top_n = st.slider("Quantidade de sugestões", 1, 10, 3)

    st.divider()

    auto_refresh = st.checkbox("Atualização automática", value=False)
    refresh_seconds = st.slider("Atualizar a cada (segundos)", 30, 600, 120, 30)
    force_update = st.button("Atualizar agora", use_container_width=True)

    st.divider()
    st.subheader("Alertas do ativo selecionado")

    alert_above_text = st.text_input("Alertar se subir até", value="")
    alert_below_text = st.text_input("Alertar se cair até", value="")


intraday_period = PERIOD_OPTIONS[intraday_period_label]
intraday_interval = INTERVAL_OPTIONS[intraday_interval_label]
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

summary, data_map, regime_info, debug_df = build_summary(
    tickers=tickers,
    intraday_period=intraday_period,
    intraday_interval=intraday_interval,
    benchmark=benchmark,
    trend_period=trend_period,
    capital=capital,
    risk_per_trade_pct=risk_per_trade_pct,
    min_rr=min_rr,
)

if summary.empty:
    st.error("O app abriu, mas nenhum ativo retornou dados válidos para esse intervalo/período.")
    if not debug_df.empty:
        st.dataframe(debug_df, use_container_width=True, hide_index=True)
    st.stop()

if not debug_df.empty:
    falhas = debug_df[debug_df["status"] != "ok"]
    if not falhas.empty:
        with st.expander("Ver ativos com problema"):
            st.dataframe(falhas, use_container_width=True, hide_index=True)

summary_top = summary.head(top_n).copy()

daily_loss_limit_value = capital * daily_loss_limit_pct
regime_icon = REGIME_COLORS.get(regime_info["regime"], "•")

st.subheader("Panorama do mercado")
rc1, rc2, rc3, rc4 = st.columns(4)
rc1.metric("Regime atual", f"{regime_icon} {regime_info['regime']}")
rc2.metric("Benchmark", benchmark)
rc3.metric("RSI do índice", f"{regime_info['rsi']:.1f}" if pd.notna(regime_info["rsi"]) else "-")
rc4.metric("Perda máxima do dia", f"R$ {daily_loss_limit_value:,.2f}")

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
cols = st.columns(len(summary_top))

for col, (_, row) in zip(cols, summary_top.iterrows()):
    signal_icon = SIGNAL_COLORS.get(row["decision"], "•")
    with col:
        st.markdown(f"#### {signal_icon} {row['ticker']}")
        st.write(f"**Decisão:** {row['decision']}")
        st.write(f"**Score:** {row['score']:.0f}")
        st.write(f"**Preço atual:** R$ {row['close']:.2f}")
        st.write(f"**Entrada:** R$ {row['entry']:.2f}")
        st.write(f"**Stop:** R$ {row['stop']:.2f}")
        st.write(f"**Alvo:** R$ {row['target']:.2f}")
        st.write(f"**Potencial ganho:** {row['upside_pct'] * 100:.2f}%")
        st.write(f"**Risco perda:** {row['stop_pct'] * 100:.2f}%")
        st.write(f"**R:R:** {row['risk_reward']:.2f}")
        st.write(f"**Qtd sugerida:** {int(row['qty'])}")
        st.write(f"**Valor da posição:** R$ {row['position_value']:.2f}")
        st.write(f"**Fortes:** {row['strengths']}")
        st.write(f"**Alertas:** {row['alerts']}")

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
            "rsi",
            "rvol",
            "volatility",
            "relative_strength",
            "strengths",
            "alerts",
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
        "RSI",
        "RVOL",
        "Volatilidade",
        "Força Relativa",
        "Fortes",
        "Alertas",
    ]

    display_df["Risco/Retorno"] = display_df["Risco/Retorno"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
    display_df["Risco %"] = display_df["Risco %"].map(format_pct)
    display_df["Potencial %"] = display_df["Potencial %"].map(format_pct)
    display_df["Volatilidade"] = display_df["Volatilidade"].map(format_pct)
    display_df["Força Relativa"] = display_df["Força Relativa"].map(format_pct)
    display_df["RSI"] = display_df["RSI"].map(lambda x: f"{x:.1f}")
    display_df["RVOL"] = display_df["RVOL"].map(lambda x: f"{x:.2f}")
    display_df["Qtd"] = display_df["Qtd"].map(lambda x: int(x))
    display_df["Valor Posição"] = display_df["Valor Posição"].map(lambda x: f"R$ {x:,.2f}")

    st.dataframe(display_df, use_container_width=True, hide_index=True)

selected_label = st.selectbox(
    "Ativo para ver o gráfico detalhado",
    options=summary["ticker"].tolist(),
    index=0,
)

selected_ticker = normalize_ticker(selected_label)
selected_row = summary.loc[summary["ticker"] == selected_label].iloc[0]
selected_data = data_map[selected_ticker]

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

alerts = check_price_alerts(latest_close, alert_above, alert_below)
if alerts:
    for msg in alerts:
        st.warning(msg)
else:
    st.info("Nenhum alerta manual acionado no ativo selecionado.")

st.subheader("Resumo operacional")

a1, a2, a3, a4 = st.columns(4)
a1.metric("Decisão", selected_row["decision"])
a2.metric("Preço atual", f"R$ {selected_row['close']:.2f}")
a3.metric("Risco / Retorno", f"{selected_row['risk_reward']:.2f}")
a4.metric("Potencial", format_pct(float(selected_row["upside_pct"])))

b1, b2, b3, b4 = st.columns(4)
b1.metric("RSI 14", f"{selected_row['rsi']:.1f}")
b2.metric("RVOL", f"{selected_row['rvol']:.2f}")
b3.metric("Volatilidade 20", format_pct(float(selected_row["volatility"])))
b4.metric("Força Relativa", format_pct(float(selected_row["relative_strength"])) if pd.notna(selected_row["relative_strength"]) else "-")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Qtd sugerida", int(selected_row["qty"]))
c2.metric("Valor da posição", f"R$ {selected_row['position_value']:.2f}")
c3.metric("Risco por ação", f"R$ {selected_row['risk_per_share']:.2f}")
c4.metric("Risco financeiro", f"R$ {selected_row['risk_amount']:.2f}")

with st.expander("Como o robô decide"):
    st.markdown(
        """
        - **OPERAR AGORA**: ativo forte, mercado favorável e risco bem controlado.
        - **OBSERVAR**: ativo interessante, mas ainda sem confirmação completa.
        - **NÃO OPERAR**: mercado ruim, sinal fraco ou risco inadequado.

        O robô bloqueia operação automaticamente quando:
        - o mercado está defensivo,
        - o ativo está abaixo da VWAP,
        - o ativo está abaixo da EMA21,
        - o risco/retorno está abaixo do mínimo,
        - ou a posição calculada fica inviável.
        """
    )

st.caption("Uso educacional. Não é garantia de lucro e não substitui gerenciamento de risco e testes da estratégia.")
