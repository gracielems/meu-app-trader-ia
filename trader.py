import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import datetime

st.set_page_config(
    page_title="Robô Profissional de Ações",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 Robô Profissional de Ações")
st.caption("Painel simples para encontrar as melhores ações com maior potencial e menor risco.")

ATIVOS_PADRAO = [
    "PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "BBAS3.SA",
    "ABEV3.SA", "WEGE3.SA", "SUZB3.SA", "JBSS3.SA", "RENT3.SA",
    "RADL3.SA", "PRIO3.SA", "LREN3.SA", "GGBR4.SA", "USIM5.SA",
    "CMIG4.SA", "CPLE6.SA", "ELET3.SA", "EQTL3.SA", "TOTS3.SA"
]

INDICE_BENCHMARK = "^BVSP"

st.sidebar.header("Configurações")

ativos_texto = st.sidebar.text_area(
    "Ativos monitorados (separados por vírgula)",
    value=", ".join(ATIVOS_PADRAO),
    height=160
)

intervalo = st.sidebar.selectbox(
    "Intervalo intradia",
    options=["5m", "15m", "30m", "1h"],
    index=1
)

periodo_intraday = st.sidebar.selectbox(
    "Período intradia",
    options=["5d", "1mo"],
    index=0
)

periodo_diario = st.sidebar.selectbox(
    "Período diário para tendência maior",
    options=["3mo", "6mo", "1y"],
    index=1
)

top_n = st.sidebar.slider("Quantidade de sugestões", 3, 5, 3)
score_minimo = st.sidebar.slider("Pontuação mínima para o pódio", 60, 90, 75)

ativos = [x.strip().upper() for x in ativos_texto.split(",") if x.strip()]


@st.cache_data(ttl=300, show_spinner=False)
def baixar_dados(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False
    )

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df.dropna().copy()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    vol = df["Volume"].replace(0, np.nan)
    return (tp * vol).cumsum() / vol.cumsum()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["EMA9"] = ema(df["Close"], 9)
    df["EMA21"] = ema(df["Close"], 21)
    df["EMA50"] = ema(df["Close"], 50)
    df["VOL20"] = df["Volume"].rolling(20).mean()
    df["RSI14"] = rsi(df["Close"], 14)
    df["ATR14"] = atr(df, 14)
    df["ATR_PCT"] = (df["ATR14"] / df["Close"]) * 100
    df["VWAP"] = vwap(df)
    df["HH20"] = df["High"].rolling(20).max()
    df["LL20"] = df["Low"].rolling(20).min()
    return df


def trend_quality(df: pd.DataFrame) -> float:
    if len(df) < 10:
        return 0.0

    e9 = df["EMA9"].tail(6)
    e21 = df["EMA21"].tail(6)

    score = 0.0
    if e9.is_monotonic_increasing:
        score += 0.5
    if e21.is_monotonic_increasing:
        score += 0.5

    return score


def relative_strength_score(stock_daily: pd.DataFrame, ibov_daily: pd.DataFrame) -> float:
    if len(stock_daily) < 25 or len(ibov_daily) < 25:
        return 0.0

    stock_ret = stock_daily["Close"].pct_change(20).iloc[-1]
    ibov_ret = ibov_daily["Close"].pct_change(20).iloc[-1]

    rs = stock_ret - ibov_ret
    if pd.isna(rs):
        return 0.0
    return float(rs)


def clamp(v, low, high):
    return max(low, min(high, v))


def classificar_momento(dist_ema21, vol_ratio, preco, vwap_val, hh20_prev):
    if preco > vwap_val and vol_ratio >= 1.3 and 0.3 <= dist_ema21 <= 2.0:
        return "Entrar agora"
    if preco > vwap_val and dist_ema21 > 2.0:
        return "Esperar pullback"
    if preco < hh20_prev * 0.995:
        return "Esperar rompimento"
    return "Aguardar confirmação"


def analisar_ativo(ticker: str, df_i: pd.DataFrame, df_d: pd.DataFrame, df_ibov_d: pd.DataFrame) -> dict:
    if df_i.empty or df_d.empty or len(df_i) < 60 or len(df_d) < 60:
        return {
            "Ativo": ticker.replace(".SA", ""),
            "Score": 0,
            "Confiança": 0,
            "Sinal": "ESPERAR",
            "Momento": "Sem dados",
            "Preço": None,
            "Entrada": None,
            "Stop": None,
            "Alvo": None,
            "R:R": None,
            "Ganho %": None,
            "Perda %": None,
            "Risco": "ALTO",
            "Volume x Média": None,
            "ATR %": None,
            "RS vs IBOV": None,
            "Resumo": "Dados insuficientes"
        }

    i = add_indicators(df_i)
    d = add_indicators(df_d)

    last_i = i.iloc[-1]
    last_d = d.iloc[-1]

    preco = float(last_i["Close"])
    vwap_val = float(last_i["VWAP"]) if not pd.isna(last_i["VWAP"]) else preco
    atr_atual = float(last_i["ATR14"]) if not pd.isna(last_i["ATR14"]) else 0.0
    atr_pct = float(last_i["ATR_PCT"]) if not pd.isna(last_i["ATR_PCT"]) else 999.0
    vol_ratio = float(last_i["Volume"] / last_i["VOL20"]) if (not pd.isna(last_i["VOL20"]) and last_i["VOL20"] > 0) else 0.0
    rs_score = relative_strength_score(d, df_ibov_d)
    tq = trend_quality(i)

    score = 0
    fortes = []
    alertas = []

    # Tendência diária
    if last_d["Close"] > last_d["EMA21"]:
        score += 12
        fortes.append("acima da EMA21 diária")
    else:
        score -= 14
        alertas.append("abaixo da EMA21 diária")

    if last_d["EMA9"] > last_d["EMA21"]:
        score += 12
        fortes.append("EMA9 diária acima da EMA21")
    else:
        score -= 10
        alertas.append("tendência diária fraca")

    # Tendência intradia
    if last_i["Close"] > last_i["VWAP"]:
        score += 12
        fortes.append("acima da VWAP")
    else:
        score -= 10
        alertas.append("abaixo da VWAP")

    if last_i["EMA9"] > last_i["EMA21"]:
        score += 10
        fortes.append("EMA9 intradia acima da EMA21")
    else:
        score -= 10
        alertas.append("intradia fraco")

    # Força relativa
    if rs_score > 0.03:
        score += 16
        fortes.append("bem mais forte que o Ibovespa")
    elif rs_score > 0.01:
        score += 10
        fortes.append("mais forte que o Ibovespa")
    elif rs_score > 0:
        score += 5
        fortes.append("ligeiramente mais forte que o Ibovespa")
    else:
        score -= 10
        alertas.append("mais fraca que o Ibovespa")

    # Volume mais rígido
    if vol_ratio >= 1.5:
        score += 15
        fortes.append("volume muito forte")
    elif vol_ratio >= 1.2:
        score += 10
        fortes.append("volume forte")
    elif vol_ratio >= 1.0:
        score += 4
        fortes.append("volume aceitável")
    elif vol_ratio >= 0.8:
        score -= 6
        alertas.append("volume abaixo da média")
    else:
        score -= 16
        alertas.append("volume fraco demais")

    # Qualidade da tendência
    if tq >= 1.0:
        score += 8
        fortes.append("tendência limpa")
    elif tq >= 0.5:
        score += 4
        fortes.append("boa inclinação")
    else:
        score -= 5
        alertas.append("inclinação fraca")

    # RSI
    if not pd.isna(last_i["RSI14"]):
        if 53 <= last_i["RSI14"] <= 67:
            score += 8
            fortes.append("RSI saudável")
        elif 68 < last_i["RSI14"] <= 75:
            score += 2
            fortes.append("momentum forte")
        elif last_i["RSI14"] > 75:
            score -= 10
            alertas.append("muito esticada")
        elif last_i["RSI14"] < 45:
            score -= 8
            alertas.append("momentum fraco")

    # Rompimento
    hh20_prev = float(i["HH20"].iloc[-2]) if len(i) >= 21 and not pd.isna(i["HH20"].iloc[-2]) else preco
    if preco >= hh20_prev and vol_ratio >= 1.2:
        score += 10
        fortes.append("rompimento com volume")
    elif preco >= hh20_prev * 0.995:
        score += 4
        fortes.append("perto do rompimento")

    # Volatilidade
    if 1.0 <= atr_pct <= 3.2:
        score += 10
        fortes.append("volatilidade controlada")
    elif 0.7 <= atr_pct < 1.0:
        score += 3
        fortes.append("volatilidade baixa")
    elif atr_pct < 0.7:
        score -= 8
        alertas.append("andar curto")
    else:
        score -= 12
        alertas.append("volatilidade alta")

    # Liquidez
    valor_financeiro = float(last_i["Close"] * last_i["Volume"])
    if valor_financeiro >= 8_000_000:
        score += 6
        fortes.append("boa liquidez")
    elif valor_financeiro < 2_000_000:
        score -= 12
        alertas.append("liquidez fraca")

    # Distância da EMA21
    dist_ema21 = ((preco / float(last_i["EMA21"])) - 1) * 100 if last_i["EMA21"] else 0.0
    if 0.3 <= dist_ema21 <= 2.0:
        score += 8
        fortes.append("entrada equilibrada")
    elif 2.0 < dist_ema21 <= 3.5:
        score += 2
        alertas.append("um pouco esticada")
    elif dist_ema21 > 3.5:
        score -= 12
        alertas.append("esticada demais")

    # Stop e alvo
    stop = round(preco - 1.2 * atr_atual, 2) if atr_atual > 0 else None
    alvo = round(preco + 2.4 * atr_atual, 2) if atr_atual > 0 else None
    entrada = round(preco, 2)

    rr = None
    ganho_pct = None
    perda_pct = None

    if stop and alvo and preco > stop:
        risco_abs = preco - stop
        retorno_abs = alvo - preco
        rr = round(retorno_abs / risco_abs, 2) if risco_abs > 0 else None
        ganho_pct = round(((alvo / preco) - 1) * 100, 2)
        perda_pct = round(((stop / preco) - 1) * 100, 2)

        if rr >= 2.0:
            score += 10
            fortes.append("risco-retorno excelente")
        elif rr >= 1.7:
            score += 2
            alertas.append("risco-retorno apenas aceitável")
        else:
            score -= 14
            alertas.append("risco-retorno fraco")

    # Risco final
    if atr_pct <= 2.2 and vol_ratio >= 1.1 and preco > vwap_val:
        risco_txt = "BAIXO"
    elif atr_pct <= 3.2 and vol_ratio >= 0.9:
        risco_txt = "MÉDIO"
    else:
        risco_txt = "ALTO"

    score = int(clamp(score, 0, 100))
    confianca = score

    momento = classificar_momento(
        dist_ema21=dist_ema21,
        vol_ratio=vol_ratio,
        preco=preco,
        vwap_val=vwap_val,
        hh20_prev=hh20_prev
    )

    # Sinal final mais seletivo
    if (
        score >= 82
        and risco_txt == "BAIXO"
        and vol_ratio >= 1.1
        and rr is not None and rr >= 2.0
        and preco > vwap_val
        and last_i["EMA9"] > last_i["EMA21"]
        and last_d["EMA9"] > last_d["EMA21"]
    ):
        sinal = "COMPRA FORTE"
    elif (
        score >= 68
        and risco_txt in ["BAIXO", "MÉDIO"]
        and rr is not None and rr >= 1.7
    ):
        sinal = "OBSERVAR"
    else:
        sinal = "ESPERAR"

    resumo_fortes = ", ".join(fortes[:4]) if fortes else "sem pontos fortes claros"
    resumo_alertas = ", ".join(alertas[:3]) if alertas else "sem alertas relevantes"
    resumo = f"Fortes: {resumo_fortes}. Alertas: {resumo_alertas}."

    return {
        "Ativo": ticker.replace(".SA", ""),
        "Score": score,
        "Confiança": confianca,
        "Sinal": sinal,
        "Momento": momento,
        "Preço": round(preco, 2),
        "Entrada": entrada,
        "Stop": stop,
        "Alvo": alvo,
        "R:R": rr,
        "Ganho %": ganho_pct,
        "Perda %": perda_pct,
        "Risco": risco_txt,
        "Volume x Média": round(vol_ratio, 2),
        "ATR %": round(atr_pct, 2),
        "RS vs IBOV": round(rs_score * 100, 2),
        "Resumo": resumo
    }


def cor_sinal(sinal: str) -> str:
    if sinal == "COMPRA FORTE":
        return "🟢"
    if sinal == "OBSERVAR":
        return "🟡"
    return "⚪"


with st.spinner("Analisando mercado..."):
    ibov_d = baixar_dados(INDICE_BENCHMARK, periodo_diario, "1d")
    resultados = []
    progresso = st.progress(0)

    total = max(len(ativos), 1)

    for idx, ativo in enumerate(ativos, start=1):
        try:
            df_intraday = baixar_dados(ativo, periodo_intraday, intervalo)
            df_daily = baixar_dados(ativo, periodo_diario, "1d")
            resultado = analisar_ativo(ativo, df_intraday, df_daily, ibov_d)
            resultados.append(resultado)
        except Exception as e:
            resultados.append({
                "Ativo": ativo.replace(".SA", ""),
                "Score": 0,
                "Confiança": 0,
                "Sinal": "ESPERAR",
                "Momento": "Erro",
                "Preço": None,
                "Entrada": None,
                "Stop": None,
                "Alvo": None,
                "R:R": None,
                "Ganho %": None,
                "Perda %": None,
                "Risco": "ALTO",
                "Volume x Média": None,
                "ATR %": None,
                "RS vs IBOV": None,
                "Resumo": f"Erro na análise: {e}"
            })

        progresso.progress(idx / total)

df_res = pd.DataFrame(resultados)

if not df_res.empty:
    df_res = df_res.sort_values(
        by=["Score", "Confiança", "Volume x Média"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

sugestoes = df_res[
    (df_res["Score"] >= score_minimo) &
    (df_res["Sinal"].isin(["COMPRA FORTE", "OBSERVAR"])) &
    (df_res["Risco"] != "ALTO")
].head(top_n)

st.subheader("🔥 Top oportunidades")

if sugestoes.empty:
    st.warning("Nenhuma ação passou no filtro hoje. Isso ajuda a evitar entrada ruim.")
else:
    cols = st.columns(len(sugestoes))
    for col, (_, row) in zip(cols, sugestoes.iterrows()):
        with col:
            st.metric(
                label=f"{cor_sinal(row['Sinal'])} {row['Ativo']}",
                value=f"Score {row['Score']}",
                delta=f"Confiança {row['Confiança']}%"
            )
            st.write(f"**Sinal:** {row['Sinal']}")
            st.write(f"**Momento:** {row['Momento']}")
            st.write(f"**Preço atual:** R$ {row['Preço']}")
            st.write(f"**Entrada:** R$ {row['Entrada']}")
            st.write(f"**Stop:** R$ {row['Stop']}")
            st.write(f"**Alvo:** R$ {row['Alvo']}")
            st.write(f"**Potencial ganho:** {row['Ganho %']}%")
            st.write(f"**Risco perda:** {row['Perda %']}%")
            st.write(f"**R:R:** {row['R:R']}")
            st.write(f"**Risco:** {row['Risco']}")
            st.caption(row["Resumo"])

st.divider()

with st.expander("📋 Ver ranking completo"):
    mostrar = df_res[[
        "Ativo", "Score", "Confiança", "Sinal", "Momento", "Preço", "Entrada",
        "Stop", "Alvo", "Ganho %", "Perda %", "R:R", "Risco",
        "Volume x Média", "ATR %", "RS vs IBOV"
    ]]
    st.dataframe(mostrar, use_container_width=True)

st.divider()

st.subheader("📈 Gráfico do melhor ativo")

if not df_res.empty and pd.notna(df_res.iloc[0]["Ativo"]):
    melhor = df_res.iloc[0]["Ativo"] + ".SA"
    melhor_df = baixar_dados(melhor, periodo_intraday, intervalo)

    if not melhor_df.empty:
        melhor_df = add_indicators(melhor_df)
        chart_df = melhor_df[["Close", "EMA9", "EMA21", "VWAP"]].dropna()
        st.line_chart(chart_df, use_container_width=True)

st.caption(f"Última atualização: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
st.caption("Uso recomendado: apoio à decisão e simulação. Não é garantia de lucro.")
