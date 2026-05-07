"""
Robô Elite B3 — Backtesting
Estratégia: RSI (14) + EMA 9/21
Fonte de dados: yfinance (histórico gratuito e longo)
"""

import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, date
from typing import Dict, List, Tuple
import math

st.set_page_config(page_title="Backtesting B3", layout="wide", page_icon="🔬")


# ---------------------------------------------------------------------------
# PARÂMETROS
# ---------------------------------------------------------------------------
class Parametros:
    RSI_PERIODO     = 14
    RSI_SOBREVENDA  = 30
    RSI_SOBRECOMPRA = 70
    EMA_RAPIDA      = 9
    EMA_LENTA       = 21
    MIN_PERIODOS    = 30


# ---------------------------------------------------------------------------
# INDICADORES (pandas puro)
# ---------------------------------------------------------------------------
def calcular_rsi(serie: pd.Series, periodo: int = 14) -> pd.Series:
    delta       = serie.diff()
    ganho       = delta.clip(lower=0)
    perda       = (-delta).clip(lower=0)
    media_ganho = ganho.ewm(alpha=1 / periodo, min_periods=periodo, adjust=False).mean()
    media_perda = perda.ewm(alpha=1 / periodo, min_periods=periodo, adjust=False).mean()
    rs          = media_ganho / media_perda.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def calcular_ema(serie: pd.Series, periodo: int) -> pd.Series:
    return serie.ewm(span=periodo, min_periods=periodo, adjust=False).mean()


def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rsi"]        = calcular_rsi(df["Close"], Parametros.RSI_PERIODO)
    df["ema_rapida"] = calcular_ema(df["Close"], Parametros.EMA_RAPIDA)
    df["ema_lenta"]  = calcular_ema(df["Close"], Parametros.EMA_LENTA)
    return df.dropna(subset=["rsi", "ema_rapida", "ema_lenta"]).reset_index()


# ---------------------------------------------------------------------------
# GERAÇÃO DE SINAIS
# ---------------------------------------------------------------------------
def gerar_sinais(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona coluna 'sinal': 1 = compra, -1 = venda, 0 = neutro."""
    df = df.copy()
    df["sinal"] = 0

    for i in range(1, len(df)):
        rsi = df.at[i, "rsi"]
        er  = df.at[i, "ema_rapida"]
        el  = df.at[i, "ema_lenta"]
        er_ant = df.at[i - 1, "ema_rapida"]
        el_ant = df.at[i - 1, "ema_lenta"]

        cruz_alta  = er > el  and er_ant <= el_ant
        cruz_baixa = er < el  and er_ant >= el_ant

        if rsi < Parametros.RSI_SOBREVENDA or cruz_alta:
            df.at[i, "sinal"] = 1
        elif rsi > Parametros.RSI_SOBRECOMPRA or cruz_baixa:
            df.at[i, "sinal"] = -1

    return df


# ---------------------------------------------------------------------------
# SIMULAÇÃO DE TRADES
# ---------------------------------------------------------------------------
def simular_trades(
    df: pd.DataFrame,
    capital_inicial: float,
    risco_pct: float,
    stop_loss_pct: float,
    take_profit_pct: float,
) -> Tuple[List[Dict], pd.Series]:
    """Simula operações de compra e venda com stop loss e take profit.

    Args:
        df:              DataFrame com sinais gerados.
        capital_inicial: Capital em reais.
        risco_pct:       % do capital arriscado por trade (ex: 0.01 = 1%).
        stop_loss_pct:   % de queda para acionar stop (ex: 0.015 = 1.5%).
        take_profit_pct: % de alta para realizar lucro (ex: 0.03 = 3%).

    Returns:
        Lista de trades executados e série de evolução do capital.
    """
    trades: List[Dict] = []
    capital   = capital_inicial
    capital_historico = [capital]

    em_posicao      = False
    preco_entrada   = 0.0
    qtd_acoes       = 0
    stop_loss       = 0.0
    take_profit     = 0.0
    data_entrada    = None
    motivo_entrada  = ""

    for i in range(len(df)):
        row   = df.iloc[i]
        preco = row["Close"]
        data  = row["Date"] if "Date" in df.columns else row.get("Datetime", row.name)

        if em_posicao:
            # Verifica stop loss e take profit
            fechou, motivo_saida = False, ""
            if preco <= stop_loss:
                fechou, motivo_saida = True, "Stop loss"
            elif preco >= take_profit:
                fechou, motivo_saida = True, "Take profit"
            elif row["sinal"] == -1:
                fechou, motivo_saida = True, "Sinal de venda"

            if fechou:
                resultado    = (preco - preco_entrada) * qtd_acoes
                capital     += resultado
                retorno_pct  = (preco - preco_entrada) / preco_entrada * 100

                trades.append({
                    "entrada":      data_entrada,
                    "saida":        data,
                    "preco_compra": round(preco_entrada, 2),
                    "preco_venda":  round(preco, 2),
                    "qtd":          qtd_acoes,
                    "resultado":    round(resultado, 2),
                    "retorno_pct":  round(retorno_pct, 2),
                    "motivo_saida": motivo_saida,
                    "capital_apos": round(capital, 2),
                })
                capital_historico.append(capital)
                em_posicao = False

        elif row["sinal"] == 1 and not em_posicao:
            # Calcula tamanho da posição baseado no risco
            risco_reais  = capital * risco_pct
            distancia    = preco * stop_loss_pct
            qtd          = max(1, math.floor(risco_reais / distancia))

            # Garante que o capital é suficiente
            custo_total  = qtd * preco
            if custo_total > capital:
                qtd = max(1, math.floor(capital / preco))

            preco_entrada  = preco
            qtd_acoes      = qtd
            stop_loss      = preco * (1 - stop_loss_pct)
            take_profit    = preco * (1 + take_profit_pct)
            data_entrada   = data
            motivo_entrada = "RSI sobrevendido" if row["rsi"] < Parametros.RSI_SOBREVENDA else "Cruzamento EMA"
            em_posicao     = True

    return trades, pd.Series(capital_historico)


# ---------------------------------------------------------------------------
# MÉTRICAS
# ---------------------------------------------------------------------------
def calcular_metricas(trades: List[Dict], capital_inicial: float, capital_final: float) -> Dict:
    if not trades:
        return {}

    resultados   = [t["resultado"] for t in trades]
    ganhos       = [r for r in resultados if r > 0]
    perdas       = [r for r in resultados if r <= 0]
    total        = len(trades)

    win_rate     = len(ganhos) / total * 100 if total else 0
    lucro_total  = sum(resultados)
    retorno_total = (capital_final - capital_inicial) / capital_inicial * 100
    media_ganho  = sum(ganhos) / len(ganhos) if ganhos else 0
    media_perda  = sum(perdas) / len(perdas) if perdas else 0
    payoff       = abs(media_ganho / media_perda) if media_perda else 0

    # Drawdown máximo
    capital_series  = pd.Series([capital_inicial] + [t["capital_apos"] for t in trades])
    pico            = capital_series.cummax()
    drawdown_series = (capital_series - pico) / pico * 100
    max_drawdown    = drawdown_series.min()

    # Sequência máxima de perdas
    seq_perdas, seq_max = 0, 0
    for r in resultados:
        if r <= 0:
            seq_perdas += 1
            seq_max = max(seq_max, seq_perdas)
        else:
            seq_perdas = 0

    return {
        "total_trades":   total,
        "win_rate":       round(win_rate, 1),
        "lucro_total":    round(lucro_total, 2),
        "retorno_total":  round(retorno_total, 2),
        "media_ganho":    round(media_ganho, 2),
        "media_perda":    round(media_perda, 2),
        "payoff":         round(payoff, 2),
        "max_drawdown":   round(max_drawdown, 2),
        "seq_max_perdas": seq_max,
        "trades_ganho":   len(ganhos),
        "trades_perda":   len(perdas),
    }


# ---------------------------------------------------------------------------
# INTERFACE
# ---------------------------------------------------------------------------
def cor_resultado(valor: float) -> str:
    return "🟢" if valor > 0 else "🔴"


def main() -> None:
    st.title("🔬 Backtesting — Robô Elite B3")
    st.caption("Estratégia: RSI (14) + EMA 9/21 | Dados: yfinance")

    # ── Sidebar de configuração ───────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Parâmetros")

        ticker         = st.text_input("Código da ação", value="BBAS3")
        data_inicio    = st.date_input("Início", value=date(2022, 1, 1))
        data_fim       = st.date_input("Fim",    value=date.today())
        capital        = st.number_input("Capital inicial (R$)", value=10000, step=1000, min_value=1000)
        risco_pct      = st.slider("Risco por trade (%)", 0.5, 5.0, 1.0, 0.5) / 100
        stop_loss_pct  = st.slider("Stop loss (%)",        0.5, 5.0, 1.5, 0.5) / 100
        take_profit_pct = st.slider("Take profit (%)",     0.5, 10.0, 3.0, 0.5) / 100

        st.divider()
        st.caption(f"Relação risco/retorno: 1 : {take_profit_pct / stop_loss_pct:.1f}")

        rodar = st.button("▶️ Rodar backtesting", type="primary", use_container_width=True)

    if not rodar:
        st.info("Configure os parâmetros na barra lateral e clique em **▶️ Rodar backtesting**.")
        return

    # ── Download de dados ─────────────────────────────────────────────────
    t = ticker.strip().upper()
    if not t.endswith(".SA"):
        t = f"{t}.SA"

    with st.spinner(f"Baixando histórico de {t} via yfinance…"):
        df_raw = yf.download(t, start=data_inicio, end=data_fim,
                             interval="1d", progress=False, auto_adjust=True)

    if df_raw.empty:
        st.error("Nenhum dado retornado. Verifique o ticker e o período.")
        return

    # Flatten MultiIndex columns if present
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)

    df_raw = df_raw.reset_index()

    st.success(f"{len(df_raw)} dias de histórico carregados ({data_inicio} → {data_fim})")

    # ── Cálculo de indicadores e sinais ───────────────────────────────────
    df = calcular_indicadores(df_raw)
    df = gerar_sinais(df)

    # ── Simulação ─────────────────────────────────────────────────────────
    trades, capital_serie = simular_trades(
        df, float(capital), risco_pct, stop_loss_pct, take_profit_pct
    )

    if not trades:
        st.warning("Nenhuma operação gerada no período. Tente ampliar o intervalo ou ajustar os parâmetros.")
        return

    capital_final = trades[-1]["capital_apos"]
    metricas      = calcular_metricas(trades, float(capital), capital_final)

    # ── Métricas principais ───────────────────────────────────────────────
    st.divider()
    st.subheader("📊 Resultado geral")

    c1, c2, c3, c4 = st.columns(4)
    retorno = metricas["retorno_total"]
    c1.metric("Retorno total",
              f"R$ {metricas['lucro_total']:,.2f}",
              f"{retorno:+.1f}%")
    c2.metric("Capital final",   f"R$ {capital_final:,.2f}")
    c3.metric("Win rate",        f"{metricas['win_rate']}%",
              f"{metricas['trades_ganho']}G / {metricas['trades_perda']}P")
    c4.metric("Drawdown máximo", f"{metricas['max_drawdown']:.1f}%")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Total de trades",     metricas["total_trades"])
    c6.metric("Payoff (ganho/perda)", f"{metricas['payoff']:.2f}x")
    c7.metric("Média por ganho",     f"R$ {metricas['media_ganho']:,.2f}")
    c8.metric("Seq. máx. de perdas", metricas["seq_max_perdas"])

    # ── Avaliação da estratégia ───────────────────────────────────────────
    st.divider()
    st.subheader("🧠 Avaliação da estratégia")

    pontos = []
    alertas = []

    if metricas["retorno_total"] > 0:
        pontos.append(f"✅ Estratégia lucrativa no período (+{metricas['retorno_total']}%)")
    else:
        alertas.append(f"❌ Estratégia com prejuízo no período ({metricas['retorno_total']}%)")

    if metricas["win_rate"] >= 50:
        pontos.append(f"✅ Win rate acima de 50% ({metricas['win_rate']}%)")
    else:
        alertas.append(f"⚠️ Win rate abaixo de 50% ({metricas['win_rate']}%) — precisa de payoff alto para compensar")

    if metricas["payoff"] >= 1.5:
        pontos.append(f"✅ Payoff saudável ({metricas['payoff']}x)")
    else:
        alertas.append(f"⚠️ Payoff baixo ({metricas['payoff']}x) — ganhos médios não superam perdas médias adequadamente")

    if metricas["max_drawdown"] > -20:
        pontos.append(f"✅ Drawdown controlado ({metricas['max_drawdown']}%)")
    else:
        alertas.append(f"⚠️ Drawdown alto ({metricas['max_drawdown']}%) — risco de perda psicológica de controle")

    if metricas["seq_max_perdas"] <= 5:
        pontos.append(f"✅ Sequência máx. de perdas aceitável ({metricas['seq_max_perdas']} seguidas)")
    else:
        alertas.append(f"⚠️ {metricas['seq_max_perdas']} perdas seguidas no pior momento — avalie sua resiliência emocional")

    for p in pontos:
        st.success(p)
    for a in alertas:
        st.warning(a)

    # ── Gráfico de evolução do capital ────────────────────────────────────
    st.divider()
    st.subheader("📈 Evolução do capital")

    datas_trades = [float(capital)] + [t["capital_apos"] for t in trades]
    st.line_chart(pd.DataFrame({"Capital (R$)": datas_trades}))

    # ── Tabela de trades ──────────────────────────────────────────────────
    st.divider()
    st.subheader("📋 Histórico de operações")

    df_trades = pd.DataFrame(trades)
    df_trades["resultado_fmt"] = df_trades["resultado"].apply(
        lambda x: f"{cor_resultado(x)} R$ {x:,.2f}"
    )

    st.dataframe(
        df_trades[[
            "entrada", "saida", "preco_compra", "preco_venda",
            "qtd", "resultado_fmt", "retorno_pct", "motivo_saida"
        ]].rename(columns={
            "entrada":      "Entrada",
            "saida":        "Saída",
            "preco_compra": "Preço compra",
            "preco_venda":  "Preço venda",
            "qtd":          "Qtd",
            "resultado_fmt":"Resultado",
            "retorno_pct":  "Retorno %",
            "motivo_saida": "Motivo saída",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # Download CSV
    csv = df_trades.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Baixar histórico de trades (.csv)",
        data=csv,
        file_name=f"backtest_{t}_{data_inicio}_{data_fim}.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
