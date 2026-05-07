"""
Robô Elite B3 — Monitor Contínuo de Pregão
Indicadores: RSI (14) + EMA 9/21
Infraestrutura: Loop de monitoramento + Log persistente + Alertas Telegram
"""

import streamlit as st
import pandas as pd
import requests
import logging
import csv
import time
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Robô Elite B3", layout="wide", page_icon="📈")


# ---------------------------------------------------------------------------
# CAMINHOS DE LOG
# ---------------------------------------------------------------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

ARQUIVO_LOG    = LOG_DIR / "robo_b3.log"
ARQUIVO_SINAIS = LOG_DIR / "sinais.csv"
ARQUIVO_ORDENS = LOG_DIR / "ordens.csv"


# ---------------------------------------------------------------------------
# SISTEMA DE LOG
# ---------------------------------------------------------------------------
def configurar_logger() -> logging.Logger:
    """Configura logger com saída simultânea em arquivo e console."""
    logger = logging.getLogger("robo_b3")
    if logger.handlers:
        return logger  # já configurado em execuções anteriores do Streamlit

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(ARQUIVO_LOG, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


log = configurar_logger()


# ---------------------------------------------------------------------------
# TELEGRAM — ALERTAS E COMANDOS
# ---------------------------------------------------------------------------
# Emojis por tipo de sinal para deixar as mensagens mais legíveis no celular
_EMOJI_SINAL = {
    "COMPRA FORTE": "🚀",
    "COMPRA":       "🟢",
    "VENDA FORTE":  "🔥",
    "VENDA":        "🔴",
    "NEUTRO":       "⚪",
}


def enviar_telegram(mensagem: str, token: str, chat_id: str) -> bool:
    """Envia uma mensagem de texto via Telegram Bot API.

    Args:
        mensagem: Texto a enviar (suporta Markdown v2 básico).
        token:    Token do bot do Telegram.
        chat_id:  ID do chat/usuário de destino.

    Returns:
        True se enviado com sucesso, False caso contrário.
    """
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id":    chat_id,
        "text":       mensagem,
        "parse_mode": "HTML",   # <b>, <i>, <code> funcionam bem no Telegram
    }
    try:
        resp = requests.post(url, json=payload, timeout=8)
        resp.raise_for_status()
        log.info(f"Telegram OK — {mensagem[:60]}…")
        return True
    except requests.exceptions.Timeout:
        log.warning("Telegram: timeout ao enviar mensagem")
    except requests.exceptions.HTTPError as e:
        log.error(f"Telegram HTTP error: {e} | resp: {resp.text[:200]}")
    except requests.exceptions.RequestException as e:
        log.error(f"Telegram erro de rede: {e}")
    return False


def montar_alerta_sinal(ticker: str, sinal: Dict, modo_sim: bool) -> str:
    """Monta a mensagem de alerta formatada para o Telegram.

    Args:
        ticker:   Código da ação normalizado.
        sinal:    Dicionário com tipo, força, preço, rsi e motivo.
        modo_sim: Se True, adiciona tag de simulação na mensagem.
    """
    tipo      = sinal.get("tipo", "NEUTRO")
    emoji     = _EMOJI_SINAL.get(tipo, "📊")
    forca     = sinal.get("forca", 0)
    preco     = sinal.get("preco", 0)
    rsi       = sinal.get("rsi", 0)
    motivo    = sinal.get("motivo", "")
    agora     = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    modo_tag  = "🧪 <i>SIMULAÇÃO</i>\n" if modo_sim else "💰 <b>MODO REAL</b>\n"

    return (
        f"{modo_tag}"
        f"{emoji} <b>SINAL: {tipo}</b>\n\n"
        f"📌 <b>Ação:</b> <code>{ticker}</code>\n"
        f"💵 <b>Preço:</b> R$ {preco:.2f}\n"
        f"📊 <b>RSI:</b> {rsi:.1f}\n"
        f"💪 <b>Força:</b> {forca}%\n"
        f"📝 <b>Motivo:</b> {motivo}\n\n"
        f"🕐 {agora}"
    )


def montar_alerta_ordem(tipo: str, ticker: str, preco: float,
                        qtd: int, motivo: str, modo_sim: bool) -> str:
    """Monta a mensagem de confirmação de ordem para o Telegram."""
    emoji    = "🟢" if "COMPRA" in tipo else "🔴"
    modo_tag = "🧪 <i>SIMULAÇÃO</i>\n" if modo_sim else "💰 <b>MODO REAL</b>\n"
    total    = preco * qtd if qtd else 0
    agora    = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    return (
        f"{modo_tag}"
        f"{emoji} <b>ORDEM: {tipo}</b>\n\n"
        f"📌 <b>Ação:</b> <code>{ticker}</code>\n"
        f"💵 <b>Preço:</b> R$ {preco:.2f}\n"
        f"📦 <b>Qtd:</b> {qtd} ações\n"
        f"💰 <b>Total:</b> R$ {total:.2f}\n"
        f"📝 <b>Motivo:</b> {motivo}\n\n"
        f"🕐 {agora}"
    )


def montar_alerta_status(ticker: str, ciclo: int, sinal: Dict) -> str:
    """Monta relatório de status enviado a cada N ciclos ou sob demanda."""
    tipo  = sinal.get("tipo", "NEUTRO")
    preco = sinal.get("preco", 0)
    rsi   = sinal.get("rsi", 0)
    agora = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    return (
        f"📡 <b>Status do Robô</b>\n\n"
        f"📌 <b>Ação:</b> <code>{ticker}</code>\n"
        f"💵 <b>Preço:</b> R$ {preco:.2f}\n"
        f"📊 <b>RSI:</b> {rsi:.1f}\n"
        f"🔁 <b>Ciclos:</b> {ciclo}\n"
        f"📶 <b>Último sinal:</b> {tipo}\n\n"
        f"🕐 {agora}"
    )


def registrar_sinal(ticker: str, sinal: Dict) -> None:
    """Persiste um sinal detectado no CSV de sinais.

    Args:
        ticker: Código da ação.
        sinal:  Dicionário com tipo, força, motivo e preço.
    """
    cabecalho = not ARQUIVO_SINAIS.exists()
    with open(ARQUIVO_SINAIS, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if cabecalho:
            writer.writerow(["timestamp", "ticker", "tipo", "forca", "preco", "rsi", "motivo"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ticker,
            sinal.get("tipo", ""),
            sinal.get("forca", 0),
            sinal.get("preco", 0),
            f"{sinal.get('rsi', 0):.2f}" if sinal.get("rsi") else "",
            sinal.get("motivo", ""),
        ])
    log.info(
        f"SINAL | {ticker} | {sinal.get('tipo')} | "
        f"preco={sinal.get('preco')} | RSI={sinal.get('rsi', 0):.1f} | {sinal.get('motivo')}"
    )


def registrar_ordem(tipo: str, ticker: str, preco: float,
                    qtd: int, motivo: str, simulacao: bool = True) -> None:
    """Persiste uma ordem (real ou simulada) no CSV de ordens.

    Args:
        tipo:      'COMPRA' ou 'VENDA'.
        ticker:    Código da ação.
        preco:     Preço de execução.
        qtd:       Quantidade de ações.
        motivo:    Descrição do gatilho.
        simulacao: True = paper trade.
    """
    modo = "SIM" if simulacao else "REAL"
    cabecalho = not ARQUIVO_ORDENS.exists()
    with open(ARQUIVO_ORDENS, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if cabecalho:
            writer.writerow(["timestamp", "modo", "tipo", "ticker",
                             "preco", "qtd", "total", "motivo"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            modo, tipo, ticker,
            f"{preco:.2f}", qtd, f"{preco * qtd:.2f}", motivo,
        ])
    log.info(f"ORDEM [{modo}] | {tipo} {qtd}x {ticker} @ R$ {preco:.2f} | {motivo}")


# ---------------------------------------------------------------------------
# HORÁRIO DE PREGÃO — B3: 10:00 – 17:30 (horário de Brasília)
# ---------------------------------------------------------------------------
ABERTURA_PREGAO   = dtime(10, 0)
FECHAMENTO_PREGAO = dtime(17, 30)


def dentro_do_pregao() -> bool:
    """Retorna True se o horário atual estiver dentro do pregão da B3."""
    agora = datetime.now().time()
    return ABERTURA_PREGAO <= agora <= FECHAMENTO_PREGAO


def status_pregao() -> Dict[str, str]:
    """Retorna ícone e texto descrevendo o estado atual do pregão."""
    agora = datetime.now().time()
    if agora < ABERTURA_PREGAO:
        return {"icone": "🕙", "texto": f"Abre às {ABERTURA_PREGAO.strftime('%H:%M')}"}
    if agora > FECHAMENTO_PREGAO:
        return {"icone": "🔒", "texto": "Pregão encerrado"}
    return {"icone": "🟢", "texto": "Pregão aberto"}


# ---------------------------------------------------------------------------
# PARÂMETROS DA ESTRATÉGIA
# ---------------------------------------------------------------------------
class Parametros:
    RSI_PERIODO      = 14
    RSI_SOBRECOMPRA  = 70
    RSI_SOBREVENDA   = 30
    EMA_RAPIDA       = 9
    EMA_LENTA        = 21
    MIN_PERIODOS     = 30
    INTERVALO_LOOP   = 60   # segundos


# ---------------------------------------------------------------------------
# CONFIGURAÇÕES
# ---------------------------------------------------------------------------
def carregar_config() -> dict:
    """Retorna as credenciais lidas de st.secrets."""
    try:
        return {
            "token_telegram": st.secrets["TOKEN_TELEGRAM"],
            "id_telegram":    st.secrets["ID_TELEGRAM"],
            "token_brapi":    st.secrets["TOKEN_BRAPI"],
        }
    except KeyError as e:
        st.error(f"Chave ausente em st.secrets: {e}. Verifique .streamlit/secrets.toml.")
        st.stop()


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
def normalizar_ticker(ticker: str) -> str:
    t = ticker.strip().upper()
    if not t.endswith(".SA"):
        t = f"{t}.SA"
    return t


def buscar_dados(ticker: str, token_brapi: str) -> pd.DataFrame:
    """Busca histórico de 5d / 5min via brapi.dev."""
    t   = normalizar_ticker(ticker)
    url = f"https://brapi.dev/api/quote/{t}?range=5d&interval=5m&token={token_brapi}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data    = resp.json()
        results = data.get("results", [])
        if not results or "historicalData" not in results[0]:
            return pd.DataFrame()
        df = pd.DataFrame(results[0]["historicalData"])
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], unit="s")
        return df
    except requests.exceptions.Timeout:
        log.warning(f"Timeout ao buscar {t}")
    except requests.exceptions.HTTPError as e:
        log.error(f"HTTP error {t}: {e}")
    except requests.exceptions.RequestException as e:
        log.error(f"Rede {t}: {e}")
    except (KeyError, IndexError, ValueError) as e:
        log.error(f"Parse {t}: {e}")
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# INDICADORES TÉCNICOS — implementados com pandas puro (sem dependências extras)
# ---------------------------------------------------------------------------
def calcular_rsi(serie: pd.Series, periodo: int = 14) -> pd.Series:
    """Calcula o RSI (Relative Strength Index) usando pandas.

    Args:
        serie:   Série de preços de fechamento.
        periodo: Número de períodos (padrão 14).

    Returns:
        Série com valores de RSI entre 0 e 100.
    """
    delta  = serie.diff()
    ganho  = delta.clip(lower=0)
    perda  = -delta.clip(upper=0)
    media_ganho = ganho.ewm(com=periodo - 1, min_periods=periodo).mean()
    media_perda = perda.ewm(com=periodo - 1, min_periods=periodo).mean()
    rs  = media_ganho / media_perda.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calcular_ema(serie: pd.Series, periodo: int) -> pd.Series:
    """Calcula a Média Móvel Exponencial (EMA) usando pandas.

    Args:
        serie:   Série de preços de fechamento.
        periodo: Número de períodos.

    Returns:
        Série com valores da EMA.
    """
    return serie.ewm(span=periodo, adjust=False).mean()


def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona RSI e EMAs ao DataFrame.

    Args:
        df: DataFrame com pelo menos a coluna 'close'.

    Returns:
        DataFrame com as colunas rsi, ema_rapida e ema_lenta adicionadas.
    """
    if df.empty or "close" not in df.columns or len(df) < Parametros.MIN_PERIODOS:
        return df
    df["rsi"]        = calcular_rsi(df["close"],  Parametros.RSI_PERIODO)
    df["ema_rapida"] = calcular_ema(df["close"],  Parametros.EMA_RAPIDA)
    df["ema_lenta"]  = calcular_ema(df["close"],  Parametros.EMA_LENTA)
    return df


def analisar_sinais(df: pd.DataFrame) -> Dict:
    if df.empty or len(df) < 2:
        return {"tipo": "NEUTRO", "forca": 0, "motivo": "Dados insuficientes"}

    u = df.iloc[-1]
    p = df.iloc[-2]

    rsi = u.get("rsi")
    er  = u.get("ema_rapida")
    el  = u.get("ema_lenta")

    if pd.isna(rsi) or pd.isna(er) or pd.isna(el):
        return {"tipo": "NEUTRO", "forca": 0, "motivo": "Aguardando dados suficientes"}

    preco      = u.get("close", 0)
    cruz_alta  = er > el  and p.get("ema_rapida", 0) <= p.get("ema_lenta", 0)
    cruz_baixa = er < el  and p.get("ema_rapida", 0) >= p.get("ema_lenta", 0)

    if rsi < Parametros.RSI_SOBREVENDA and cruz_alta:
        return {"tipo": "COMPRA FORTE", "forca": 90, "preco": preco, "rsi": rsi,
                "motivo": f"RSI sobrevendido ({rsi:.1f}) + cruzamento EMA de alta"}
    if rsi < Parametros.RSI_SOBREVENDA:
        return {"tipo": "COMPRA", "forca": 60, "preco": preco, "rsi": rsi,
                "motivo": f"RSI sobrevendido ({rsi:.1f})"}
    if cruz_alta:
        return {"tipo": "COMPRA", "forca": 50, "preco": preco, "rsi": rsi,
                "motivo": "Cruzamento EMA de alta"}
    if rsi > Parametros.RSI_SOBRECOMPRA and cruz_baixa:
        return {"tipo": "VENDA FORTE", "forca": 90, "preco": preco, "rsi": rsi,
                "motivo": f"RSI sobrecomprado ({rsi:.1f}) + cruzamento EMA de baixa"}
    if rsi > Parametros.RSI_SOBRECOMPRA:
        return {"tipo": "VENDA", "forca": 60, "preco": preco, "rsi": rsi,
                "motivo": f"RSI sobrecomprado ({rsi:.1f})"}
    if cruz_baixa:
        return {"tipo": "VENDA", "forca": 50, "preco": preco, "rsi": rsi,
                "motivo": "Cruzamento EMA de baixa"}

    return {"tipo": "NEUTRO", "forca": 0, "preco": preco, "rsi": rsi,
            "motivo": f"Sem sinal claro (RSI: {rsi:.1f})"}


# ---------------------------------------------------------------------------
# CICLO DE MONITORAMENTO
# ---------------------------------------------------------------------------
def executar_ciclo(ticker: str, config: dict) -> Optional[Dict]:
    """Busca dados → calcula indicadores → analisa sinal → retorna sinal."""
    log.info(f"Ciclo iniciado para {ticker}")
    df = buscar_dados(ticker, config["token_brapi"])
    if df.empty:
        log.warning(f"Sem dados para {ticker}")
        return None
    df    = calcular_indicadores(df)
    sinal = analisar_sinais(df)
    if sinal["tipo"] != "NEUTRO":
        registrar_sinal(ticker, sinal)
    return sinal


# ---------------------------------------------------------------------------
# INTERFACE
# ---------------------------------------------------------------------------
def exibir_sinal(sinal: Dict) -> None:
    tipo  = sinal.get("tipo", "NEUTRO")
    forca = sinal.get("forca", 0)
    if "COMPRA" in tipo:
        st.success(f"### 🟢 {tipo}")
    elif "VENDA" in tipo:
        st.error(f"### 🔴 {tipo}")
    else:
        st.info(f"### ⚪ {tipo}")
    if forca:
        st.progress(forca / 100, text=f"Força do sinal: {forca}%")
    st.write(f"**Motivo:** {sinal.get('motivo', '')}")
    c1, c2 = st.columns(2)
    if "preco" in sinal:
        c1.metric("Preço atual", f"R$ {sinal['preco']:.2f}")
    if "rsi" in sinal:
        c2.metric("RSI", f"{sinal['rsi']:.1f}")


def main() -> None:
    config = carregar_config()

    # ── Session state ─────────────────────────────────────────────────────
    for chave, padrao in [
        ("monitorando", False),
        ("ultimo_sinal", None),
        ("ciclos", 0),
        ("ultima_atualizacao", None),
    ]:
        if chave not in st.session_state:
            st.session_state[chave] = padrao

    # ── Cabeçalho ─────────────────────────────────────────────────────────
    st.title("📈 Robô Elite B3 — Monitor de Pregão")

    info  = status_pregao()
    agora = datetime.now().strftime("%H:%M:%S")
    st.markdown(
        f"{info['icone']} **{info['texto']}** &nbsp;·&nbsp; "
        f"<span style='color:gray;font-size:13px'>{agora}</span>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configurações")
        acao      = st.text_input("Código da ação", value="BBAS3")
        intervalo = st.slider("Intervalo entre ciclos (s)",
                              min_value=30, max_value=300,
                              value=Parametros.INTERVALO_LOOP, step=30)
        modo_sim  = st.toggle("Modo simulação (paper trade)", value=True)

        if modo_sim:
            st.success("🧪 Paper trade — nenhuma ordem real será enviada")
        else:
            st.warning("⚠️ Modo real — ordens serão enviadas à corretora")

        st.divider()
        st.subheader("📲 Telegram")
        telegram_ativo = st.toggle("Enviar alertas via Telegram", value=True)
        alertar_neutro = st.toggle("Alertar também sinais NEUTROS", value=False)
        freq_status    = st.number_input(
            "Relatório de status a cada N ciclos (0 = desativado)",
            min_value=0, max_value=100, value=10, step=5,
        )
        if st.button("🧪 Testar Telegram agora", use_container_width=True):
            ok = enviar_telegram(
                "✅ <b>Robô Elite B3</b>\nConexão com Telegram funcionando!",
                config["token_telegram"],
                config["id_telegram"],
            )
            if ok:
                st.success("Mensagem enviada! Verifique seu Telegram.")
            else:
                st.error("Falha. Verifique TOKEN e ID no secrets.toml.")

        st.divider()
        st.caption(f"RSI Sobrevenda  < {Parametros.RSI_SOBREVENDA}")
        st.caption(f"RSI Sobrecompra > {Parametros.RSI_SOBRECOMPRA}")
        st.caption(f"EMA Rápida: {Parametros.EMA_RAPIDA} períodos")
        st.caption(f"EMA Lenta:  {Parametros.EMA_LENTA} períodos")
        st.divider()
        st.caption(f"Ciclos executados: {st.session_state.ciclos}")
        if st.session_state.ultima_atualizacao:
            st.caption(f"Última atualização: {st.session_state.ultima_atualizacao}")

    # ── Botões de controle ────────────────────────────────────────────────
    col_ligar, col_parar = st.columns(2)
    with col_ligar:
        if st.button("▶️ Ligar Monitor", type="primary",
                     use_container_width=True,
                     disabled=st.session_state.monitorando):
            st.session_state.monitorando = True
            log.info(f"Monitor LIGADO — {acao.strip().upper()}")
            if telegram_ativo:
                enviar_telegram(
                    f"▶️ <b>Robô Elite B3 LIGADO</b>\n"
                    f"📌 Monitorando: <code>{normalizar_ticker(acao)}</code>\n"
                    f"⏱ Intervalo: {intervalo}s\n"
                    f"{'🧪 Modo simulação' if modo_sim else '💰 Modo real'}",
                    config["token_telegram"], config["id_telegram"],
                )
            st.rerun()

    with col_parar:
        if st.button("⏹️ Parar Monitor",
                     use_container_width=True,
                     disabled=not st.session_state.monitorando):
            st.session_state.monitorando = False
            log.info("Monitor DESLIGADO pelo usuário")
            if telegram_ativo:
                enviar_telegram(
                    f"⏹️ <b>Robô Elite B3 DESLIGADO</b>\n"
                    f"🔁 Ciclos executados: {st.session_state.ciclos}",
                    config["token_telegram"], config["id_telegram"],
                )
            st.rerun()

    # ── Painel do último sinal ────────────────────────────────────────────
    st.subheader("📡 Último sinal detectado")
    sinal_area = st.empty()

    if st.session_state.ultimo_sinal:
        with sinal_area.container():
            exibir_sinal(st.session_state.ultimo_sinal)
    else:
        sinal_area.info("Aguardando o primeiro ciclo de monitoramento…")

    # ── Tabs de histórico ─────────────────────────────────────────────────
    st.divider()
    tab_sinais, tab_ordens, tab_log = st.tabs(["📋 Sinais", "📝 Ordens", "🗒️ Log bruto"])

    with tab_sinais:
        if ARQUIVO_SINAIS.exists():
            df_s = pd.read_csv(ARQUIVO_SINAIS)
            if not df_s.empty:
                st.dataframe(df_s.sort_values("timestamp", ascending=False).head(50),
                             use_container_width=True, hide_index=True)
            else:
                st.info("Nenhum sinal registrado ainda.")
        else:
            st.info("Nenhum sinal registrado ainda.")

    with tab_ordens:
        if ARQUIVO_ORDENS.exists():
            df_o = pd.read_csv(ARQUIVO_ORDENS)
            if not df_o.empty:
                st.dataframe(df_o.sort_values("timestamp", ascending=False).head(50),
                             use_container_width=True, hide_index=True)
            else:
                st.info("Nenhuma ordem registrada ainda.")
        else:
            st.info("Nenhuma ordem registrada ainda.")

    with tab_log:
        if ARQUIVO_LOG.exists():
            linhas = ARQUIVO_LOG.read_text(encoding="utf-8").strip().splitlines()
            st.code("\n".join(linhas[-50:]), language="text")
        else:
            st.info("Nenhuma entrada de log ainda.")

    # ── Loop de monitoramento ─────────────────────────────────────────────
    # O Streamlit re-executa o script inteiro a cada st.rerun().
    # enquanto monitorando=True, um ciclo é executado e o próximo é agendado.
    if st.session_state.monitorando:
        if not dentro_do_pregao():
            info_p = status_pregao()
            st.warning(
                f"{info_p['icone']} {info_p['texto']} — monitoramento pausado. "
                "Será retomado automaticamente ao abrir o pregão."
            )
            log.info("Fora do pregão — aguardando 60s...")
            time.sleep(60)
            st.rerun()

        ticker_norm = normalizar_ticker(acao)

        with st.spinner(f"Analisando {ticker_norm}…"):
            sinal = executar_ciclo(acao, config)

        if sinal:
            st.session_state.ultimo_sinal       = sinal
            st.session_state.ciclos            += 1
            st.session_state.ultima_atualizacao = datetime.now().strftime("%H:%M:%S")
            tipo_sinal = sinal["tipo"]

            # ── Envio de alerta Telegram para sinais de compra/venda ──────
            if telegram_ativo:
                deve_alertar = (
                    "COMPRA" in tipo_sinal or
                    "VENDA"  in tipo_sinal or
                    (alertar_neutro and tipo_sinal == "NEUTRO")
                )
                if deve_alertar:
                    msg = montar_alerta_sinal(ticker_norm, sinal, modo_sim)
                    enviar_telegram(msg, config["token_telegram"], config["id_telegram"])

            # ── Relatório de status periódico ─────────────────────────────
            if (telegram_ativo and freq_status > 0
                    and st.session_state.ciclos % freq_status == 0):
                msg_status = montar_alerta_status(
                    ticker_norm, st.session_state.ciclos, sinal
                )
                enviar_telegram(msg_status, config["token_telegram"], config["id_telegram"])

            # ── Registra ordem quando há sinal de compra ou venda ─────────
            if "COMPRA" in tipo_sinal or "VENDA" in tipo_sinal:
                registrar_ordem(
                    tipo=tipo_sinal,
                    ticker=ticker_norm,
                    preco=sinal.get("preco", 0),
                    qtd=0,          # implementar position sizing
                    motivo=sinal.get("motivo", ""),
                    simulacao=modo_sim,
                )
                # Alerta de ordem separado do alerta de sinal
                if telegram_ativo:
                    msg_ordem = montar_alerta_ordem(
                        tipo=tipo_sinal,
                        ticker=ticker_norm,
                        preco=sinal.get("preco", 0),
                        qtd=0,
                        motivo=sinal.get("motivo", ""),
                        modo_sim=modo_sim,
                    )
                    enviar_telegram(msg_ordem, config["token_telegram"], config["id_telegram"])

        time.sleep(intervalo)
        st.rerun()


if __name__ == "__main__":
    main()
