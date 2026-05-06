"""
Robô Elite B3 — Analisador de ações da B3 com indicadores técnicos
Indicadores: RSI, Médias Móveis Exponenciais (9 e 21 períodos)
"""

import streamlit as st
import pandas as pd
import requests
import pandas_ta as ta
from datetime import datetime
from typing import Optional, Dict, Tuple

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Robô Elite B3", layout="wide", page_icon="📈")


# ---------------------------------------------------------------------------
# PARÂMETROS DE ESTRATÉGIA
# ---------------------------------------------------------------------------
class Parametros:
    """Parâmetros da estratégia de trading."""
    
    RSI_PERIODO = 14
    RSI_SOBRECOMPRA = 70
    RSI_SOBREVENDA = 30
    
    EMA_RAPIDA = 9
    EMA_LENTA = 21
    
    # Quantidade mínima de dados para calcular indicadores
    MIN_PERIODOS = 30


# ---------------------------------------------------------------------------
# CONFIGURAÇÕES — lidas de st.secrets
# ---------------------------------------------------------------------------
def carregar_config() -> dict:
    """Retorna as credenciais lidas de st.secrets com fallback informativo."""
    try:
        return {
            "token_telegram": st.secrets["TOKEN_TELEGRAM"],
            "id_telegram": st.secrets["ID_TELEGRAM"],
            "token_brapi": st.secrets["TOKEN_BRAPI"],
        }
    except KeyError as e:
        st.error(
            f"Chave ausente em st.secrets: {e}. "
            "Crie o arquivo .streamlit/secrets.toml com as credenciais necessárias."
        )
        st.stop()


# ---------------------------------------------------------------------------
# FUNÇÕES DE API
# ---------------------------------------------------------------------------
def normalizar_ticker(ticker: str) -> str:
    """Garante que o ticker esteja no formato .SA para brapi.dev."""
    t = ticker.strip().upper()
    if not t.endswith(".SA"):
        t = f"{t}.SA"
    return t


def buscar_dados(ticker: str, token_brapi: str) -> pd.DataFrame:
    """Busca o histórico de 5 dias / intervalo de 5 min via brapi.dev.
    
    Args:
        ticker: Código da ação (ex: PETR4 ou PETR4.SA).
        token_brapi: Token de autenticação da API brapi.dev.
    
    Returns:
        DataFrame com os dados históricos ou DataFrame vazio em caso de falha.
    """
    t = normalizar_ticker(ticker)
    url = f"https://brapi.dev/api/quote/{t}?range=5d&interval=5m&token={token_brapi}"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        results = data.get("results", [])
        
        if not results or "historicalData" not in results[0]:
            st.warning("A API não retornou dados históricos para este ticker.")
            return pd.DataFrame()
        
        df = pd.DataFrame(results[0]["historicalData"])
        
        # Converter timestamp para datetime se existir
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], unit="s")
        
        return df
    
    except requests.exceptions.Timeout:
        st.error("Tempo de conexão esgotado. Verifique sua internet e tente novamente.")
    except requests.exceptions.HTTPError as e:
        st.error(f"Erro HTTP ao acessar a API: {e}")
    except requests.exceptions.RequestException as e:
        st.error(f"Erro de rede inesperado: {e}")
    except (KeyError, IndexError, ValueError) as e:
        st.error(f"Formato de resposta inesperado da API: {e}")
    
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# CÁLCULO DE INDICADORES TÉCNICOS
# ---------------------------------------------------------------------------
def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula RSI e médias móveis exponenciais no DataFrame.
    
    Args:
        df: DataFrame com pelo menos a coluna 'close'.
    
    Returns:
        DataFrame com as novas colunas: rsi, ema_rapida, ema_lenta.
    """
    if df.empty or "close" not in df.columns:
        return df
    
    if len(df) < Parametros.MIN_PERIODOS:
        st.warning(
            f"Dados insuficientes para cálculo ({len(df)} períodos). "
            f"Mínimo necessário: {Parametros.MIN_PERIODOS}."
        )
        return df
    
    # RSI (Relative Strength Index)
    df["rsi"] = ta.rsi(df["close"], length=Parametros.RSI_PERIODO)
    
    # Médias Móveis Exponenciais
    df["ema_rapida"] = ta.ema(df["close"], length=Parametros.EMA_RAPIDA)
    df["ema_lenta"] = ta.ema(df["close"], length=Parametros.EMA_LENTA)
    
    return df


def analisar_sinais(df: pd.DataFrame) -> Dict[str, any]:
    """Analisa os indicadores e retorna sinais de compra/venda.
    
    Args:
        df: DataFrame com indicadores calculados.
    
    Returns:
        Dicionário com tipo de sinal, força e detalhes.
    """
    if df.empty or len(df) < 2:
        return {"tipo": "NEUTRO", "forca": 0, "motivo": "Dados insuficientes"}
    
    # Pega os últimos valores (mais recente)
    ultimo = df.iloc[-1]
    penultimo = df.iloc[-2]
    
    rsi_atual = ultimo.get("rsi")
    ema_rapida_atual = ultimo.get("ema_rapida")
    ema_lenta_atual = ultimo.get("ema_lenta")
    
    # Verifica se os indicadores estão disponíveis
    if pd.isna(rsi_atual) or pd.isna(ema_rapida_atual) or pd.isna(ema_lenta_atual):
        return {"tipo": "NEUTRO", "forca": 0, "motivo": "Aguardando dados suficientes"}
    
    preco_atual = ultimo.get("close", 0)
    
    # ---------------------------------------------------------------------------
    # LÓGICA DE SINAIS
    # ---------------------------------------------------------------------------
    
    # Sinal de COMPRA: RSI em sobrevenda + EMA rápida cruzou acima da lenta
    cruzamento_alta = (
        ema_rapida_atual > ema_lenta_atual and 
        penultimo.get("ema_rapida", 0) <= penultimo.get("ema_lenta", 0)
    )
    
    if rsi_atual < Parametros.RSI_SOBREVENDA and cruzamento_alta:
        return {
            "tipo": "COMPRA FORTE",
            "forca": 90,
            "motivo": f"RSI em sobrevenda ({rsi_atual:.1f}) + cruzamento de médias para alta",
            "preco": preco_atual,
            "rsi": rsi_atual,
        }
    
    elif rsi_atual < Parametros.RSI_SOBREVENDA:
        return {
            "tipo": "COMPRA",
            "forca": 60,
            "motivo": f"RSI em sobrevenda ({rsi_atual:.1f})",
            "preco": preco_atual,
            "rsi": rsi_atual,
        }
    
    elif cruzamento_alta:
        return {
            "tipo": "COMPRA",
            "forca": 50,
            "motivo": "EMA rápida cruzou acima da lenta (tendência de alta)",
            "preco": preco_atual,
            "rsi": rsi_atual,
        }
    
    # Sinal de VENDA: RSI em sobrecompra + EMA rápida cruzou abaixo da lenta
    cruzamento_baixa = (
        ema_rapida_atual < ema_lenta_atual and 
        penultimo.get("ema_rapida", 0) >= penultimo.get("ema_lenta", 0)
    )
    
    if rsi_atual > Parametros.RSI_SOBRECOMPRA and cruzamento_baixa:
        return {
            "tipo": "VENDA FORTE",
            "forca": 90,
            "motivo": f"RSI em sobrecompra ({rsi_atual:.1f}) + cruzamento de médias para baixa",
            "preco": preco_atual,
            "rsi": rsi_atual,
        }
    
    elif rsi_atual > Parametros.RSI_SOBRECOMPRA:
        return {
            "tipo": "VENDA",
            "forca": 60,
            "motivo": f"RSI em sobrecompra ({rsi_atual:.1f})",
            "preco": preco_atual,
            "rsi": rsi_atual,
        }
    
    elif cruzamento_baixa:
        return {
            "tipo": "VENDA",
            "forca": 50,
            "motivo": "EMA rápida cruzou abaixo da lenta (tendência de baixa)",
            "preco": preco_atual,
            "rsi": rsi_atual,
        }
    
    # Sem sinal claro
    return {
        "tipo": "NEUTRO",
        "forca": 0,
        "motivo": f"Aguardando configuração (RSI: {rsi_atual:.1f})",
        "preco": preco_atual,
        "rsi": rsi_atual,
    }


# ---------------------------------------------------------------------------
# INTERFACE STREAMLIT
# ---------------------------------------------------------------------------
def exibir_painel_sinal(sinal: Dict) -> None:
    """Exibe o sinal de trading em destaque."""
    tipo = sinal.get("tipo", "NEUTRO")
    forca = sinal.get("forca", 0)
    motivo = sinal.get("motivo", "")
    
    # Define cor baseada no tipo de sinal
    if "COMPRA" in tipo:
        cor = "🟢"
        estilo = "success"
    elif "VENDA" in tipo:
        cor = "🔴"
        estilo = "error"
    else:
        cor = "⚪"
        estilo = "info"
    
    st.markdown(f"### {cor} Sinal: {tipo}")
    
    if forca > 0:
        st.progress(forca / 100, text=f"Força do sinal: {forca}%")
    
    st.info(f"**Motivo:** {motivo}")
    
    # Exibe métricas adicionais
    col1, col2 = st.columns(2)
    
    if "preco" in sinal:
        col1.metric("Preço atual", f"R$ {sinal['preco']:.2f}")
    
    if "rsi" in sinal:
        col2.metric("RSI", f"{sinal['rsi']:.1f}")


def main() -> None:
    """Ponto de entrada principal da aplicação Streamlit."""
    config = carregar_config()
    
    st.title("📈 Robô Elite B3 — Análise Técnica")
    st.caption("Indicadores: RSI (14) + Médias Móveis Exponenciais (9/21)")
    
    # Sidebar com parâmetros
    with st.sidebar:
        st.header("⚙️ Configurações")
        acao = st.text_input("Código da ação", value="BBAS3")
        
        st.divider()
        st.caption(f"RSI Sobrevenda: < {Parametros.RSI_SOBREVENDA}")
        st.caption(f"RSI Sobrecompra: > {Parametros.RSI_SOBRECOMPRA}")
        st.caption(f"EMA Rápida: {Parametros.EMA_RAPIDA} períodos")
        st.caption(f"EMA Lenta: {Parametros.EMA_LENTA} períodos")
    
    if st.button("🔍 Analisar Ação", type="primary", use_container_width=True):
        if not acao.strip():
            st.warning("Por favor, informe o código de uma ação.")
            return
        
        ticker_normalizado = normalizar_ticker(acao)
        
        with st.spinner(f"Buscando dados de {ticker_normalizado}..."):
            df = buscar_dados(acao, config["token_brapi"])
        
        if df.empty:
            st.warning("Não foi possível obter dados. Mercado pode estar fechado ou ticker inválido.")
            return
        
        # Calcula indicadores
        with st.spinner("Calculando indicadores técnicos..."):
            df = calcular_indicadores(df)
        
        # Analisa sinais
        sinal = analisar_sinais(df)
        
        # Exibe o sinal em destaque
        st.divider()
        exibir_painel_sinal(sinal)
        
        # Exibe dados detalhados
        st.divider()
        st.subheader("📊 Últimos dados com indicadores")
        
        # Seleciona colunas relevantes para exibição
        colunas_exibir = ["date", "close", "rsi", "ema_rapida", "ema_lenta"]
        colunas_disponiveis = [c for c in colunas_exibir if c in df.columns]
        
        st.dataframe(
            df[colunas_disponiveis].tail(20),
            use_container_width=True,
            hide_index=True
        )
        
        # Informações adicionais
        with st.expander("ℹ️ Como interpretar os sinais"):
            st.markdown("""
            **Sinais de COMPRA:**
            - RSI abaixo de 30 indica que o ativo está sobrevendido (possível alta)
            - EMA rápida cruzando acima da lenta indica início de tendência de alta
            - Sinal FORTE quando ambas as condições ocorrem simultaneamente
            
            **Sinais de VENDA:**
            - RSI acima de 70 indica que o ativo está sobrecomprado (possível queda)
            - EMA rápida cruzando abaixo da lenta indica início de tendência de baixa
            - Sinal FORTE quando ambas as condições ocorrem simultaneamente
            
            **IMPORTANTE:** Estes são apenas indicadores técnicos. Sempre use stop loss 
            e nunca opere com dinheiro que não pode perder.
            """)


if __name__ == "__main__":
    main()
