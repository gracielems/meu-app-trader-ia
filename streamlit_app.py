"""
Robô Elite B3 — Analisador de ações da B3 via brapi.dev
"""

import streamlit as st
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Robô Elite B3", layout="wide")


# ---------------------------------------------------------------------------
# CONFIGURAÇÕES — lidas de st.secrets (nunca hardcoded)
# Crie o arquivo .streamlit/secrets.toml com:
#   TOKEN_TELEGRAM = "seu-token"
#   ID_TELEGRAM    = "seu-id"
#   TOKEN_BRAPI    = "seu-token"
# ---------------------------------------------------------------------------
def carregar_config() -> dict:
    """Retorna as credenciais lidas de st.secrets com fallback informativo."""
    try:
        return {
            "token_telegram": st.secrets["TOKEN_TELEGRAM"],
            "id_telegram":    st.secrets["ID_TELEGRAM"],
            "token_brapi":    st.secrets["TOKEN_BRAPI"],
        }
    except KeyError as e:
        st.error(
            f"Chave ausente em st.secrets: {e}. "
            "Crie o arquivo .streamlit/secrets.toml com as credenciais necessárias."
        )
        st.stop()


# ---------------------------------------------------------------------------
# LÓGICA DE NEGÓCIO
# ---------------------------------------------------------------------------
def normalizar_ticker(ticker: str) -> str:
    """Garante que o ticker esteja no formato esperado pela brapi.dev (ex: PETR4.SA).

    Args:
        ticker: Código da ação digitado pelo usuário.

    Returns:
        Ticker em maiúsculas com sufixo '.SA'.
    """
    t = ticker.strip().upper()
    if not t.endswith(".SA"):
        t = f"{t}.SA"
    return t


def buscar_dados(ticker: str, token_brapi: str) -> pd.DataFrame:
    """Busca o histórico de 5 dias / intervalo de 5 min via brapi.dev.

    Args:
        ticker:      Código da ação (ex: PETR4 ou PETR4.SA).
        token_brapi: Token de autenticação da API brapi.dev.

    Returns:
        DataFrame com os dados históricos ou DataFrame vazio em caso de falha.
    """
    t = normalizar_ticker(ticker)
    url = f"https://brapi.dev/api/quote/{t}?range=5d&interval=5m&token={token_brapi}"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()          # lança HTTPError para 4xx/5xx

        data = response.json()
        results = data.get("results", [])

        if not results or "historicalData" not in results[0]:
            st.warning("A API não retornou dados históricos para este ticker.")
            return pd.DataFrame()

        return pd.DataFrame(results[0]["historicalData"])

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
# INTERFACE
# ---------------------------------------------------------------------------
def main() -> None:
    """Ponto de entrada principal da aplicação Streamlit."""
    config = carregar_config()

    st.title("🚀 Analista de Elite B3")
    st.write("Status: Aguardando abertura do mercado (10:00)")

    acao = st.text_input("Digite a ação (ex: PETR4, VALE3)", value="BBAS3")

    if st.button("Ligar Monitor"):
        if not acao.strip():
            st.warning("Por favor, informe o código de uma ação.")
            return

        st.info(f"Monitorando {acao.strip().upper()}…")
        df = buscar_dados(acao, config["token_brapi"])

        if not df.empty:
            st.success("Dados recebidos com sucesso!")
            st.dataframe(df.tail(), use_container_width=True)
        else:
            st.warning("Mercado fechado ou erro na API. Tente após as 10:00.")


if __name__ == "__main__":
    main()
