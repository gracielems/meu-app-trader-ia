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
# INTERFACE VISUAL (STREAMLIT)
# ═══════════════════════════════════════════════════════════

st.title("🦈 Robô Tubarão B3")
st.markdown("### Painel de Controle - Day Trade")

st.divider()

# Criando 3 colunas para mostrar os parâmetros da estratégia
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Meta Diária", value=f"R$ {Config.META_DIA}")

with col2:
    st.metric(label="Perda Máxima (Stop)", value=f"R$ {Config.PERDA_MAX_DIA}")

with col3:
    st.metric(label="Risco por Trade", value=f"{Config.RISCO_POR_TRADE * 100}%")

st.divider()

st.info("🟢 Interface carregada com sucesso! O próximo passo é puxar os dados de mercado e mostrar os gráficos aqui.")
