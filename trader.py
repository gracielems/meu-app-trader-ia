import time
import requests
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Robô Elite B3", page_icon="🚀", layout="wide")

# CONFIGURAÇÕES - Verifique se seu Token está correto aqui
TOKEN_TELEGRAM = "8512230023:AAGyZ0QwPmnWqnZeQKfS7KCkwcVf1fBckCY"
ID_TELEGRAM = "7453152256"
TOKEN_BRAPI = "u4ufLNdYFG1Qo3xwHmuoem"

def load_data_brapi(ticker, interval="5m"):
    t = ticker.strip().upper().replace(".SA", "")
    # Mudamos para range de 5d para garantir que venha histórico para os cálculos
    url = f"https://brapi.dev/api/quote/{t}?range=5d&interval={interval}&token={TOKEN_BRAPI}"
    
    try:
        response = requests.get(url, timeout=10)
        
        # DEBUG: Se não for 200 (OK), avisa o que aconteceu
        if response.status_code != 200:
            st.error(f"Erro na API ({t}): Código {response.status_code} - {response.reason}")
            return pd.DataFrame()
            
        data = response.json()
        if 'results' in data and data['results'][0].get('historicalData'):
            df = pd.DataFrame(data['results'][0]['historicalData'])
            df['date'] = pd.to_datetime(df['date'], unit='s')
            df = df.rename(columns={'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low', 'volume': 'Volume'})
            return df.set_index('date').sort_index()
        
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Falha de conexão: {e}")
        return pd.DataFrame()

def main():
    st.title("📈 Monitor de Elite B3 - Real Time")
    
    tickers_input = st.sidebar.text_input("Ações (Sem .SA)", "PETR4, VALE3, BBAS3, ITUB4")
    intervalo = st.sidebar.selectbox("Gráfico", ["1m", "5m", "15m", "30m", "1h"], index=1)
    
    if st.sidebar.button("🚀 LIGAR ROBÔ"):
        tickers = [t.strip().upper() for t in tickers_input.split(",")]
        status = st.empty()
        tabela = st.empty()
        
        while True:
            hora_br = (datetime.utcnow() - timedelta(hours=3)).strftime('%H:%M:%S')
            status.info(f"🔄 Varredura Ativa | Horário B3: {hora_br}")
            
            resultados = []
            for t in tickers:
                df = load_data_brapi(t, intervalo)
                if not df.empty and len(df) > 20:
                    # Cálculo simplificado mas forte para ver se funciona
                    t_last = df.iloc[-1]
                    ema9 = df['Close'].ewm(span=9).mean().iloc[-1]
                    
                    res = {
                        "ticker": t,
                        "preço": f"R$ {t_last['Close']:.2f}",
                        "tendência": "ALTA" if t_last['Close'] > ema9 else "BAIXA",
                        "score": 100 if t_last['Close'] > ema9 else 0
                    }
                    resultados.append(res)
            
            if resultados:
                tabela.table(pd.DataFrame(resultados))
            else:
                st.warning("Aguardando dados da Brapi... Se persistir, verifique o Token.")

            time.sleep(60)
            st.rerun()

if __name__ == "__main__":
    main()
