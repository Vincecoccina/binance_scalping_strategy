import pandas as pd
import pandas_ta as ta
import numpy as np
import os
from binance.client import Client
from binance import ThreadedWebsocketManager
from dotenv import load_dotenv
from datetime import datetime, timedelta
import time

# Chargement des variables d'environnement pour la sécurité des clés API
load_dotenv()
api_key = os.getenv("API_KEY_TEST")
secret_key = os.getenv("SECRET_KEY_TEST")

client = Client(api_key=api_key, api_secret=secret_key, tld='com',testnet=False)

# I - Récupérer les données historiques
# ------------------------------------------
# Fonction de récupération des données
def get_historical_data(symbol, interval, limit=500):
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=['Open_time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time', 'Quote_asset_volume', 'Number_of_trades', 'Taker_buy_base_asset_volume', 'Taker_buy_quote_asset_volume', 'Ignore'])
    df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].astype(float)
    return df

# Récupération des données + netoyage
data = get_historical_data('BTCUSDT', "1d")
data = data.drop(columns=['Quote_asset_volume', 'Close_time', 'Number_of_trades','Taker_buy_base_asset_volume', 'Taker_buy_quote_asset_volume', 'Ignore' ])
data['Open_time'] = pd.to_datetime(data['Open_time'], unit='ms')
data.rename(columns={'Open_time': 'Date'}, inplace=True)
data.set_index('Date', inplace=True)

# II - Calculer SMA et SLOPE
# ------------------------------------------
# Calculer le SMA (Le SMA aide a identifier la tendance du marché, un SMA croissant indique une tendance haussière)
# Le SMA calcule la moyenne des prix de cloture sur une période donnée => tendance générale
def calculate_sma(data, length: int):
    return ta.sma(data['Close'], length)

# Calculer le SLOPE (utilisé pour identifier la direction et la force de la tendance)
def calculate_slope(series, period: int = 5):
    slopes = [0 for _ in range(period-1)]
    for i in range(period-1, len(series)):
        x = np.arange(period)
        y = series[i-period+1:i+1].values
        slope = np.polyfit(x, y, 1)[0]
        percent_slope = (slope / y[0]) * 100
        slopes.append(percent_slope)
    return slopes

# Ajout des calcules SMA et SLOPE au DataFrame
data["SMA_10"] = calculate_sma(data, 10)
data["SMA_20"] = calculate_sma(data, 20)
data["SMA_30"] = calculate_sma(data, 30)
data["Slope"] = calculate_slope(data["SMA_20"])
data.dropna(inplace=True)
data.reset_index(inplace=True, drop=True)

# III - Déterminer la tendance (haussière, baissière ou absence de tendance)
# -------------------------------------------------------------------------
# Fonction pour connaitre la tendance
def determinate_trend(data):
    if data["SMA_10"] > data["SMA_20"] > data["SMA_30"]:
        return 2 # Tendance haussière
    elif data["SMA_10"] < data["SMA_20"] < data["SMA_30"]:
        return 1 # Tendance baissière
    else:
        return 0 # Pas de tendance

# Ajout du calcule de la TENDANCE au DataFrame
data["Trend"] = data.apply(determinate_trend, axis=1)

# IV - Savoir si les chandeliers japonais au dessus ou en dessous de la moyenne mobile (MA)
# ------------------------------------------------------------------------------------------
# Fonction qui compare la bougie à la moyenne mobile
def check_candles(data, backcandles, ma_column):
    categories = [0 for _ in range(backcandles)]
    for i in range(backcandles, len(data)):
        if all(data["Close"][i-backcandles:i] > data[ma_column][i-backcandles:i]):
            categories.append(2) # Tendance Haussière
        elif all(data["Close"][i-backcandles:i] < data[ma_column][i-backcandles:i]):
            categories.append(1) #Tendance Baissière
        else:
            categories.append(0)

    return categories

# Ajout du calcule de la CATEGORY au DataFrame
data["Category"] = check_candles(data, 5, "SMA_20")

# V - Confirmation de la tendance avec l'indicateur ADX (Déterminer la force d'une tendance)
# ------------------------------------------------------------------------------------------
# Calcule de l'ADX
data.ta.adx(append=True)
# Fonction qui génère un signal pour la tendnace basé sur l'ADX
def generate_trend_signal(data, threshold=40):
    trend_signal = []
    for i in range(len(data)):
        if data["ADX"][i] > threshold:
            if data["DMP"][i] > data["DMN"][i]:
                trend_signal.append(2) # Tendance haussière
            else:
                trend_signal.append(1) # Tendance Baissière
        else:
            trend_signal.append(0) # Pas de tendance claire
    return trend_signal

data = data.rename(columns=lambda x: x[:3] if x.startswith("ADX") else x)
data = data.rename(columns=lambda x: x[:3] if x.startswith("DM") else x)

# Ajout du calcule du SIGNAL DE LA TENDANCE au DataFrame
data["Trend Signal"] = generate_trend_signal(data)
# Ajout de la CONFIRMATION DU SIGNAL DE LA TENDANCE au DataFrame
data["Confirmed Signal"] = data.apply(lambda row: row["Category"] if row["Category"] == row["Trend Signal"] else 0, axis=1)
data = data[data["Confirmed Signal"]!=0]


print(data)