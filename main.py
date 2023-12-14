import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
import os
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException
from binance import ThreadedWebsocketManager
from dotenv import load_dotenv
from datetime import datetime, timedelta
import time

# Chargement des variables d'environnement pour la sécurité des clés API
load_dotenv()
api_key = os.getenv("API_KEY_TEST")
secret_key = os.getenv("SECRET_KEY_TEST")

# Initialisation de l'API Binance avec les clés API
client = Client(api_key=api_key, api_secret=secret_key, tld='com', testnet=True)

symbol = 'TRXUSDT'
#symbol = 'RNDREUR'
# symbol = 'XRPUSDT'
# symbol = 'ETHUSDT'
# symbol = 'BTCUSDT'
# symbol = 'XMRUSDT'
#symbol = 'BNBUSDT'
# symbol = 'LTCUSDT'
# symbol = 'SOLUSDT'


class ScalpingTrader():
    
    def __init__(self, symbol, bar_length, units, stop_loss):
        self.symbol = symbol
        self.bar_length = bar_length
        self.available_intervals = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]
        self.units = units
        self.position = 0
        self.last_buy_time = None
        self.last_buy_price = None
        self.profit_target = 1.0
        self.max_hold_time = 5
        self.stop_loss = stop_loss
        self.trades = pd.DataFrame(columns=['Time',"Symbol", 'Type', 'OrderID', 'Price', 'Quantity','Total', 'Status'])

    def start_trading(self, historical_days):
        
        self.twm = ThreadedWebsocketManager()
        self.twm.start()
        
        if self.bar_length in self.available_intervals:
            self.get_most_recent(symbol = self.symbol, interval = self.bar_length,
                                 days = historical_days)
            self.twm.start_kline_socket(callback = self.stream_candles,
                                        symbol = self.symbol, interval = self.bar_length)
            
    def get_most_recent(self, symbol, interval, days):
            now = datetime.utcnow()
            past = str(now - timedelta(days = days))

            bars = client.get_historical_klines(symbol = symbol, interval = interval,
                                                start_str=past, end_str=None, limit=500)
            df = pd.DataFrame(bars)
            df["Date"] = pd.to_datetime(df.iloc[:,0], unit = "ms")
            df.columns = ["Open Time", "Open", "High", "Low", "Close", "Volume",
                        "Clos Time", "Quote Asset Volume", "Number of Trades",
                        "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume", "Ignore", "Date"]
            df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
            df.set_index("Date", inplace = True)
            df[['High', 'Low', 'Close', 'Volume']] = df[['High', 'Low', 'Close', 'Volume']].astype(float)
            for column in df.columns:
                df[column] = pd.to_numeric(df[column], errors = "coerce")
            df["Complete"] = [True for row in range(len(df)-1)] + [False]
            self.data = df
            self.calculate_indicators()
    
    def calculate_vwap_signal(self, backcandles=15):
        VWAPsignal = [0] * len(self.data)

        for row in range(backcandles, len(self.data)):
            upt = 1
            dnt = 1
            for i in range(row - backcandles, row + 1):
                if max(self.data["Open"].iloc[i], self.data["Close"].iloc[i]) >= self.data["VWAP"].iloc[i]:
                    dnt = 0
                if min(self.data["Open"].iloc[i], self.data["Close"].iloc[i]) <= self.data["VWAP"].iloc[i]:
                    upt = 0
            if upt == 1 and dnt == 1:
                VWAPsignal[row] = 3
            elif upt == 1:
                VWAPsignal[row] = 2
            elif dnt == 1:
                VWAPsignal[row] = 1

        self.data['VWAPSignal'] = VWAPsignal
    
    def total_signal(self, l):
        current_price = float(self.data["Close"].iloc[l])
        

        # Signal d'achat
        if (self.data["VWAPSignal"].iloc[l] == 2 and
            current_price <= self.data['Lower Band'].iloc[l] and
            self.data["RSI"].iloc[l] < 45):
            self.last_buy_time = self.data.index[l]
            self.last_buy_price = current_price
            self.stop_loss = current_price * (1 - self.stop_loss / 100)  # Exemple de stop-loss à un certain pourcentage sous le prix d'achat
            return 2

        # Signal de vente basé sur le profit
        if self.last_buy_price and (current_price >= self.last_buy_price * (1 + self.profit_target / 100)):
            self.last_buy_time = None
            self.last_buy_price = None
            return 1

        # Signal de vente basé sur le stop-loss
        if self.last_buy_price and (current_price <= self.stop_loss):
            self.last_buy_time = None
            self.last_buy_price = None
            return 1

        return 0


    def calculate_total_signal(self):
        TotSignal = [0]*len(self.data)
        backcandles = 15  # ou toute autre valeur appropriée
        for row in range(backcandles, len(self.data)):
            TotSignal[row] = self.total_signal(row)
        self.data['TotalSignal'] = TotSignal
    
    def pointposbreak(self, x):
        if x['TotalSignal'] == 1:
            return x['High'] + 1e-4
        elif x['TotalSignal'] == 2:
            return x['Low'] - 1e-4
        else:
            return np.nan

    def calculate_pointposbreak(self):
        self.data['pointposbreak'] = self.data.apply(self.pointposbreak, axis=1)

    def calculate_indicators(self):
        self.data['VWAP'] = ta.vwap(self.data['High'], self.data['Low'], self.data['Close'], self.data['Volume'])
        self.data['RSI'] = ta.rsi(self.data['Close'], length=16)
        self.data.ta.bbands(length=14, std=2.0, append=True)
        self.data["Upper Band"] = self.data["BBU_14_2.0"]
        self.data["Lower Band"] = self.data["BBL_14_2.0"]
        self.calculate_vwap_signal()
        self.calculate_total_signal()
        self.calculate_pointposbreak()  
                    
    def stream_candles(self, msg):
        event_time = pd.to_datetime(msg["E"], unit = "ms")
        start_time = pd.to_datetime(msg["k"]["t"], unit = "ms")
        first   = float(msg["k"]["o"])
        high    = float(msg["k"]["h"])
        low     = float(msg["k"]["l"])
        close   = float(msg["k"]["c"])
        volume  = float(msg["k"]["v"])
        complete=       msg["k"]["x"]
        print("Time: {} | Price: {}".format(event_time, close))
        new_data = pd.DataFrame({
        "Open": [first],
        "High": [high],
        "Low": [low],
        "Close": [close],
        "Volume": [volume],
        "Complete": [complete]
        }, index=[start_time])

        # Concaténer avec les données existantes
        self.data = pd.concat([self.data, new_data])
        # Recalcule des indicateurs
        self.calculate_indicators()

    def execute_trades(self):
        for index, row in self.data.iterrows():
            signal = row['TotalSignal']

            try:
                if signal == 2:  # Signal d'achat
                    # Vérifiez si vous n'avez pas déjà acheté
                    if self.position == 0:
                        order = client.create_order(symbol=self.symbol, side="BUY", type="MARKET", quantity=self.units)
                        self.position = 1  # Mise à jour de la position
                        print(f"Achat effectué à {index}: ", order)
                        self.record_trade(order, 'BUY', index)
                elif signal == 1:  # Signal de vente
                    # Vérifiez si vous avez une position à vendre
                    if self.position == 1:
                        order = client.create_order(symbol=self.symbol, side="SELL", type="MARKET", quantity=self.units)
                        self.position = 0  # Réinitialisation de la position
                        print(f"Vente effectuée à {index}: ", order)
                        self.record_trade(order, 'SELL', index)
            except BinanceOrderException as e:
                print(f"Erreur lors de la passation de l'ordre à {index}: {e}")
    
    def record_trade(self, order, type, time):
        price = float(order['fills'][0]['price'])
        quantity = float(order['fills'][0]['qty'])
        new_trade = {
            'Time': time,
            "Symbol": symbol,
            'Type': type,
            'OrderID': order['orderId'],
            'Price': price,
            'Quantity': quantity,
            'Total': price * quantity,
            'Status': order['status']
        }
        self.trades = self.trades._append(new_trade, ignore_index=True)




    
    # def plot_data(self):
    #     # Préparer les données de signal
    #     achat_signals = self.data[self.data['TotalSignal'] == 2]
    #     vente_signals = self.data[self.data['TotalSignal'] == 1]

    #     # Créer le graphique à chandeliers
    #     fig = go.Figure(data=[go.Candlestick(x=self.data.index,
    #                                          open=self.data["Open"],
    #                                          high=self.data["High"],
    #                                          low=self.data["Low"],
    #                                          close=self.data["Close"])])
    #     # Ajouter des lignes pour les indicateurs
    #     fig.add_trace(go.Scatter(x=self.data.index, y=self.data["VWAP"], mode="lines", name="VWAP", line=dict(color="blue")))
    #     fig.add_trace(go.Scatter(x=self.data.index, y=self.data["Lower Band"], mode="lines", name="Lower Band", line=dict(color="crimson")))
    #     fig.add_trace(go.Scatter(x=self.data.index, y=self.data["Upper Band"], mode="lines", name="Upper Band", line=dict(color="green")))

    #     # Ajouter des marqueurs pour les signaux
    #     fig.add_trace(go.Scatter(x=achat_signals.index, y=achat_signals['Close'],
    #                              mode='markers', marker=dict(color='green', size=10),
    #                              name='Signal Achat'))
    #     fig.add_trace(go.Scatter(x=vente_signals.index, y=vente_signals['Close'],
    #                              mode='markers', marker=dict(color='red', size=10),
    #                              name='Signal Vente'))
        
    #     # Afficher le graphique
    #     fig.show()

bar_length = "5m"
return_thresh = 0
volume_thresh = [-3, 3]
units = 80


trader = ScalpingTrader(symbol=symbol, bar_length=bar_length, units=units, stop_loss=5)
trader.start_trading(historical_days = 2)

run_time = 60 
start_time = time.time()
while time.time() - start_time < run_time:
    trader.execute_trades()
    time.sleep(1)

trader.twm.stop()
filtered_data = trader.data[trader.data["TotalSignal"]!=0]
print(filtered_data)
print(trader.trades[50:])
# trader.plot_data()