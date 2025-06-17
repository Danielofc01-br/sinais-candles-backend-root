import pandas as pd
import requests
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from sklearn.ensemble import RandomForestClassifier
import numpy as np

model_cache = {}
PAIRS = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]

def get_binance_klines(symbol="BTCUSDT", interval="1m", limit=100):
    url = f"https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    data = requests.get(url, params=params).json()
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'num_trades',
        'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

def add_indicators(df):
    df['rsi'] = RSIIndicator(df['close']).rsi()
    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    bb = BollingerBands(df['close'])
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    return df.dropna()

def prepare_data(df):
    df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
    X = df[['rsi', 'macd', 'macd_signal', 'bb_high', 'bb_low']]
    y = df['target']
    return X[:-1], y[:-1], X.iloc[[-1]]

def train_model(X, y):
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    return model

def get_signal_for_pair(pair):
    df = get_binance_klines(pair, interval="1m")
    df = add_indicators(df)
    X, y, latest = prepare_data(df)
    if pair not in model_cache:
        model_cache[pair] = train_model(X, y)
    model = model_cache[pair]
    prob = model.predict_proba(latest)[0]
    sinal = "compra" if prob[1] > 0.5 else "venda"
    return {
        "par": pair,
        "timeframe": "1m",
        "sinal": sinal,
        "confianca": round(max(prob) * 100, 2),
    }

def get_latest_signals():
    sinais = []
    for pair in PAIRS:
        try:
            sinais.append(get_signal_for_pair(pair))
        except Exception as e:
            sinais.append({"par": pair, "erro": str(e)})
    return sinais
