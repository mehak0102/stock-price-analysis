import pandas as pd
import numpy as np
from typing import List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Moving Averages
def calculate_moving_averages(data: pd.DataFrame, windows: List[int] = [20, 50, 200]) -> pd.DataFrame:
    try:
        result = pd.DataFrame()

        for window in windows:
            result[f"MA_{window}"] = data["Close"].rolling(window=window).mean()

        return result

    except Exception as e:
        logger.error(f"Error calculating moving averages: {e}")
        return pd.DataFrame()


# RSI
def calculate_rsi(data: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    try:
        delta = data["Close"].diff()

        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return pd.DataFrame({"RSI": rsi})

    except Exception as e:
        logger.error(f"Error calculating RSI: {e}")
        return pd.DataFrame()


# Bollinger Bands
def calculate_bollinger_bands(data: pd.DataFrame, window: int = 20, num_std: float = 2) -> pd.DataFrame:

    try:
        result = pd.DataFrame()

        middle = data["Close"].rolling(window).mean()
        std = data["Close"].rolling(window).std()

        result["BB_Middle"] = middle
        result["BB_Upper"] = middle + (std * num_std)
        result["BB_Lower"] = middle - (std * num_std)
        result["BB_Width"] = (result["BB_Upper"] - result["BB_Lower"]) / middle

        return result

    except Exception as e:
        logger.error(f"Error calculating Bollinger Bands: {e}")
        return pd.DataFrame()


# MACD
def calculate_macd(data: pd.DataFrame, fast=12, slow=26, signal=9):

    try:
        result = pd.DataFrame()

        exp1 = data["Close"].ewm(span=fast, adjust=False).mean()
        exp2 = data["Close"].ewm(span=slow, adjust=False).mean()

        result["MACD"] = exp1 - exp2
        result["Signal"] = result["MACD"].ewm(span=signal, adjust=False).mean()
        result["MACD_Hist"] = result["MACD"] - result["Signal"]

        return result

    except Exception as e:
        logger.error(f"Error calculating MACD: {e}")
        return pd.DataFrame()


# Stochastic Oscillator
def calculate_stochastic_oscillator(data: pd.DataFrame, k_window=14, d_window=3):

    try:
        result = pd.DataFrame()

        low_min = data["Low"].rolling(k_window).min()
        high_max = data["High"].rolling(k_window).max()

        result["%K"] = 100 * ((data["Close"] - low_min) / (high_max - low_min))
        result["%D"] = result["%K"].rolling(d_window).mean()

        return result

    except Exception as e:
        logger.error(f"Error calculating Stochastic Oscillator: {e}")
        return pd.DataFrame()


# ATR
def calculate_atr(data: pd.DataFrame, window=14):

    try:
        high_low = data["High"] - data["Low"]
        high_close = abs(data["High"] - data["Close"].shift())
        low_close = abs(data["Low"] - data["Close"].shift())

        ranges = pd.concat([high_low, high_close, low_close], axis=1)

        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window).mean()

        return pd.DataFrame({"ATR": atr})

    except Exception as e:
        logger.error(f"Error calculating ATR: {e}")
        return pd.DataFrame()


# OBV
def calculate_obv(data: pd.DataFrame):

    try:
        obv = (np.sign(data["Close"].diff()) * data["Volume"]).fillna(0).cumsum()

        return pd.DataFrame({"OBV": obv})

    except Exception as e:
        logger.error(f"Error calculating OBV: {e}")
        return pd.DataFrame()


# ADX
def calculate_adx(data: pd.DataFrame, window=14):

    try:
        result = pd.DataFrame()

        high_diff = data["High"].diff()
        low_diff = data["Low"].diff()

        plus_dm = high_diff.where((high_diff > 0) & (high_diff > -low_diff), 0)
        minus_dm = (-low_diff).where((low_diff > 0) & (low_diff > -high_diff), 0)

        tr1 = data["High"] - data["Low"]
        tr2 = abs(data["High"] - data["Close"].shift())
        tr3 = abs(data["Low"] - data["Close"].shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        plus_di = 100 * (plus_dm.ewm(alpha=1/window).mean() / tr.ewm(alpha=1/window).mean())
        minus_di = 100 * (minus_dm.ewm(alpha=1/window).mean() / tr.ewm(alpha=1/window).mean())

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)

        adx = dx.ewm(alpha=1/window).mean()

        result["ADX"] = adx
        result["+DI"] = plus_di
        result["-DI"] = minus_di

        return result

    except Exception as e:
        logger.error(f"Error calculating ADX: {e}")
        return pd.DataFrame()


# Ichimoku
def calculate_ichimoku(data: pd.DataFrame):

    try:
        result = pd.DataFrame()

        high9 = data["High"].rolling(9).max()
        low9 = data["Low"].rolling(9).min()

        result["Tenkan"] = (high9 + low9) / 2

        high26 = data["High"].rolling(26).max()
        low26 = data["Low"].rolling(26).min()

        result["Kijun"] = (high26 + low26) / 2

        result["Senkou_A"] = ((result["Tenkan"] + result["Kijun"]) / 2).shift(26)

        high52 = data["High"].rolling(52).max()
        low52 = data["Low"].rolling(52).min()

        result["Senkou_B"] = ((high52 + low52) / 2).shift(26)

        result["Chikou"] = data["Close"].shift(-26)

        return result

    except Exception as e:
        logger.error(f"Error calculating Ichimoku: {e}")
        return pd.DataFrame()


# Fibonacci
def calculate_fibonacci_levels(data: pd.DataFrame):

    try:
        result = pd.DataFrame()

        high = data["High"].max()
        low = data["Low"].min()

        diff = high - low

        result["FIB_0"] = [high]
        result["FIB_0.236"] = [high - diff * 0.236]
        result["FIB_0.382"] = [high - diff * 0.382]
        result["FIB_0.5"] = [high - diff * 0.5]
        result["FIB_0.618"] = [high - diff * 0.618]
        result["FIB_0.786"] = [high - diff * 0.786]
        result["FIB_1"] = [low]

        return result

    except Exception as e:
        logger.error(f"Error calculating Fibonacci: {e}")
        return pd.DataFrame()


# Momentum Indicators
def calculate_momentum_indicators(data: pd.DataFrame):

    try:
        result = pd.DataFrame()

        result["ROC"] = data["Close"].pct_change(10) * 100
        result["Momentum"] = data["Close"].diff(10)

        high14 = data["High"].rolling(14).max()
        low14 = data["Low"].rolling(14).min()

        result["Williams_R"] = -100 * (high14 - data["Close"]) / (high14 - low14)

        typical_price = (data["High"] + data["Low"] + data["Close"]) / 3

        sma = typical_price.rolling(20).mean()

        mad = typical_price.rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x))))

        result["CCI"] = (typical_price - sma) / (0.015 * mad)

        return result

    except Exception as e:
        logger.error(f"Error calculating momentum indicators: {e}")
        return pd.DataFrame()