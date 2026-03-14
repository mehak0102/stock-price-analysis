import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
import time
import logging
from typing import Dict, List, Optional
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@st.cache_data(ttl=1800)
def load_stock_data(ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Load stock data for the given ticker and date range.
    """

    try:
        # Input validation
        if not isinstance(ticker, str) or not ticker.strip():
            raise ValueError("Invalid ticker symbol")

        if not isinstance(start_date, datetime) or not isinstance(end_date, datetime):
            raise ValueError("Invalid date format")

        if start_date > end_date:
            raise ValueError("Start date cannot be after end date")

        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        logger.info(f"Fetching data for {ticker} from {start_str} to {end_str}")

        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:

                data = yf.download(
                    ticker,
                    start=start_str,
                    end=end_str,
                    progress=False,
                    auto_adjust=True
                )

                if data.empty:
                    logger.warning(f"No data found for {ticker}")
                    time.sleep(retry_delay)
                    continue

                data = data.reset_index()

                required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in data.columns for col in required_columns):
                    logger.warning("Missing required columns")
                    return pd.DataFrame()

                # Technical indicators
                data['Daily Return'] = data['Close'].pct_change() * 100
                data['Cumulative Return'] = (1 + data['Daily Return']/100).cumprod() - 1
                data['Volatility'] = data['Daily Return'].rolling(20).std() * np.sqrt(252)

                data['Volume MA'] = data['Volume'].rolling(20).mean()
                data['Volume Change'] = data['Volume'].pct_change() * 100

                logger.info(f"Successfully loaded data for {ticker}")

                return data

            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed: {e}")
                time.sleep(retry_delay)

        return pd.DataFrame()

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        st.error(str(e))
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_stock_info(ticker: str) -> Dict:
    """
    Get detailed stock information
    """

    try:
        if not ticker:
            raise ValueError("Invalid ticker")

        stock = yf.Ticker(ticker)
        info = stock.info

        result = {
            "shortName": info.get("shortName", ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "currentPrice": info.get("currentPrice", 0),
            "marketCap": info.get("marketCap", 0),
            "trailingPE": info.get("trailingPE", 0),
            "beta": info.get("beta", 0),
            "volume": info.get("volume", 0),
        }

        return result

    except Exception as e:
        logger.error(f"Error getting info: {e}")
        st.error(str(e))
        return {}


@st.cache_data(ttl=1800)
def get_multiple_stocks_data(tickers: List[str], start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Load multiple stocks for comparison
    """

    try:
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            group_by='ticker',
            auto_adjust=True
        )

        adj_close = pd.DataFrame()

        for ticker in tickers:
            try:
                adj_close[ticker] = data[ticker]["Close"]
            except:
                continue

        if adj_close.empty:
            return pd.DataFrame()

        # Normalize data
        adj_close = adj_close / adj_close.iloc[0] * 100

        return adj_close

    except Exception as e:
        logger.error(f"Error loading multiple stocks: {e}")
        st.error(str(e))
        return pd.DataFrame()


def get_real_time_price(ticker: str) -> Optional[float]:
    """
    Get real-time stock price
    """

    try:
        stock = yf.Ticker(ticker)
        return float(stock.info.get("currentPrice", 0))

    except Exception as e:
        logger.error(f"Error getting real-time price: {e}")
        return None