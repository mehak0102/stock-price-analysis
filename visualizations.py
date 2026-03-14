import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


def plot_stock_data(data, ticker, show_ma=True, ma_periods=[20, 50, 200], show_rsi=True):
    """
    Create an interactive plot of stock data with technical indicators.

    Args:
        data (pd.DataFrame): Stock data
        ticker (str): Stock ticker symbol
        show_ma (bool): Whether to show moving averages
        ma_periods (list): Periods for moving averages
        show_rsi (bool): Whether to show RSI indicator

    Returns:
        go.Figure: Plotly figure object
    """

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=[f"{ticker} Stock Price", "Volume / RSI"]
    )

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data['Date'],
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name=ticker
        ),
        row=1, col=1
    )

    # Volume chart
    fig.add_trace(
        go.Bar(
            x=data['Date'],
            y=data['Volume'],
            name="Volume",
            marker=dict(color='rgba(100,100,255,0.5)')
        ),
        row=2, col=1
    )

    # Moving averages
    if show_ma and ma_periods:
        colors = [
            'rgba(255,0,0,0.7)',
            'rgba(0,255,0,0.7)',
            'rgba(0,0,255,0.7)'
        ]

        for i, period in enumerate(ma_periods):
            ma_col = f"MA_{period}"

            if ma_col in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data['Date'],
                        y=data[ma_col],
                        name=f"{period}-day MA",
                        line=dict(color=colors[i % len(colors)], width=1.5)
                    ),
                    row=1, col=1
                )

    # RSI indicator
    if show_rsi and "RSI" in data.columns:

        fig.add_trace(
            go.Scatter(
                x=data['Date'],
                y=data['RSI'],
                name="RSI",
                line=dict(color="purple", width=1.5)
            ),
            row=2, col=1
        )

        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # Layout
    fig.update_layout(
        title=f"{ticker} Stock Analysis",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        xaxis_rangeslider_visible=False,
        height=700,
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)

    if show_rsi and "RSI" in data.columns:
        fig.update_yaxes(title_text="RSI", row=2, col=1)
    else:
        fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig


def plot_prediction_results(historical_data, predictions, future_dates, ticker):
    """
    Plot historical data with predictions.
    """

    fig = go.Figure()

    # Historical data
    fig.add_trace(
        go.Scatter(
            x=historical_data['Date'],
            y=historical_data['Close'],
            name="Historical Data",
            line=dict(color="blue")
        )
    )

    # Prediction line
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=predictions,
            name="Prediction",
            line=dict(color="red", dash="dash")
        )
    )

    # Last known price marker
    last_date = historical_data['Date'].iloc[-1]
    last_price = historical_data['Close'].iloc[-1]

    fig.add_trace(
        go.Scatter(
            x=[last_date],
            y=[last_price],
            mode="markers",
            marker=dict(color="green", size=10),
            name="Last Known Price"
        )
    )

    fig.update_layout(
        title=f"{ticker} Price Prediction",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        height=500,
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    fig.add_vline(
        x=last_date,
        line_dash="dash",
        line_color="black",
        annotation_text="Prediction Start",
        annotation_position="top right"
    )

    return fig


def plot_multiple_stocks_comparison(data, title="Stock Price Comparison"):
    """
    Compare multiple stocks on one chart.
    """

    fig = go.Figure()

    for column in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[column],
                name=column,
                mode="lines"
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Normalized Price (Base 100)",
        height=600,
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified"
    )

    fig.add_hline(
        y=100,
        line_dash="dash",
        line_color="black",
        annotation_text="Base Value",
        annotation_position="bottom right"
    )

    return fig