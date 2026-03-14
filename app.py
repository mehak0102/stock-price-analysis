import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import pytz
import plotly.graph_objects as go
import os
from plotly.subplots import make_subplots

# Import custom modules
from data_loader import load_stock_data, get_stock_info, get_multiple_stocks_data
from technical_indicators import (
    calculate_moving_averages, calculate_rsi, 
    calculate_bollinger_bands, calculate_macd,
    calculate_stochastic_oscillator
)
from visualization import plot_stock_data, plot_prediction_results, plot_multiple_stocks_comparison
from prediction import make_prediction, evaluate_prediction
from database import (
    initialize_database, add_user, get_user_by_username,
    add_to_watchlist, remove_from_watchlist, get_watchlist,
    save_prediction, get_user_predictions, get_prediction_accuracy
)

# Initialize database
initialize_database()

# Set page config
st.set_page_config(
    page_title="Nifty 50 Stock Analysis & Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define some constants
IST = pytz.timezone('Asia/Kolkata')
DEFAULT_START_DATE = datetime.now(IST) - timedelta(days=180)
DEFAULT_END_DATE = datetime.now(IST)

# Define Nifty 50 stocks with their sectors
NIFTY50_STOCKS = {
    "Finance": ["HDFCBANK.NS", "ICICIBANK.NS", "AXISBANK.NS", "SBIN.NS", "BAJFINANCE.NS", "KOTAKBANK.NS", "BAJAJFINSV.NS", "INDUSINDBK.NS"],
    "Information Technology": ["TCS.NS", "INFY.NS", "WIPRO.NS", "TECHM.NS", "HCLTECH.NS"],
    "Oil & Gas": ["RELIANCE.NS", "ONGC.NS", "IOC.NS", "GAIL.NS", "BPCL.NS"],
    "Automobile": ["MARUTI.NS", "M&M.NS", "TATAMOTORS.NS", "HEROMOTOCO.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS"],
    "Consumer Goods": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "TITAN.NS", "BRITANNIA.NS"],
    "Metals & Mining": ["TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "COALINDIA.NS"],
    "Pharmaceuticals": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS"],
    "Infrastructure": ["LT.NS", "ADANIPORTS.NS", "NTPC.NS", "POWERGRID.NS"],
    "Telecom": ["BHARTIARTL.NS"],
    "Others": ["ASIANPAINT.NS", "ULTRACEMCO.NS", "GRASIM.NS", "UPL.NS", "HDFCLIFE.NS", "ZOMATO.NS"]
}

# Function to format currency
def format_currency(value):
    """Format a value as Indian Rupees."""
    if pd.isna(value) or value is None:
        return "N/A"
    
    if value >= 10000000:  # Convert to Crores (1 Crore = 10 Million)
        return f"₹ {value/10000000:.2f} Cr"
    elif value >= 100000:  # Convert to Lakhs (1 Lakh = 100 Thousand)
        return f"₹ {value/100000:.2f} L"
    else:
        return f"₹ {value:.2f}"

# Function to handle login/registration
def handle_auth():
    """Handle user authentication."""
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    
    if 'username' not in st.session_state:
        st.session_state.username = None
    
    # If already logged in, show logout button
    if st.session_state.user_id is not None:
        st.sidebar.write(f"Logged in as: {st.session_state.username}")
        if st.sidebar.button("Logout"):
            st.session_state.user_id = None
            st.session_state.username = None
            st.rerun()
        return
    
    # Login/Registration form
    auth_option = st.sidebar.radio("Login or Register", ["Login", "Register"])
    
    username = st.sidebar.text_input("Username")
    
    if auth_option == "Login":
        if st.sidebar.button("Login"):
            if username:
                user = get_user_by_username(username)
                if user:
                    st.session_state.user_id = user.id
                    st.session_state.username = user.username
                    st.sidebar.success(f"Welcome back, {username}!")
                    st.rerun()
                else:
                    st.sidebar.error("User not found. Please register.")
            else:
                st.sidebar.warning("Please enter a username.")
    else:  # Register
        if st.sidebar.button("Register"):
            if username:
                # Check if user exists
                existing_user = get_user_by_username(username)
                if existing_user:
                    st.sidebar.error("Username already taken.")
                else:
                    user_id = add_user(username)
                    if user_id:
                        st.session_state.user_id = user_id
                        st.session_state.username = username
                        st.sidebar.success(f"Welcome, {username}! Account created successfully.")
                        st.rerun()
            else:
                st.sidebar.warning("Please enter a username.")

# Function to display watchlist
def display_watchlist(user_id):
    """Display user's watchlist."""
    st.subheader("Your Watchlist")
    
    watchlist = get_watchlist(user_id)
    
    if not watchlist:
        st.info("Your watchlist is empty. Add stocks by clicking the 'Add to Watchlist' button when viewing a stock.")
        return
    
    # Create columns for a grid layout
    cols = st.columns(3)
    
    # Display stock cards
    for i, symbol in enumerate(watchlist):
        col_idx = i % 3
        
        with cols[col_idx]:
            # Get basic stock info
            info = get_stock_info(symbol)
            
            # Create a card-like display
            card = st.container()
            card.markdown(f"### {info.get('shortName', symbol)}")
            
            # Price and change
            price = info.get('currentPrice', 0)
            change = info.get('dayChange', 0)
            
            price_color = "green" if change >= 0 else "red"
            change_symbol = "▲" if change >= 0 else "▼"
            
            card.markdown(f"""
            <div style="display: flex; justify-content: space-between;">
                <span style="color: {price_color}; font-size: 20px; font-weight: bold;">{format_currency(price)}</span>
                <span style="color: {price_color};">{change_symbol} {abs(change):.2f}%</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Show some metrics
            card.markdown(f"""
            <div style="display: flex; justify-content: space-between; margin-top: 10px;">
                <span>P/E: {info.get('trailingPE', 0):.2f}</span>
                <span>Div: {info.get('dividendYield', 0):.2f}%</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Action buttons
            col1, col2 = card.columns(2)
            if col1.button(f"View {symbol}", key=f"view_{symbol}"):
                st.session_state.selected_stock = symbol
                st.session_state.view = "stock_detail"
                st.rerun(
if col2.button(f"Remove", key=f"remove_{symbol}"):
                if remove_from_watchlist(user_id, symbol):
                    st.success(f"{symbol} removed from watchlist.")
                    st.rerun()
                else:
                    st.error(f"Failed to remove {symbol} from watchlist.")

# Function to display recent predictions
def display_predictions(user_id):
    """Display recent predictions for the user."""
    st.subheader("Your Recent Predictions")
 predictions_df = get_user_predictions(user_id)
 if predictions_df.empty:
        st.info("You haven't made any predictions yet. Go to any stock detail page to make predictions.")
        return
    
    # Format the DataFrame for display
    display_df = predictions_df.copy()
    
    # Format dates
    display_df['prediction_date'] = display_df['prediction_date'].dt.strftime('%Y-%m-%d')
    display_df['target_date'] = display_df['target_date'].dt.strftime('%Y-%m-%d')
    
    # Format prices
    display_df['predicted_price'] = display_df['predicted_price'].apply(lambda x: f"₹ {x:.2f}")
    display_df['actual_price'] = display_df['actual_price'].apply(lambda x: f"₹ {x:.2f}" if pd.notna(x) else "Pending")
    
    # Rename columns for display
    display_df = display_df.rename(columns={
        'stock_symbol': 'Stock',
        'prediction_date': 'Date Made',
        'target_date': 'Target Date',
        'predicted_price': 'Predicted Price',
        'actual_price': 'Actual Price',
        'model_type': 'Model'
    })
    
    # Reorder and select columns
    display_df = display_df[['Stock', 'Date Made', 'Target Date', 'Predicted Price', 'Actual Price', 'Model']]
    
    # Display as a table
    st.table(display_df)

def display_stock_info(stock_info: dict):
    """
    Display comprehensive stock information in a formatted way.
    Args:
        stock_info (dict): Stock information dictionary
    """
    if not stock_info:
        st.warning("No stock information available")
        return
    
    # Create columns for better layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Current Price",
            f"₹{stock_info.get('currentPrice', 0):,.2f}",
            f"{stock_info.get('dayChange', 0):+.2f}%"
        )
        st.metric(
            "Market Cap",
            format_currency(stock_info.get('marketCap', 0))
        )
        st.metric(
            "Dividend Yield",
            f"{stock_info.get('dividendYield', 0):.2f}%"
        )
 with col2:
        st.metric(
            "PE Ratio (TTM)",
            f"{stock_info.get('trailingPE', 0):.2f}"
        )
        st.metric(
            "Forward PE",
            f"{stock_info.get('forwardPE', 0):.2f}"
        )
        st.metric(
            "Beta",
            f"{stock_info.get('beta', 0):.2f}"
        )
 with col3:
        st.metric(
            "Sector",
            stock_info.get('sector', 'N/A')
        )
        st.metric(
            "Industry",
            stock_info.get('industry', 'N/A')
        )
        st.metric(
            "Exchange",
            stock_info.get('exchange', 'N/A')
        )

# Main app layout
def main():
    # Initialize selected_stock at the start
    if 'selected_stock' not in st.session_state:
        st.session_state.selected_stock = None

    # CSS for better styling
    st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stock-card {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .metric-container {
        display: flex;
        justify-content: space-between;
    }
    .metric-box {
        border: 1px solid #f0f0f0;
        border-radius: 5px;
        padding: 10px;
        text-align: center;
        background-color: #f9f9f9;
    }
    .metric-label {
        font-size: 0.8em;
        color: #666;
    }
    .metric-value {
        font-size: 1.2em;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar - User Authentication
    st.sidebar.title("Nifty 50 Stock Analysis")
    handle_auth()
    
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    
    # Only show personal sections if logged in
    if st.session_state.user_id is not None:
        navigation_options = ["Home", "Watchlist", "Sector Analysis", "My Predictions"]
    else:
        navigation_options = ["Home", "Sector Analysis"]
if 'view' not in st.session_state:
        st.session_state.view = navigation_options[0]
    
    # When a stock is selected, add it to the navigation options
    if st.session_state.selected_stock:
        stock_view = f"Stock: {st.session_state.selected_stock}"
        if stock_view not in navigation_options:
            navigation_options.append(stock_view)
    
    selected_view = st.sidebar.radio("Go to", navigation_options)
    
    # Update the view based on selection
    if selected_view != st.session_state.view:
        st.session_state.view = selected_view
        # Clear selected stock if not viewing stock detail
        if "Stock:" not in selected_view:
            st.session_state.selected_stock = None
    
    # Sidebar - Stock Selection (for Home view)
    if st.session_state.view == "Home":
        st.sidebar.subheader("Select Stock")
        
        # Group stocks by sector
        selected_sector = st.sidebar.selectbox(
            "Sector",
            list(NIFTY50_STOCKS.keys()),
            key="sector_selector"
        )
        
        # Show stocks in the selected sector
        if selected_sector:
            selected_stock = st.sidebar.selectbox(
                "Stock",
                NIFTY50_STOCKS[selected_sector],
                key="stock_selector"
            )
if st.sidebar.button("View Stock", key="view_stock_button"):
                st.session_state.selected_stock = selected_stock
                st.session_state.view = "stock_detail"
                st.rerun()
    
    # Sidebar - Technical Analysis Configuration
    if st.session_state.view == "stock_detail" or "Stock:" in st.session_state.view:
        st.sidebar.subheader("Technical Analysis")
        show_ma = st.sidebar.checkbox("Show Moving Averages", value=True)
        show_bb = st.sidebar.checkbox("Show Bollinger Bands", value=False)
        show_rsi = st.sidebar.checkbox("Show RSI", value=True)
        show_macd = st.sidebar.checkbox("Show MACD", value=False)
    
    # Sidebar - Date Range
    if st.session_state.view in ["Home", "stock_detail", "Sector Analysis"] or "Stock:" in st.session_state.view:
        st.sidebar.subheader("Date Range")
        start_date = st.sidebar.date_input(
            "Start Date",
            DEFAULT_START_DATE.date()
        )
        end_date = st.sidebar.date_input(
            "End Date",
            DEFAULT_END_DATE.date()
        )
        
        # Convert to datetime
        start_date = datetime.combine(start_date, datetime.min.time())
        end_date = datetime.combine(end_date, datetime.min.time())
    
            # Main Content Area
    if st.session_state.view == "Home":
        st.title("Nifty 50 Stock Analysis")

        # Featured Stock Section (Full Width)
        st.subheader("Featured Stock")
        # Add selectbox to choose featured stock dynamically
        all_stocks = [stock for sector_stocks in NIFTY50_STOCKS.values() for stock in sector_stocks]
        featured_stock = st.selectbox("Select Featured Stock", all_stocks, index=all_stocks.index("RELIANCE.NS") if "RELIANCE.NS" in all_stocks else 0)
        stock_info = get_stock_info(featured_stock)
        if stock_info:
            display_stock_info(stock_info)

        # Market Overview Metrics (Based on NIFTY 50 Index)
        st.subheader("Market Overview (Nifty 50)")
        
        # Load Nifty 50 Index data
        nifty_data = load_stock_data("^NSEI", start_date, end_date)
 if not nifty_data.empty:
            col1, col2, col3, col4 = st.columns(4)
current_price = float(nifty_data['Close'].iloc[-1].item())
            prev_price = float(nifty_data['Close'].iloc[-2].item())
            price_change = ((current_price - prev_price) / prev_price) * 100
col1.metric(
                "Current Value",
                f"₹{current_price:,.2f}",
                f"{price_change:+.2f}%"
            )
col2.metric(
                "52 Week High",
                f"₹{float(nifty_data['High'].max()):,.2f}"
            )
col3.metric(
                "52 Week Low",
                f"₹{float(nifty_data['Low'].min()):,.2f}"
            )
col4.metric(
                "Volume",
                f"{int(nifty_data['Volume'].iloc[-1].item()):,}"
            )
        else:
            st.warning("Unable to load Nifty 50 Index data.")

        # Featured Sectors Section
        st.subheader("Featured Sectors")

        # Sector Mapping
        sector_mapping = {
            "IT": ["TCS.NS", "INFY.NS", "WIPRO.NS"],
            "Banking": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS"],
            "Auto": ["MARUTI.NS", "TATAMOTORS.NS", "M&M.NS"]
        }

        all_sector_stocks = []
        for stocks in sector_mapping.values():
            all_sector_stocks.extend(stocks)
sector_data = get_multiple_stocks_data(all_sector_stocks, start_date, end_date)
  if not sector_data.empty:
            sector_avg = pd.DataFrame(index=sector_data.index)
for sector, stocks in sector_mapping.items():
                valid_stocks = [s for s in stocks if s in sector_data.columns]
                if valid_stocks:
                    sector_avg[sector] = sector_data[valid_stocks].mean(axis=1)
 fig = plot_multiple_stocks_comparison(sector_avg, title="Sector Performance Comparison")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Unable to load sector comparison data.")


            
    elif st.session_state.view == "stock_detail":
        if st.session_state.selected_stock:
            selected_stock = st.session_state.selected_stock
            st.title(f"Stock Detail: {selected_stock}")
            
            # Stock Information Section
            st.subheader("Stock Information")
            stock_info = get_stock_info(selected_stock)
            if stock_info:
                display_stock_info(stock_info)
            
            # Load stock data for chart
            stock_data = load_stock_data(selected_stock, start_date, end_date)
            
            if not stock_data.empty:
                # Create stock price chart
                fig = go.Figure()
                
                # Add candlestick chart
                fig.add_trace(
                    go.Candlestick(
                        x=stock_data['Date'],
                        open=stock_data['Open'],
                        high=stock_data['High'],
                        low=stock_data['Low'],
                        close=stock_data['Close'],
                        name=selected_stock
                    )
                )
                
                # Add volume bars
                fig.add_trace(
                    go.Bar(
                        x=stock_data['Date'],
                        y=stock_data['Volume'],
                        name='Volume',
                        yaxis='y2',
                        marker=dict(
                            color='rgba(100, 100, 255, 0.3)'
                        )
                    )
                )
                
                # Update layout
                fig.update_layout(
                    title=f"{selected_stock} Stock Price",
                    yaxis_title="Price (₹)",
                    yaxis2=dict(
                        title="Volume",
                        overlaying="y",
                        side="right"
                    ),
                    xaxis_rangeslider_visible=False,
                    height=600,
                    template="plotly_white",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                # Add range selector
                fig.update_xaxes(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1M", step="month", stepmode="backward"),
                            dict(count=3, label="3M", step="month", stepmode="backward"),
                            dict(count=6, label="6M", step="month", stepmode="backward"),
                            dict(count=1, label="YTD", step="year", stepmode="todate"),
                            dict(count=1, label="1Y", step="year", stepmode="backward"),
                            dict(step="all", label="ALL")
                        ])
                    )
                )
                
                # Display chart
                st.plotly_chart(fig, use_container_width=True)
                
                # Additional stock details
                st.subheader("Company Information")
                col1, col2 = st.columns(2)
                 with col1:
                    st.write(f"**Sector:** {stock_info.get('sector', 'N/A')}")
                    st.write(f"**Industry:** {stock_info.get('industry', 'N/A')}")
                    st.write(f"**Market Cap:** {format_currency(stock_info.get('marketCap', 0))}")
                    with col2:
                    st.write(f"**P/E Ratio:** {stock_info.get('trailingPE', 0):.2f}")
                    st.write(f"**Forward P/E:** {stock_info.get('forwardPE', 0):.2f}")
                    st.write(f"**Dividend Yield:** {stock_info.get('dividendYield', 0):.2f}%")
            else:
                st.error(f"Failed to load data for {selected_stock}")
        else:
            st.warning("Please select a stock to view details")
    elif st.session_state.view == "Watchlist":
        st.title("Your Watchlist")
       if st.session_state.user_id is None:
            st.warning("Please login to use the watchlist feature.")
        else:
            display_watchlist(st.session_state.user_id)
    elif st.session_state.view == "Sector Analysis":
        st.title("Sector Analysis")
        
        # Select sector for analysis
        selected_sector = st.selectbox(
            "Select Sector",
            list(NIFTY50_STOCKS.keys())
        )
         if selected_sector:
            st.subheader(f"{selected_sector} Sector Analysis")
            
            # Get stocks in the sector
            sector_stocks = NIFTY50_STOCKS[selected_sector]
            
            # Load data for all stocks in the sector
            sector_data = get_multiple_stocks_data(sector_stocks, start_date, end_date)
            
            if not sector_data.empty:
                # Plot sector comparison
                fig = plot_multiple_stocks_comparison(sector_data, title=f"{selected_sector} Stocks Comparison")
                st.plotly_chart(fig, use_container_width=True)
                
                # Display sector stocks in a grid with info cards
                st.subheader(f"{selected_sector} Stocks")
                
                # Create columns for grid layout
                cols = st.columns(3)
                
                for i, symbol in enumerate(sector_stocks):
                    col_idx = i % 3
                    
                    # Try to get stock info
                    try:
                        info = get_stock_info(symbol)
                        if not info:
                            continue
                         with cols[col_idx]:
                            st.markdown(f"### {info.get('shortName', symbol)}")
                            
                            # Current price and change
                            price = info.get('currentPrice', 0)
                            change = info.get('dayChange', 0)
                            price_color = "green" if change >= 0 else "red"
                            st.markdown(f"""
                            <div style="display: flex; justify-content: space-between;">
                                <span style="font-size: 18px;">{format_currency(price)}</span>
                                <span style="color: {price_color};">{change:.2f}%</span>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Company info
                            st.write(f"**Industry:** {info.get('industry', 'N/A')}")
                            st.write(f"**Market Cap:** {format_currency(info.get('marketCap', 0))}")
                            
                            # View button
                            if st.button(f"View {symbol}", key=f"view_sector_{symbol}"):
                                st.session_state.selected_stock = symbol
                                st.session_state.view = "stock_detail"
                                st.rerun()
                    except:
                        with cols[col_idx]:
                            st.markdown(f"### {symbol}")
                            st.warning(f"Unable to load data for {symbol}")
            else:
                st.error(f"Failed to load data for {selected_sector} sector.")
    elif st.session_state.view == "My Predictions":
        st.title("My Predictions")
        if st.session_state.user_id is None:
            st.warning("Please login to view your predictions.")
        else:
            display_predictions(st.session_state.user_id)
     elif st.session_state.view == "stock_detail" or "Stock:" in st.session_state.view:
        # Determine which stock to display
        if "Stock:" in st.session_state.view:
            # Extract stock symbol from the view name
            selected_stock = st.session_state.view.split("Stock: ")[1]
            st.session_state.selected_stock = selected_stock
        else:
            selected_stock = st.session_state.selected_stock
        
        # Load stock data
        stock_data = load_stock_data(selected_stock, start_date, end_date)
        if stock_data.empty:
            st.error(f"Failed to load data for {selected_stock}.")
            return
        
        # Get stock info
        stock_info = get_stock_info(selected_stock)
        company_name = stock_info.get('longName', selected_stock)
        
        # Main title
        st.title(f"{company_name} ({selected_stock})")
        
        # Stock Price and Stats
        col1, col2 = st.columns([2, 1])
        with col1:
            current_price = stock_info.get('currentPrice', 0)
            day_change = stock_info.get('dayChange', 0)
            
            # Format and display price
            price_color = "green" if day_change >= 0 else "red"
            change_symbol = "▲" if day_change >= 0 else "▼"
            
            st.markdown(f"""
            <div style="display: flex; align-items: baseline;">
                <span style="font-size: 36px; margin-right: 15px;">{format_currency(current_price)}</span>
                <span style="color: {price_color}; font-size: 20px;">{change_symbol} {abs(day_change):.2f}%</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Add to watchlist button (if logged in)
            if st.session_state.user_id is not None:
                watchlist = get_watchlist(st.session_state.user_id)
                if selected_stock in watchlist:
                    if st.button("Remove from Watchlist"):
                        if remove_from_watchlist(st.session_state.user_id, selected_stock):
                            st.success(f"{selected_stock} removed from watchlist.")
                            st.rerun()
                        else:
                            st.error("Failed to remove from watchlist.")
                else:
                    if st.button("Add to Watchlist"):
                        if add_to_watchlist(st.session_state.user_id, selected_stock):
                            st.success(f"{selected_stock} added to watchlist.")
                            st.rerun()
                        else:
                            st.error("Failed to add to watchlist.")
        
        # Company Information
        st.subheader("Company Information")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Sector", stock_info.get('sector', 'N/A'))
        col2.metric("Industry", stock_info.get('industry', 'N/A'))
        col3.metric("Market Cap", format_currency(stock_info.get('marketCap', 0)))
        col4.metric("P/E Ratio", f"{stock_info.get('trailingPE', 0):.2f}")
        
        # Interactive Stock Chart
        st.subheader("Stock Price Chart")
        
        # Calculate technical indicators if requested
        if show_ma:
            ma_data = calculate_moving_averages(stock_data)
            for col in ma_data.columns:
                stock_data[col] = ma_data[col]
        if show_bb:
            bb_data = calculate_bollinger_bands(stock_data)
            for col in bb_data.columns:
                stock_data[col] = bb_data[col]
             if show_rsi:
            rsi_data = calculate_rsi(stock_data)
            stock_data['RSI'] = rsi_data['RSI']
          if show_macd:
            macd_data = calculate_macd(stock_data)
            for col in macd_data.columns:
                stock_data[col] = macd_data[col]
        
        # Plot the stock chart
        fig = plot_stock_data(
            stock_data, 
            selected_stock, 
            show_ma=show_ma, 
            show_rsi=show_rsi
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Stock Prediction
        st.subheader("Stock Price Prediction")
        col1, col2 = st.columns(2)
         with col1:
            model_type = st.selectbox(
                "Prediction Model",
                ["Linear Regression", "Random Forest", "ARIMA"]
            )
         with col2:
            prediction_days = st.slider(
                "Prediction Period (Days)",
                min_value=5,
                max_value=30,
                value=7,
                step=1
            )
           if st.button("Generate Prediction"):
            with st.spinner("Generating prediction..."):
                try:
                    # Make prediction
                    predictions, future_dates = make_prediction(
                        stock_data,
                        prediction_days=prediction_days,
                        model_type=model_type
                    )
                    
                    # Plot prediction results
                    pred_fig = plot_prediction_results(
                        stock_data,
                        predictions,
                        future_dates,
                        selected_stock
                    )
                    st.plotly_chart(pred_fig, use_container_width=True)
                    # Show prediction metrics
                    metrics = evaluate_prediction(stock_data, model_type)
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Mean Absolute Error", f"₹ {metrics['mae']:.2f}")
                    col2.metric("R² Score", f"{metrics['r2']:.4f}")
                    col3.metric("Confidence Score", f"{metrics['confidence']:.2f}%")
                    
                    # Save prediction if logged in
                    if st.session_state.user_id is not None:
                        target_date = future_dates[-1]
                        predicted_price = predictions[-1]
                         pred_id = save_prediction(
                            selected_stock,
                            target_date,
                            predicted_price,
                            model_type,
                            st.session_state.user_id)
                     if pred_id:
                            st.success("Prediction saved to your account.")
                       except Exception as e:
 st.error(f"Error generating prediction: {str(e)}")
# Run the app
if __name__ == "__main__":
    main()
