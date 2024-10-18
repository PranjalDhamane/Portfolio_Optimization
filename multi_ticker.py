import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import streamlit as st

# Function to fetch data for multiple tickers
def fetch_multiple_ticker_data(tickers, start, end):
    all_data = {}
    all_returns = {}
    for ticker in tickers:
        data = yf.download(ticker, start=start, end=end)
        if data.empty:
            raise ValueError(f"No data fetched for the ticker: {ticker} and date range.")
        
        returns = data['Adj Close'].pct_change().dropna()
        all_data[ticker] = data
        all_returns[ticker] = returns
    return all_data, pd.DataFrame(all_returns)

# Monte Carlo simulation for efficient frontier
def monte_carlo_simulation(mean_returns, cov_matrix, num_portfolios=5000, risk_free_rate=0.01):
    num_assets = len(mean_returns)
    results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)

        portfolio_return = np.dot(weights, mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        results[0, i] = portfolio_return
        results[1, i] = portfolio_volatility
        results[2, i] = (portfolio_return - risk_free_rate) / portfolio_volatility

    return results, weights_record

# Function to plot efficient frontier
def plot_efficient_frontier(results):
    max_sharpe_idx = np.argmax(results[2])

    plt.figure(figsize=(10, 6))
    plt.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis', marker='o', alpha=0.6)
    plt.colorbar(label='Sharpe Ratio')

    plt.scatter(results[1, max_sharpe_idx], results[0, max_sharpe_idx], c='red', marker='*', s=200, label='Max Sharpe Ratio')
    plt.title('Efficient Frontier')
    plt.xlabel('Volatility (Risk)')
    plt.ylabel('Expected Return')
    plt.legend()
    st.pyplot(plt.gcf())

# Function to plot candlestick charts for multiple tickers
def plot_candlestick_charts_for_multiple_tickers(stock_data):
    for ticker, data in stock_data.items():
        st.write(f"*Candlestick Chart for {ticker}*")
        if "Volume" not in data.columns:
            st.error(f"Error: Volume data not available for {ticker}.")
            continue
        
        data_ohlc = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        mpf.plot(data_ohlc, type='candle', volume=True, title=f'Candlestick chart for {ticker}', style='yahoo')
        st.pyplot(plt.gcf())

# Function to run multiple ticker analysis
def run_multiple_ticker_analysis(tickers, start_date, end_date):
    try:
        stock_data, stock_returns = fetch_multiple_ticker_data(tickers, start=start_date, end=end_date)
        mean_returns = stock_returns.mean() * 252
        cov_matrix = stock_returns.cov() * 252

        # Calculate metrics
        metrics = []
        for ticker in tickers:
            returns = stock_returns[ticker]
            mean_return = mean_returns[ticker]
            volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = (mean_return - 0.01) / volatility  # Assume risk-free rate is 1%
            
            metrics.append({
                'Ticker': ticker,
                'Expected Annual Return': f"{mean_return:.4f}",
                'Annual Volatility': f"{volatility:.4f}",
                'Sharpe Ratio': f"{sharpe_ratio:.4f}"
            })

        # Convert metrics to DataFrame
        metrics_df = pd.DataFrame(metrics)
        
        # Display metrics table in Streamlit
        st.write("### Metrics Table for Multiple Ticker Analysis")
        st.table(metrics_df)

        # Monte Carlo Simulation
        results, weights_record = monte_carlo_simulation(mean_returns, cov_matrix)

        # Plot Efficient Frontier
        plot_efficient_frontier(results)

        # Display candlestick charts
        plot_candlestick_charts_for_multiple_tickers(stock_data)

    except Exception as e:
        st.error(f"Error: {e}")
