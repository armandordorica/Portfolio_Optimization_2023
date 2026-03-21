"""
Flask Web Application for Monte Carlo Stock Price Simulation
Uses yfinance for data fetching and implements geometric Brownian motion
"""

from flask import Flask, render_template, request, jsonify
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.utils
import json
from datetime import datetime, timedelta

app = Flask(__name__)

class StockMonteCarloSimulation:
    """
    Monte Carlo simulation for individual stock using geometric Brownian motion
    """

    def __init__(self, ticker, num_simulations=1000, forecast_days=252):
        self.ticker = ticker
        self.num_simulations = num_simulations
        self.forecast_days = forecast_days
        self.simulated_prices = None
        self.last_price = None
        self.mean_return = None
        self.std_return = None

    def fetch_data(self, period="1y"):
        """Fetch historical data using yfinance"""
        try:
            stock = yf.Ticker(self.ticker)
            df = stock.history(period=period)

            if df.empty:
                raise ValueError(f"No data found for ticker {self.ticker}")

            return df
        except Exception as e:
            raise ValueError(f"Error fetching data for {self.ticker}: {str(e)}")

    def calculate_statistics(self, df):
        """Calculate mean and standard deviation of daily returns"""
        df['Daily_Return'] = df['Close'].pct_change()
        self.mean_return = df['Daily_Return'].mean()
        self.std_return = df['Daily_Return'].std()
        self.last_price = df['Close'].iloc[-1]

        return df

    def run_simulation(self):
        """Run Monte Carlo simulation using geometric Brownian motion"""
        # Fetch and prepare data
        df = self.fetch_data()
        self.calculate_statistics(df)

        # Initialize array to hold simulated prices
        simulated_prices = np.zeros((self.forecast_days, self.num_simulations))

        # Run simulations
        for sim in range(self.num_simulations):
            prices = [self.last_price]

            for day in range(self.forecast_days):
                # Geometric Brownian motion formula
                # S(t+1) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
                # Where Z is a random normal variable
                drift = (self.mean_return - 0.5 * self.std_return ** 2)
                shock = self.std_return * np.random.randn()
                price = prices[-1] * np.exp(drift + shock)
                prices.append(price)

            simulated_prices[:, sim] = prices[1:]

        self.simulated_prices = simulated_prices
        return simulated_prices

    def get_summary_statistics(self):
        """Calculate summary statistics from simulation results"""
        if self.simulated_prices is None:
            raise ValueError("Simulation must be run before calculating statistics")

        final_prices = self.simulated_prices[-1, :]

        summary = {
            'mean': np.mean(final_prices),
            'median': np.median(final_prices),
            'std': np.std(final_prices),
            'min': np.min(final_prices),
            'max': np.max(final_prices),
            'var_5': np.percentile(final_prices, 5),  # Value at Risk (5th percentile)
            'var_95': np.percentile(final_prices, 95),
            'current_price': self.last_price,
            'mean_return_pct': ((np.mean(final_prices) - self.last_price) / self.last_price) * 100,
            'var_5_return_pct': ((np.percentile(final_prices, 5) - self.last_price) / self.last_price) * 100,
        }

        return summary

    def create_plotly_chart(self):
        """Create interactive Plotly chart of simulation paths"""
        if self.simulated_prices is None:
            raise ValueError("Simulation must be run before creating chart")

        # Create figure
        fig = go.Figure()

        # Add a subset of simulation paths (showing all can be too cluttered)
        num_paths_to_show = min(100, self.num_simulations)
        indices = np.random.choice(self.num_simulations, num_paths_to_show, replace=False)

        for idx in indices:
            fig.add_trace(go.Scatter(
                y=self.simulated_prices[:, idx],
                mode='lines',
                line=dict(width=0.5, color='rgba(100, 149, 237, 0.3)'),
                showlegend=False,
                hovertemplate='Day: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ))

        # Add mean path
        mean_path = np.mean(self.simulated_prices, axis=1)
        fig.add_trace(go.Scatter(
            y=mean_path,
            mode='lines',
            name='Mean Path',
            line=dict(width=3, color='red'),
            hovertemplate='Day: %{x}<br>Mean Price: $%{y:.2f}<extra></extra>'
        ))

        # Add percentile bands
        percentile_5 = np.percentile(self.simulated_prices, 5, axis=1)
        percentile_95 = np.percentile(self.simulated_prices, 95, axis=1)

        fig.add_trace(go.Scatter(
            y=percentile_95,
            mode='lines',
            name='95th Percentile',
            line=dict(width=2, color='green', dash='dash'),
            hovertemplate='Day: %{x}<br>95th Percentile: $%{y:.2f}<extra></extra>'
        ))

        fig.add_trace(go.Scatter(
            y=percentile_5,
            mode='lines',
            name='5th Percentile (VaR)',
            line=dict(width=2, color='orange', dash='dash'),
            hovertemplate='Day: %{x}<br>5th Percentile: $%{y:.2f}<extra></extra>'
        ))

        # Update layout
        fig.update_layout(
            title=f'Monte Carlo Simulation: {self.ticker} ({self.num_simulations} simulations, {self.forecast_days} days)',
            xaxis_title='Trading Days',
            yaxis_title='Stock Price ($)',
            hovermode='x unified',
            template='plotly_white',
            height=600,
            showlegend=True
        )

        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/simulate', methods=['POST'])
def simulate():
    """Handle simulation request"""
    try:
        # Get form data
        ticker = request.form.get('ticker', '').upper().strip()
        num_simulations = int(request.form.get('num_simulations', 1000))
        forecast_days = int(request.form.get('forecast_days', 252))

        # Validate inputs
        if not ticker:
            return jsonify({'error': 'Please provide a stock ticker'}), 400

        if num_simulations < 100 or num_simulations > 10000:
            return jsonify({'error': 'Number of simulations must be between 100 and 10,000'}), 400

        if forecast_days < 1 or forecast_days > 1000:
            return jsonify({'error': 'Forecast days must be between 1 and 1,000'}), 400

        # Run simulation
        mc_sim = StockMonteCarloSimulation(
            ticker=ticker,
            num_simulations=num_simulations,
            forecast_days=forecast_days
        )

        mc_sim.run_simulation()
        summary = mc_sim.get_summary_statistics()
        chart_json = mc_sim.create_plotly_chart()

        # Prepare response
        response = {
            'success': True,
            'ticker': ticker,
            'num_simulations': num_simulations,
            'forecast_days': forecast_days,
            'summary': summary,
            'chart': chart_json
        }

        return jsonify(response)

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
