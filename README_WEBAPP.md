# Monte Carlo Stock Price Simulator - Web App

A Flask-based web application that performs Monte Carlo simulations on stock prices using geometric Brownian motion. Users can input a stock ticker, number of simulations, and forecast horizon to visualize potential future price paths.

## Features

- **Interactive Web Interface**: Clean, modern UI for inputting parameters
- **Real-time Data**: Fetches historical stock data using yfinance
- **Monte Carlo Simulation**: Uses geometric Brownian motion to forecast stock prices
- **Interactive Visualizations**: Plotly charts showing simulation paths, mean path, and confidence intervals
- **Risk Metrics**: Displays key statistics including:
  - Current price
  - Mean and median forecasts
  - Value at Risk (VaR) at 5th percentile
  - 95th percentile (best case scenario)
  - Price range (min/max)
  - Expected returns

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. **Clone or navigate to the repository**:
   ```bash
   cd /Users/aordorica/Documents/Portfolio_Optimization_2023
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   ```

3. **Activate the virtual environment**:

   On macOS/Linux:
   ```bash
   source venv/bin/activate
   ```

   On Windows:
   ```bash
   venv\Scripts\activate
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. **Start the Flask server**:
   ```bash
   python app.py
   ```

2. **Open your web browser** and navigate to:
   ```
   http://localhost:5000
   ```

3. **Use the application**:
   - Enter a stock ticker symbol (e.g., AAPL, MSFT, TSLA)
   - Set the number of simulations (100-10,000, default: 1,000)
   - Set the forecast horizon in trading days (1-1,000, default: 252 = 1 year)
   - Click "Run Simulation"

## How It Works

### Geometric Brownian Motion

The simulation uses the geometric Brownian motion model, which is the standard model for stock price movements:

```
S(t+1) = S(t) * exp((μ - 0.5σ²)Δt + σ√Δt * Z)
```

Where:
- `S(t)` = stock price at time t
- `μ` = mean daily return (drift)
- `σ` = standard deviation of daily returns (volatility)
- `Z` = random variable from standard normal distribution
- `Δt` = time step (1 day)

### Process Flow

1. **Data Fetching**: Downloads 1 year of historical stock data using yfinance
2. **Statistical Analysis**: Calculates mean return (drift) and volatility (std deviation)
3. **Simulation**: Runs multiple Monte Carlo paths starting from the last closing price
4. **Visualization**: Displays results with interactive Plotly charts
5. **Risk Analysis**: Calculates summary statistics including VaR

## API Endpoints

### `GET /`
Renders the main application page.

### `POST /simulate`
Runs the Monte Carlo simulation.

**Request Parameters** (form data):
- `ticker` (string): Stock ticker symbol
- `num_simulations` (integer): Number of simulation paths (100-10,000)
- `forecast_days` (integer): Number of trading days to forecast (1-1,000)

**Response** (JSON):
```json
{
  "success": true,
  "ticker": "AAPL",
  "num_simulations": 1000,
  "forecast_days": 252,
  "summary": {
    "mean": 175.50,
    "median": 173.20,
    "std": 15.30,
    "min": 120.40,
    "max": 245.80,
    "var_5": 145.60,
    "var_95": 210.30,
    "current_price": 170.00,
    "mean_return_pct": 3.24,
    "var_5_return_pct": -14.35
  },
  "chart": "..."
}
```

## Understanding the Results

### Summary Statistics

- **Current Price**: The last closing price from historical data
- **Mean Forecast**: Average predicted price across all simulations
- **Median Forecast**: Middle value of all predicted prices
- **VaR 5% (Worst Case)**: There's a 95% chance the price will be above this value
- **95th Percentile**: Only 5% of simulations resulted in prices higher than this
- **Price Range**: Minimum and maximum prices across all simulations

### Chart Elements

- **Blue Lines**: Individual simulation paths (up to 100 shown for clarity)
- **Red Line**: Mean path across all simulations
- **Green Dashed Line**: 95th percentile boundary
- **Orange Dashed Line**: 5th percentile boundary (VaR)

## Limitations and Considerations

1. **Historical Performance**: The model assumes future returns follow the same statistical distribution as historical returns
2. **Market Changes**: Does not account for major market events, policy changes, or company-specific news
3. **Volatility Clustering**: Assumes constant volatility, but real markets often show volatility clustering
4. **No Mean Reversion**: Stock prices can drift indefinitely in the model
5. **Trading Days**: Assumes 252 trading days per year (excludes weekends and holidays)

## Troubleshooting

### "No data found for ticker"
- Verify the ticker symbol is correct and traded on major exchanges
- Check your internet connection
- Try a different ticker symbol

### Port 5000 already in use
Change the port in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)
```

### Module not found errors
Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## Technical Stack

- **Backend**: Flask 3.0.0
- **Data Source**: yfinance 0.2.36
- **Numerical Computing**: NumPy 1.26.3, Pandas 2.1.4
- **Visualization**: Plotly 5.18.0
- **Optional**: Alpaca Trade API (included for compatibility with existing codebase)

## License

This project is part of the Portfolio_Optimization_2023 repository.

## Contributing

Feel free to submit issues or pull requests for improvements.

## Contact

For questions or feedback, please open an issue in the repository.
