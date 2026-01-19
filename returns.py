from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import re
from typing import Optional, Tuple


def _price_series(df: pd.DataFrame) -> pd.Series:
    """Return a per-row price series, preferring Close and falling back to Price."""
    if 'Close' in df.columns and 'Price' in df.columns:
        return df['Close'].fillna(df['Price'])
    if 'Close' in df.columns:
        return df['Close']
    if 'Price' in df.columns:
        return df['Price']
    raise KeyError("Input dataframe must contain a 'Close' or 'Price' column")


def _parse_lookback(lookback: str) -> Tuple[int, str]:
    """Parse lookback strings like '30D', '6M', '1Y' into (value, unit)."""
    match = re.match(r"^\s*(\d+)\s*([DdMmYy])\s*$", str(lookback))
    if not match:
        raise ValueError("lookback must look like '30D', '6M', or '1Y'")
    value = int(match.group(1))
    unit = match.group(2).upper()
    if value <= 0:
        raise ValueError("lookback value must be > 0")
    return value, unit


def get_return_by_lookback(
    input_df: pd.DataFrame,
    lookback: str,
    most_recent_date: Optional[str] = None,
) -> float:
    """Calculate return over a lookback window like '1Y', '6M', '30D'.

    Uses 'Close' when available; falls back to 'Price' when Close is null/missing.

    Args:
        input_df: DataFrame for a single stock with date-like index.
        lookback: String lookback like '30D', '6M', '1Y'.
        most_recent_date: Optional end date ('YYYY-MM-DD'). If omitted, uses
            the latest date available in input_df.

    Returns:
        Fractional return over the window (e.g. 0.12 for +12%). Returns np.nan
        if no usable prices exist in the computed window.
    """
    if input_df.empty:
        return np.nan

    df = input_df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    end_dt = pd.to_datetime(most_recent_date) if most_recent_date else df.index.max()

    value, unit = _parse_lookback(lookback)
    if unit == 'D':
        start_dt = end_dt - pd.Timedelta(days=value)
    elif unit == 'M':
        start_dt = end_dt - pd.DateOffset(months=value)
    elif unit == 'Y':
        start_dt = end_dt - pd.DateOffset(years=value)
    else:
        raise ValueError("Unsupported lookback unit")

    # Align start/end to actual available trading dates (use closest on-or-before).
    eligible_start = df.index[df.index <= start_dt]
    if len(eligible_start) == 0:
        start_ix = df.index.min()
    else:
        start_ix = eligible_start.max()

    eligible_end = df.index[df.index <= end_dt]
    if len(eligible_end) == 0:
        return np.nan
    end_ix = eligible_end.max()

    window_df = df[(df.index >= start_ix) & (df.index <= end_ix)]
    prices = _price_series(window_df).dropna()
    if prices.empty:
        return np.nan

    initial_price = float(prices.iloc[0])
    end_price = float(prices.iloc[-1])
    return (end_price - initial_price) / initial_price


def get_return(input_df: pd.DataFrame, days_ago: int, most_recent_date: str) -> float:
    """
    Calculates the return over a specified number of days ending on a given date.
    
    Args:
        input_df: A pandas DataFrame containing the stock prices for a single stock.
        days_ago: An integer specifying the number of days to go back from the most_recent_date.
        most_recent_date: A string in 'YYYY-MM-DD' format representing the end date of the time period.
        
    Returns:
        A float representing the percentage return over the specified time period.
    """
    # Convert most_recent_date to a datetime object
    max_date = datetime.strptime(most_recent_date, '%Y-%m-%d')

    # Calculate the initial date by subtracting days_ago from max_date
    initial_date = (max_date - timedelta(days=days_ago)).strftime("%Y-%m-%d")

    # Find the index of the most recent date on or before initial_date
    initial_index = max(input_df[input_df.index<=initial_date].index)

    # Find the index of the most recent date on or before max_date
    max_index = max(input_df[input_df.index<=max_date.strftime('%Y-%m-%d')].index)

    # Subset the DataFrame to the time period of interest
    subset_df = input_df[(input_df.index>=initial_index) & (input_df.index<=max_index)]

    # Calculate the return as the percentage difference between the initial and ending prices
    prices = _price_series(subset_df).dropna()
    if prices.empty:
        return np.nan

    end_price = prices.iloc[-1]
    initial_price = prices.iloc[0]
    return_pct = (end_price - initial_price)/initial_price

    return return_pct


def get_returns_df_by_lookback_period(input_df: pd.DataFrame, max_date: str) -> pd.DataFrame:
    """
    Calculates the returns for a given stock over a range of lookback periods.
    
    Args:
        input_df: A pandas DataFrame containing the stock prices for a single stock.
        max_date: A string in 'YYYY-MM-DD' format representing the end date of the time period.
        
    Returns:
        A pandas DataFrame containing the returns for the stock over a range of lookback periods.
        The DataFrame has three columns: 'Stock Name', 'lookback_periods', and 'returns'.
    """
    # Generate an array of lookback periods
    lookback_periods = np.arange(10, len(input_df), 10)
    
    # Calculate the returns for each lookback period
    returns = []
    for x in lookback_periods: 
        returns.append(get_return(input_df, int(x), max_date))
    
    # Create a DataFrame with the results
    results_df = pd.DataFrame({
        'Stock Name': [input_df['Stock_name'].unique()[0]]*len(lookback_periods),
        'lookback_periods': lookback_periods,
        'returns': returns
    })

    return results_df


def get_returns_table(
    prices: pd.DataFrame | dict,
    tickers: list[str],
    horizons: list[str],
    most_recent_date: Optional[str] = None,
    stock_col: str = 'Stock_name',
) -> pd.DataFrame:
    """Compute a wide returns table for multiple tickers and horizons.

    Args:
        prices: Either (a) a single DataFrame containing multiple tickers with a
            stock identifier column (default 'Stock_name'), or (b) a dict mapping
            ticker -> per-ticker DataFrame.
        tickers: List of ticker symbols to include (one row per ticker).
        horizons: List of lookback horizons like ['6M','1Y','3Y'] (one column per horizon).
        most_recent_date: Optional end date ('YYYY-MM-DD'). If omitted, uses per-ticker
            latest date available.
        stock_col: Column name in `prices` used to identify tickers (when `prices` is a DataFrame).

    Returns:
        DataFrame indexed by ticker with columns equal to `horizons`.
    """
    if tickers is None or len(tickers) == 0:
        raise ValueError('tickers must be a non-empty list')
    if horizons is None or len(horizons) == 0:
        raise ValueError('horizons must be a non-empty list')

    rows = []
    for ticker in tickers:
        if isinstance(prices, dict):
            ticker_df = prices.get(ticker, pd.DataFrame())
        else:
            if stock_col not in prices.columns:
                raise KeyError(f"prices DataFrame must contain column '{stock_col}'")
            ticker_df = prices[prices[stock_col] == ticker]

        row = {'Ticker': ticker}
        for horizon in horizons:
            try:
                row[horizon] = get_return_by_lookback(
                    ticker_df,
                    horizon,
                    most_recent_date=most_recent_date,
                )
            except Exception:
                # Keep the table shape stable; failures become NaN.
                row[horizon] = np.nan
        rows.append(row)

    out = pd.DataFrame(rows).set_index('Ticker')
    # Ensure column order matches requested horizons
    out = out[horizons]
    return out


def get_stddev_by_lookback(
    input_df: pd.DataFrame,
    lookback: str,
    most_recent_date: Optional[str] = None,
) -> float:
    """Compute standard deviation of daily returns over a lookback window.

    The window selection matches `get_return_by_lookback` (snap to closest
    available trading dates on-or-before the target start/end). Prices are taken
    from 'Close' when available, falling back to 'Price' for null/missing Close.

    Args:
        input_df: DataFrame for a single stock with date-like index.
        lookback: String lookback like '30D', '6M', '1Y'.
        most_recent_date: Optional end date ('YYYY-MM-DD'). If omitted, uses
            the latest date available in input_df.

    Returns:
        Std dev of daily percentage returns within the window. Returns np.nan if
        insufficient data.
    """
    if input_df.empty:
        return np.nan

    df = input_df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    end_dt = pd.to_datetime(most_recent_date) if most_recent_date else df.index.max()

    value, unit = _parse_lookback(lookback)
    if unit == 'D':
        start_dt = end_dt - pd.Timedelta(days=value)
    elif unit == 'M':
        start_dt = end_dt - pd.DateOffset(months=value)
    elif unit == 'Y':
        start_dt = end_dt - pd.DateOffset(years=value)
    else:
        raise ValueError("Unsupported lookback unit")

    eligible_start = df.index[df.index <= start_dt]
    if len(eligible_start) == 0:
        start_ix = df.index.min()
    else:
        start_ix = eligible_start.max()

    eligible_end = df.index[df.index <= end_dt]
    if len(eligible_end) == 0:
        return np.nan
    end_ix = eligible_end.max()

    window_df = df[(df.index >= start_ix) & (df.index <= end_ix)]
    prices = _price_series(window_df).dropna()
    if len(prices) < 2:
        return np.nan

    daily_returns = prices.pct_change().dropna()
    if daily_returns.empty:
        return np.nan
    return float(daily_returns.std(ddof=1))


def get_stddev_table(
    prices: pd.DataFrame | dict,
    tickers: list[str],
    horizons: list[str],
    most_recent_date: Optional[str] = None,
    stock_col: str = 'Stock_name',
) -> pd.DataFrame:
    """Compute a wide std-dev table for multiple tickers and horizons.

    Inputs mirror `get_returns_table`.

    Returns:
        DataFrame indexed by ticker with columns equal to `horizons`.
    """
    if tickers is None or len(tickers) == 0:
        raise ValueError('tickers must be a non-empty list')
    if horizons is None or len(horizons) == 0:
        raise ValueError('horizons must be a non-empty list')

    rows = []
    for ticker in tickers:
        if isinstance(prices, dict):
            ticker_df = prices.get(ticker, pd.DataFrame())
        else:
            if stock_col not in prices.columns:
                raise KeyError(f"prices DataFrame must contain column '{stock_col}'")
            ticker_df = prices[prices[stock_col] == ticker]

        row = {'Ticker': ticker}
        for horizon in horizons:
            try:
                row[horizon] = get_stddev_by_lookback(
                    ticker_df,
                    horizon,
                    most_recent_date=most_recent_date,
                )
            except Exception:
                row[horizon] = np.nan
        rows.append(row)

    out = pd.DataFrame(rows).set_index('Ticker')
    out = out[horizons]
    return out
