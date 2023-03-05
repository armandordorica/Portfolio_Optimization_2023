from datetime import datetime, timedelta
import pandas as pd
import numpy as np


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
    end_price = subset_df['Close'].iloc[-1]
    initial_price = subset_df['Close'].iloc[0]
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
