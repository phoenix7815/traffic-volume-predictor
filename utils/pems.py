import numpy as np
import pandas as pd

def create_index(start_date, end_date, timestamp_window) -> np.ndarray:
    """
    Create timestamp index from start_date to end_date with given window.

    Args:
        start_date: Start date string (e.g., '2016-07-01 08:00:00')
        end_date: End date string (e.g., '2016-07-01 13:00:00')
        timestamp_window: Frequency string (e.g., '1H' for hourly, '30min' for 30 minutes)

    Returns:
        np.ndarray: Array of timestamp strings
    """
    timestamps = pd.date_range(start=start_date, end=end_date, freq=timestamp_window)
    return np.array([str(ts) for ts in timestamps])
