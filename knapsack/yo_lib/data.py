import pandas as pd


def get_baggage() -> pd.DataFrame:
    """
    Returns the contents of baggage.csv as a Pandas DataFrame
    """
    return pd.read_csv("knapsack/yo_lib/baggage.csv")
