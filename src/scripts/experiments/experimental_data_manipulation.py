"""
Copyright (C) 2022  Konstantinos Stavratis
For the full notice of the program, see "main.py"
"""

import pandas as pd

def distance_metric_mean(df: pd.DataFrame):
    return df.mean(axis=1)