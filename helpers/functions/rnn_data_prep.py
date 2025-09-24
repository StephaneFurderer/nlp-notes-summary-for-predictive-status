"""
Prepare data for RNN model
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any

def split_data_for_rnn(data, split_ratio: List[float] = [0.8, 0.1, 0.1]):
    """
    Elegantly split data into training, validation, and testing sets.

    Args:
        data (array-like or pd.DataFrame): The dataset to split.
        split_ratio (list of float): Proportions for train, val, and test splits. Must sum to 1.

    Returns:
        tuple: (train_data, val_data, test_data)
    """
    if not np.isclose(sum(split_ratio), 1.0):
        raise ValueError("Split ratios must sum to 1.")
    n = len(data)
    train_end = int(split_ratio[0] * n)
    val_end = train_end + int(split_ratio[1] * n)
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    return train_data, val_data, test_data

