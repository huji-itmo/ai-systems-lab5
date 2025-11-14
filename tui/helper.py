import numpy as np
import pandas as pd


def sanitize_id(text: str) -> str:
    """Convert a string into a valid Textual widget ID."""
    # Replace spaces and invalid chars with underscores
    sanitized = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in text)
    # Ensure it doesn't start with a digit
    if sanitized and sanitized[0].isdigit():
        sanitized = "_" + sanitized
    return sanitized


def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(probs * np.log2(probs + 1e-9))  # add small value to avoid log(0)


def info(feature, y):
    unique_vals = np.unique(feature)
    info_val = 0
    for val in unique_vals:
        subset_y = y[feature == val]
        info_val += len(subset_y) / len(y) * entropy(subset_y)
    return info_val


def information_gain(feature, y):
    return entropy(y) - info(feature, y)


def split_info(feature):
    _, counts = np.unique(feature, return_counts=True)
    probs = counts / len(feature)
    return -np.sum(probs * np.log2(probs + 1e-9))


def gain_ratio(feature, y):
    ig = information_gain(feature, y)
    si = split_info(feature)
    return ig / si if si > 1e-9 else 0
