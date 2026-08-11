import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.config import ensure_dir


def save_residual_plot(y_true: pd.Series, y_pred: np.ndarray, path: str) -> None:
    """Save a residuals-vs-predicted scatter plot to the given path."""
    residuals = y_true.values - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.3)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted Price")
    plt.ylabel("Residual")
    plt.title("Residuals vs Predicted Price")
    ensure_dir(path)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def save_price_distribution(prices: pd.Series, path: str) -> None:
    """Save a histogram of selling prices to the given path."""
    plt.figure(figsize=(8, 6))
    plt.hist(prices, bins=50, alpha=0.7)
    plt.xlabel("Selling Price")
    plt.ylabel("Frequency")
    plt.title("Selling Price Distribution")
    ensure_dir(path)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
