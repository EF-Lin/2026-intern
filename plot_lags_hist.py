import pandas as pd

from src.plot import plot_hist
from src.utils import load_config

if __name__ == "__main__":
    cfg = load_config()["hist"]
    plot_hist(pd.read_csv(cfg["csv"]), "lags_AB", highlight_time=cfg["highlight_time"], **cfg["lags_AB"])
