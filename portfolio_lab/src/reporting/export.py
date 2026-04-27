"""Export tables and figures to disk.

All functions accept a full Path object. Parent directories are created
automatically if they do not exist, so callers do not need to pre-create
output directories.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)


def save_table(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame as a CSV file.

    Parent directories are created automatically.

    Args:
        df: DataFrame to save.
        path: Full destination path, e.g. Path('outputs/tables/risk.csv').

    Raises:
        ValueError: If df is empty.
        OSError: If the file cannot be written (permissions, disk full, etc.).
    """
    path = Path(path)

    if df.empty:
        raise ValueError(f"save_table: DataFrame is empty, not saving to {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    logger.info(f"Table saved -> {path}  ({df.shape[0]} rows x {df.shape[1]} cols)")


def save_figure(
    fig: plt.Figure,
    path: Path,
    dpi: int = 150,
) -> None:
    """Save a matplotlib Figure as a PNG file and close it.

    Parent directories are created automatically.
    The figure is closed after saving to release memory.

    Args:
        fig: matplotlib Figure object to save.
        path: Full destination path, e.g. Path('outputs/figures/hist.png').
        dpi: Resolution in dots per inch (default 150).

    Raises:
        OSError: If the file cannot be written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure saved -> {path}  (dpi={dpi})")
