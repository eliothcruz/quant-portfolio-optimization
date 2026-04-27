"""Diagnostics for return series quality and temporal alignment.

Placeholder module — scheduled for Phase 2.

Planned functions:
- check_series_alignment: verify all series share the same DatetimeIndex
- check_stationarity: basic stationarity assessment (ADF or visual)
"""

import pandas as pd


def check_series_alignment(returns: pd.DataFrame) -> dict:
    """Verify that all return series share the same date index.

    Placeholder — Phase 2.
    After align_to_common_period this should always pass, but
    is useful as a defensive check before optimization.
    """
    raise NotImplementedError("diagnostics.check_series_alignment: Phase 2")


def check_stationarity(returns: pd.DataFrame) -> dict:
    """Run basic stationarity checks on return series.

    Placeholder — Phase 2.
    Returns are typically stationary by construction, but this is
    useful to verify and document for each dataset.
    """
    raise NotImplementedError("diagnostics.check_stationarity: Phase 2")
