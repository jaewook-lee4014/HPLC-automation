"""Replicate statistics, outlier detection, and quality metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


@dataclass
class QualityReport:
    """Quality assessment for a substrate's time series."""
    substrate: str
    max_cv: float  # maximum CV across timepoints (%)
    max_cv_time: float  # time at which max CV occurs
    high_cv_times: List[float]  # timepoints where CV > 20%
    monotonicity_score: float  # fraction of intervals with decreasing conc
    increasing_intervals: List[tuple]  # (t_start, t_end, delta) for increases


def compute_replicate_stats(
    df: pd.DataFrame,
    substrate_col: str,
) -> pd.DataFrame:
    """Compute per-timepoint statistics from replicate measurements.

    Args:
        df: DataFrame with 'Time (h)' and the substrate column.
        substrate_col: Name of the substrate column (e.g. 'Glucose (g/L)').

    Returns:
        DataFrame indexed by time with columns:
        mean, std, sem, n, cv_pct, ci95_lower, ci95_upper
    """
    grouped = df.groupby('Time (h)')[substrate_col]

    result = pd.DataFrame({
        'mean': grouped.mean(),
        'std': grouped.std(ddof=1),
        'n': grouped.size(),
    })

    # For single-replicate timepoints, std is NaN → set to 0
    result['std'] = result['std'].fillna(0.0)
    result['sem'] = result['std'] / np.sqrt(result['n'])

    # CV (%) — avoid division by zero
    result['cv_pct'] = np.where(
        result['mean'] > 1e-6,
        (result['std'] / result['mean']) * 100,
        0.0,
    )

    # 95% CI using t-distribution
    t_crit = np.where(
        result['n'] > 1,
        sp_stats.t.ppf(0.975, result['n'] - 1),
        np.nan,
    )
    margin = t_crit * result['sem']
    result['ci95_lower'] = result['mean'] - margin
    result['ci95_upper'] = result['mean'] + margin

    # For n=1, CI is just the mean
    mask_n1 = result['n'] == 1
    result.loc[mask_n1, 'ci95_lower'] = result.loc[mask_n1, 'mean']
    result.loc[mask_n1, 'ci95_upper'] = result.loc[mask_n1, 'mean']

    return result


def detect_outliers(
    df: pd.DataFrame,
    substrate_col: str,
    threshold: float = 3.5,
) -> pd.DataFrame:
    """Flag outlier measurements using Modified Z-score (MAD-based).

    Only applied at timepoints with n >= 3 replicates.

    Args:
        df: DataFrame with 'Time (h)' and the substrate column.
        substrate_col: Name of the substrate column.
        threshold: Modified Z-score threshold (default 3.5).

    Returns:
        DataFrame of flagged outliers with columns:
        Time (h), value, modified_z_score, replicate_index
    """
    outliers = []

    for time, group in df.groupby('Time (h)'):
        values = group[substrate_col].values
        if len(values) < 3:
            continue

        median = np.median(values)
        mad = np.median(np.abs(values - median))

        if mad < 1e-10:
            continue

        modified_z = 0.6745 * (values - median) / mad

        for i, (val, mz) in enumerate(zip(values, modified_z)):
            if abs(mz) > threshold:
                outliers.append({
                    'Time (h)': time,
                    'value': val,
                    'modified_z_score': mz,
                    'replicate_index': i,
                })

    return pd.DataFrame(outliers)


def compute_quality_report(
    stats_df: pd.DataFrame,
    substrate: str,
) -> QualityReport:
    """Assess data quality for a substrate time series.

    Args:
        stats_df: Output of compute_replicate_stats.
        substrate: Substrate name for reporting.

    Returns:
        QualityReport with CV analysis and monotonicity assessment.
    """
    times = stats_df.index.values
    means = stats_df['mean'].values
    cvs = stats_df['cv_pct'].values

    # Max CV
    max_cv_idx = np.argmax(cvs)
    max_cv = cvs[max_cv_idx]
    max_cv_time = times[max_cv_idx]

    # High CV timepoints (>20%)
    high_cv_times = times[cvs > 20.0].tolist()

    # Monotonicity: fraction of intervals where concentration decreases
    diffs = np.diff(means)
    n_intervals = len(diffs)
    n_decreasing = np.sum(diffs < 0)
    monotonicity_score = n_decreasing / n_intervals if n_intervals > 0 else 1.0

    # Identify increasing intervals (non-physical for consumption)
    increasing = []
    for i in range(len(diffs)):
        if diffs[i] > 0.01:  # threshold to avoid noise
            increasing.append((times[i], times[i + 1], float(diffs[i])))

    return QualityReport(
        substrate=substrate,
        max_cv=max_cv,
        max_cv_time=max_cv_time,
        high_cv_times=high_cv_times,
        monotonicity_score=monotonicity_score,
        increasing_intervals=increasing,
    )
