"""Data loading, column auto-detection, and validation for HPLC CSV files."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import pandas as pd


# Regex patterns for column auto-detection
_TIME_PATTERN = re.compile(r'[Tt]ime.*\(h\)')
_SUBSTRATE_PATTERN = re.compile(r'\[(.+?)\]\s*\(g/L\)')
_OD_PATTERN = re.compile(r'OD\d*|optical\s*density', re.IGNORECASE)

# Known non-substrate columns to skip
_SKIP_SUBSTRATES = {'OD600', 'OD', 'Biomass'}


@dataclass
class DatasetMetadata:
    """Metadata extracted from an HPLC CSV file."""
    filepath: str
    time_column: str
    substrate_columns: Dict[str, str]  # substrate_name -> original_col_name
    ignored_columns: List[str]
    n_rows: int
    n_timepoints: int
    time_range: tuple  # (min_h, max_h)
    replicates_per_timepoint: Dict[float, int]


def _detect_columns(columns: List[str]) -> tuple:
    """Auto-detect time and substrate columns from header names.

    Returns:
        (time_col, substrate_map, ignored_cols) where substrate_map is
        {substrate_name: original_column_name}.
    """
    time_col = None
    substrate_map = {}
    ignored = []

    for col in columns:
        col_stripped = col.strip()

        # Check for time column
        if _TIME_PATTERN.search(col_stripped):
            time_col = col
            continue

        # Check for substrate columns: [Name] (g/L)
        m = _SUBSTRATE_PATTERN.search(col_stripped)
        if m:
            substrate_name = m.group(1).strip()
            if substrate_name in _SKIP_SUBSTRATES:
                ignored.append(col)
                continue
            substrate_map[substrate_name] = col
            continue

        # Check for OD / biomass columns to ignore
        if _OD_PATTERN.search(col_stripped):
            ignored.append(col)
            continue

        ignored.append(col)

    if time_col is None:
        raise ValueError(
            f"Could not detect time column. Expected pattern matching "
            f"'{_TIME_PATTERN.pattern}'. Found columns: {columns}"
        )
    if not substrate_map:
        raise ValueError(
            f"No substrate columns detected. Expected pattern matching "
            f"'{_SUBSTRATE_PATTERN.pattern}'. Found columns: {columns}"
        )

    return time_col, substrate_map, ignored


def _validate(df: pd.DataFrame, time_col: str,
              substrate_cols: Dict[str, str]) -> List[str]:
    """Validate loaded data. Returns list of warnings."""
    warnings = []

    # Check for NaN
    nan_counts = df.isna().sum()
    for col, cnt in nan_counts.items():
        if cnt > 0:
            warnings.append(f"Column '{col}' has {cnt} NaN values")

    # Check non-negative concentrations
    for name, col in substrate_cols.items():
        neg = (df[col] < 0).sum()
        if neg > 0:
            warnings.append(
                f"Substrate '{name}' has {neg} negative concentration values"
            )

    # Check time is sorted (within tolerance for replicates)
    times = df[time_col].values
    if not all(times[i] <= times[i + 1] + 1e-9 for i in range(len(times) - 1)):
        warnings.append("Time column is not monotonically non-decreasing")

    return warnings


def load_hplc_csv(
    filepath: str | Path,
    encoding: str = 'utf-8-sig',
) -> tuple[pd.DataFrame, DatasetMetadata]:
    """Load an HPLC CSV file with automatic column detection.

    Args:
        filepath: Path to the CSV file.
        encoding: File encoding (default handles BOM).

    Returns:
        (df, metadata) where df contains only time + substrate columns
        with cleaned column names, and metadata describes the dataset.
    """
    filepath = Path(filepath)
    raw = pd.read_csv(filepath, encoding=encoding)

    # Strip whitespace from column names
    raw.columns = [c.strip() for c in raw.columns]

    # Detect columns
    time_col, substrate_map, ignored = _detect_columns(list(raw.columns))

    # Validate
    warnings = _validate(raw, time_col, substrate_map)
    for w in warnings:
        print(f"  WARNING [{filepath.name}]: {w}")

    # Sort by time
    raw = raw.sort_values(time_col).reset_index(drop=True)

    # Build clean DataFrame: rename to standard names
    rename = {time_col: 'Time (h)'}
    for name, col in substrate_map.items():
        rename[col] = f'{name} (g/L)'

    keep_cols = [time_col] + list(substrate_map.values())
    df = raw[keep_cols].rename(columns=rename).copy()

    # Compute replicates per timepoint
    reps = df.groupby('Time (h)').size().to_dict()

    metadata = DatasetMetadata(
        filepath=str(filepath),
        time_column=time_col,
        substrate_columns=substrate_map,
        ignored_columns=ignored,
        n_rows=len(df),
        n_timepoints=df['Time (h)'].nunique(),
        time_range=(df['Time (h)'].min(), df['Time (h)'].max()),
        replicates_per_timepoint=reps,
    )

    return df, metadata
