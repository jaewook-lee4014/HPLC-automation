"""Pipeline orchestrator: load → stats → kinetics → diauxic → export."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .loader import load_hplc_csv, DatasetMetadata
from .stats import (
    compute_replicate_stats,
    detect_outliers,
    compute_quality_report,
    QualityReport,
)
from .kinetics import compute_kinetics, SubstrateKinetics
from .diauxic import analyze_diauxic, DiauxicAnalysis


@dataclass
class ExperimentResults:
    """Complete analysis results for one experiment."""
    metadata: DatasetMetadata
    raw_df: pd.DataFrame
    stats: Dict[str, pd.DataFrame]          # substrate -> stats DataFrame
    outliers: Dict[str, pd.DataFrame]       # substrate -> outlier DataFrame
    quality: Dict[str, QualityReport]       # substrate -> quality report
    kinetics: Dict[str, SubstrateKinetics]  # substrate -> kinetics
    diauxic: DiauxicAnalysis


def analyze_experiment(
    filepath: str | Path,
    depletion_thresholds: Optional[Dict[str, float]] = None,
) -> ExperimentResults:
    """Run the full analysis pipeline on a single HPLC CSV file.

    Args:
        filepath: Path to the CSV file.
        depletion_thresholds: Optional per-substrate depletion thresholds.

    Returns:
        ExperimentResults with all analysis outputs.
    """
    depletion_thresholds = depletion_thresholds or {}

    # Step 1: Load
    df, metadata = load_hplc_csv(filepath)

    # Step 2: Stats + outliers + quality per substrate
    stats_dict = {}
    outlier_dict = {}
    quality_dict = {}
    kinetics_dict = {}

    for substrate, orig_col in metadata.substrate_columns.items():
        col = f'{substrate} (g/L)'

        # Replicate statistics
        st = compute_replicate_stats(df, col)
        stats_dict[substrate] = st

        # Outlier detection
        ol = detect_outliers(df, col)
        outlier_dict[substrate] = ol

        # Quality report
        qr = compute_quality_report(st, substrate)
        quality_dict[substrate] = qr

        # Step 3: Kinetics
        times = st.index.values
        means = st['mean'].values
        threshold = depletion_thresholds.get(substrate, None)
        kin = compute_kinetics(times, means, substrate, threshold)
        kinetics_dict[substrate] = kin

    # Step 4: Diauxic analysis
    diauxic = analyze_diauxic(kinetics_dict)

    return ExperimentResults(
        metadata=metadata,
        raw_df=df,
        stats=stats_dict,
        outliers=outlier_dict,
        quality=quality_dict,
        kinetics=kinetics_dict,
        diauxic=diauxic,
    )


def results_to_summary_df(
    results: ExperimentResults,
    experiment_label: Optional[str] = None,
) -> pd.DataFrame:
    """Convert experiment results to a single-row summary DataFrame."""
    row = {}

    if experiment_label:
        row['experiment'] = experiment_label
    else:
        row['experiment'] = Path(results.metadata.filepath).stem

    for substrate, kin in results.kinetics.items():
        prefix = substrate
        row[f'{prefix}_S0'] = kin.S0
        row[f'{prefix}_Se'] = kin.Se
        row[f'{prefix}_delta_S'] = kin.delta_S
        row[f'{prefix}_efficiency_pct'] = kin.efficiency
        row[f'{prefix}_q_avg'] = kin.q_avg
        row[f'{prefix}_q_max'] = kin.q_max
        row[f'{prefix}_t_qmax'] = kin.t_qmax
        row[f'{prefix}_t_lag'] = kin.t_lag
        row[f'{prefix}_t_depletion'] = kin.t_depletion
        row[f'{prefix}_t_50'] = kin.t_50
        row[f'{prefix}_t_90'] = kin.t_90
        row[f'{prefix}_t_active'] = kin.t_active
        row[f'{prefix}_q_active'] = kin.q_active
        row[f'{prefix}_max_cv_pct'] = results.quality[substrate].max_cv
        row[f'{prefix}_monotonicity'] = results.quality[substrate].monotonicity_score

    # Diauxic parameters
    d = results.diauxic
    row['diauxic_detected'] = d.diauxic_detected
    row['primary_substrate'] = d.primary_substrate
    row['consumption_order'] = ' -> '.join(d.consumption_order)
    row['t_diauxic_lag'] = d.t_diauxic_lag
    row['overlap_fraction'] = d.overlap_fraction

    return pd.DataFrame([row])


def results_to_timeseries_df(
    results: ExperimentResults,
    experiment_label: Optional[str] = None,
) -> pd.DataFrame:
    """Convert experiment results to a detailed time-series DataFrame."""
    frames = []

    for substrate, st in results.stats.items():
        kin = results.kinetics[substrate]
        ts_df = st.copy()
        ts_df['substrate'] = substrate
        ts_df['consumption_rate'] = np.interp(
            ts_df.index.values, kin.times, kin.rates
        )
        # Normalized consumption: (S0 - S(t)) / delta_S
        if kin.delta_S > 0:
            ts_df['normalized_consumption'] = (kin.S0 - ts_df['mean']) / kin.delta_S
        else:
            ts_df['normalized_consumption'] = 0.0

        frames.append(ts_df.reset_index())

    combined = pd.concat(frames, ignore_index=True)

    label = experiment_label or Path(results.metadata.filepath).stem
    combined.insert(0, 'experiment', label)

    return combined


def export_results(
    results: ExperimentResults,
    output_dir: str | Path = 'output',
    experiment_label: Optional[str] = None,
) -> tuple[Path, Path]:
    """Export analysis results to CSV files.

    Args:
        results: ExperimentResults from analyze_experiment.
        output_dir: Directory for output files.
        experiment_label: Optional label for the experiment.

    Returns:
        (summary_path, timeseries_path) paths to created files.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    label = experiment_label or Path(results.metadata.filepath).stem
    safe_label = label.replace(' ', '_').replace('/', '_')

    summary = results_to_summary_df(results, label)
    summary_path = out / f'{safe_label}_summary.csv'
    summary.to_csv(summary_path, index=False)

    timeseries = results_to_timeseries_df(results, label)
    ts_path = out / f'{safe_label}_timeseries.csv'
    timeseries.to_csv(ts_path, index=False)

    return summary_path, ts_path


def batch_analyze(
    filepaths: List[str | Path],
    output_dir: str | Path = 'output',
    labels: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Analyze multiple experiments and export combined summary.

    Args:
        filepaths: List of CSV file paths.
        output_dir: Directory for output files.
        labels: Optional experiment labels (same length as filepaths).

    Returns:
        Combined summary DataFrame.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    labels = labels or [None] * len(filepaths)

    for fp, label in zip(filepaths, labels):
        print(f"\nAnalyzing: {fp}")
        results = analyze_experiment(fp)
        export_results(results, output_dir, label)
        all_summaries.append(results_to_summary_df(results, label))

    combined = pd.concat(all_summaries, ignore_index=True)
    combined.to_csv(out / 'batch_summary.csv', index=False)

    return combined
