"""Visualization functions for HPLC substrate analysis."""

from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from .pipeline import ExperimentResults


# Consistent color scheme
SUBSTRATE_COLORS = {
    'Glucose': '#2196F3',
    'Xylose': '#FF9800',
}

def _get_color(substrate: str) -> str:
    return SUBSTRATE_COLORS.get(substrate, '#4CAF50')


def plot_consumption_curves(
    results: ExperimentResults,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    show_replicates: bool = True,
) -> plt.Axes:
    """Plot substrate consumption curves with mean +/- 95% CI.

    Shows replicate scatter, depletion time vertical lines.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    for substrate, st in results.stats.items():
        color = _get_color(substrate)
        times = st.index.values
        means = st['mean'].values

        # CI band
        ax.fill_between(
            times, st['ci95_lower'], st['ci95_upper'],
            alpha=0.2, color=color,
        )
        # Mean line
        ax.plot(times, means, 'o-', color=color, label=substrate, linewidth=2,
                markersize=5)

        # Replicate scatter
        if show_replicates:
            raw = results.raw_df
            col = f'{substrate} (g/L)'
            ax.scatter(
                raw['Time (h)'], raw[col],
                color=color, alpha=0.3, s=15, zorder=1,
            )

        # Depletion line
        kin = results.kinetics[substrate]
        if kin.t_depletion is not None:
            ax.axvline(kin.t_depletion, color=color, linestyle='--', alpha=0.6,
                       label=f'{substrate} depletion ({kin.t_depletion:.1f}h)')

    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Concentration (g/L)')
    ax.set_title(title or 'Substrate Consumption Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_consumption_rates(
    results: ExperimentResults,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    """Plot consumption rate vs time for each substrate.

    Shows q_max marker and active-phase shading.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    for substrate, kin in results.kinetics.items():
        color = _get_color(substrate)

        ax.plot(kin.times, kin.rates, 'o-', color=color, label=substrate,
                linewidth=2, markersize=5)

        # q_max marker
        ax.plot(kin.t_qmax, kin.q_max, '*', color=color, markersize=15,
                zorder=5, label=f'{substrate} q_max={kin.q_max:.2f}')

        # Active phase shading
        if substrate in results.diauxic.active_windows:
            t_s, t_e = results.diauxic.active_windows[substrate]
            ax.axvspan(t_s, t_e, alpha=0.08, color=color)

    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Consumption Rate (g/L/h)')
    ax.set_title(title or 'Consumption Rate vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linewidth=0.5)
    return ax


def plot_normalized_consumption(
    results: ExperimentResults,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    """Plot normalized consumption (S0-S(t))/delta_S for cross-comparison.

    Shows 50% and 90% horizontal lines.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    for substrate, kin in results.kinetics.items():
        color = _get_color(substrate)

        if kin.delta_S > 0:
            normalized = (kin.S0 - kin.means) / kin.delta_S
            normalized = np.clip(normalized, 0, 1.05)
        else:
            normalized = np.zeros_like(kin.means)

        ax.plot(kin.times, normalized, 'o-', color=color, label=substrate,
                linewidth=2, markersize=5)

    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.6, label='50%')
    ax.axhline(0.9, color='gray', linestyle='--', alpha=0.6, label='90%')

    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Fractional Consumption')
    ax.set_title(title or 'Normalized Substrate Consumption')
    ax.set_ylim(-0.05, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_diauxic_summary(
    results: ExperimentResults,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    """Plot diauxic analysis summary showing both substrates with phase markers."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    d = results.diauxic

    for substrate, kin in results.kinetics.items():
        color = _get_color(substrate)
        ax.plot(kin.times, kin.means, 'o-', color=color, label=substrate,
                linewidth=2, markersize=5)

        # Active window shading
        if substrate in d.active_windows:
            t_s, t_e = d.active_windows[substrate]
            ax.axvspan(t_s, t_e, alpha=0.1, color=color,
                       label=f'{substrate} active phase')

    # Diauxic lag annotation
    if d.diauxic_detected and d.primary_substrate and d.t_diauxic_lag is not None:
        primary_kin = results.kinetics[d.primary_substrate]
        if primary_kin.t_depletion is not None:
            t_dep = primary_kin.t_depletion
            ax.axvspan(t_dep, t_dep + d.t_diauxic_lag, alpha=0.2,
                       color='red', label=f'Diauxic lag ({d.t_diauxic_lag:.1f}h)')

    status_text = "Diauxic shift DETECTED" if d.diauxic_detected else "No diauxic shift"
    ax.text(0.02, 0.98, status_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Concentration (g/L)')
    ax.set_title(title or 'Diauxic Analysis Summary')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    return ax


def plot_cv_variability(
    results: ExperimentResults,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    """Plot CV% vs time for each substrate with 20% threshold line."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    for substrate, st in results.stats.items():
        color = _get_color(substrate)
        ax.plot(st.index, st['cv_pct'], 'o-', color=color, label=substrate,
                linewidth=2, markersize=5)

    ax.axhline(20, color='red', linestyle='--', alpha=0.6, label='20% threshold')

    ax.set_xlabel('Time (h)')
    ax.set_ylabel('CV (%)')
    ax.set_title(title or 'Replicate Variability (CV%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_experiment_dashboard(
    results: ExperimentResults,
    title: Optional[str] = None,
) -> plt.Figure:
    """Create a 2x2 dashboard with 4 key plots for one experiment."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title or 'Experiment Dashboard', fontsize=14, fontweight='bold')

    plot_consumption_curves(results, ax=axes[0, 0])
    plot_consumption_rates(results, ax=axes[0, 1])
    plot_normalized_consumption(results, ax=axes[1, 0])
    plot_diauxic_summary(results, ax=axes[1, 1])

    fig.tight_layout()
    return fig


def plot_multi_experiment_comparison(
    results_list: List[ExperimentResults],
    labels: List[str],
    title: Optional[str] = None,
) -> plt.Figure:
    """Create a 2x2 comparison dashboard across multiple experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title or 'Multi-Experiment Comparison', fontsize=14,
                 fontweight='bold')

    linestyles = ['-', '--', ':', '-.']

    # Top-left: consumption curves comparison
    ax = axes[0, 0]
    for i, (res, label) in enumerate(zip(results_list, labels)):
        ls = linestyles[i % len(linestyles)]
        for substrate, kin in res.kinetics.items():
            color = _get_color(substrate)
            ax.plot(kin.times, kin.means, linestyle=ls, color=color,
                    linewidth=2, label=f'{label} - {substrate}')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Concentration (g/L)')
    ax.set_title('Consumption Curves')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Top-right: consumption rates comparison
    ax = axes[0, 1]
    for i, (res, label) in enumerate(zip(results_list, labels)):
        ls = linestyles[i % len(linestyles)]
        for substrate, kin in res.kinetics.items():
            color = _get_color(substrate)
            ax.plot(kin.times, kin.rates, linestyle=ls, color=color,
                    linewidth=2, label=f'{label} - {substrate}')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Rate (g/L/h)')
    ax.set_title('Consumption Rates')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom-left: normalized comparison
    ax = axes[1, 0]
    for i, (res, label) in enumerate(zip(results_list, labels)):
        ls = linestyles[i % len(linestyles)]
        for substrate, kin in res.kinetics.items():
            color = _get_color(substrate)
            if kin.delta_S > 0:
                norm = np.clip((kin.S0 - kin.means) / kin.delta_S, 0, 1.05)
            else:
                norm = np.zeros_like(kin.means)
            ax.plot(kin.times, norm, linestyle=ls, color=color,
                    linewidth=2, label=f'{label} - {substrate}')
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(0.9, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Fractional Consumption')
    ax.set_title('Normalized Consumption')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom-right: bar chart of key parameters
    ax = axes[1, 1]
    bar_data = []
    for i, (res, label) in enumerate(zip(results_list, labels)):
        for substrate, kin in res.kinetics.items():
            bar_data.append({
                'Experiment': label,
                'Substrate': substrate,
                'q_max': kin.q_max,
                't_depletion': kin.t_depletion or 0,
                'efficiency': kin.efficiency,
            })

    x = np.arange(len(bar_data))
    bar_labels = [f"{d['Experiment']}\n{d['Substrate']}" for d in bar_data]
    colors = [_get_color(d['Substrate']) for d in bar_data]
    efficiencies = [d['efficiency'] for d in bar_data]

    bars = ax.bar(x, efficiencies, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=8)
    ax.set_ylabel('Consumption Efficiency (%)')
    ax.set_title('Efficiency Comparison')
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis='y')

    for bar, eff in zip(bars, efficiencies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{eff:.1f}%', ha='center', va='bottom', fontsize=9)

    fig.tight_layout()
    return fig
