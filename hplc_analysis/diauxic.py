"""Diauxic shift analysis for multi-substrate consumption."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .kinetics import SubstrateKinetics


@dataclass
class DiauxicAnalysis:
    """Cross-substrate diauxic consumption analysis."""
    diauxic_detected: bool
    primary_substrate: Optional[str]
    secondary_substrate: Optional[str]
    t_diauxic_lag: Optional[float]  # gap between primary depletion and secondary active start
    overlap_fraction: Optional[float]  # fraction of time both substrates consumed simultaneously
    consumption_order: List[str]
    active_windows: Dict[str, tuple]  # substrate -> (t_start, t_end) of active consumption


def _find_active_window(
    kinetics: SubstrateKinetics, rate_threshold_frac: float = 0.1
) -> tuple:
    """Find the active consumption window for a substrate.

    Active = rate > rate_threshold_frac * q_max.

    Returns:
        (t_start, t_end) of active consumption window.
    """
    threshold = rate_threshold_frac * kinetics.q_max
    active_mask = kinetics.rates > threshold

    if not np.any(active_mask):
        return (kinetics.times[0], kinetics.times[0])

    active_indices = np.where(active_mask)[0]
    t_start = float(kinetics.times[active_indices[0]])
    t_end = float(kinetics.times[active_indices[-1]])

    return (t_start, t_end)


def analyze_diauxic(
    kinetics_dict: Dict[str, SubstrateKinetics],
) -> DiauxicAnalysis:
    """Analyze diauxic consumption patterns across substrates.

    Args:
        kinetics_dict: Mapping of substrate name to its kinetics results.

    Returns:
        DiauxicAnalysis with cross-substrate pattern information.
    """
    if len(kinetics_dict) < 2:
        return DiauxicAnalysis(
            diauxic_detected=False,
            primary_substrate=None,
            secondary_substrate=None,
            t_diauxic_lag=None,
            overlap_fraction=None,
            consumption_order=list(kinetics_dict.keys()),
            active_windows={},
        )

    # Find active windows for each substrate
    active_windows = {}
    for name, kin in kinetics_dict.items():
        active_windows[name] = _find_active_window(kin)

    # Determine consumption order by depletion time (or end of active window)
    order_key = {}
    for name, kin in kinetics_dict.items():
        if kin.t_depletion is not None:
            order_key[name] = kin.t_depletion
        else:
            order_key[name] = active_windows[name][1]

    consumption_order = sorted(order_key, key=lambda k: order_key[k])
    primary = consumption_order[0]
    secondary = consumption_order[1]

    # Get the active windows
    p_start, p_end = active_windows[primary]
    s_start, s_end = active_windows[secondary]

    # Diauxic lag: time between primary depletion and secondary active start
    primary_depletion = kinetics_dict[primary].t_depletion
    if primary_depletion is not None:
        t_diauxic_lag = max(0.0, s_start - primary_depletion)
    else:
        t_diauxic_lag = max(0.0, s_start - p_end)

    # Overlap: fraction of time both substrates are actively consumed
    overlap_start = max(p_start, s_start)
    overlap_end = min(p_end, s_end)
    overlap_duration = max(0.0, overlap_end - overlap_start)

    total_duration = max(p_end, s_end) - min(p_start, s_start)
    overlap_fraction = overlap_duration / total_duration if total_duration > 0 else 0.0

    # Diauxic shift detected if there's clear sequential consumption
    # (primary mostly consumed before secondary ramps up significantly)
    diauxic_detected = (
        primary_depletion is not None
        and kinetics_dict[primary].efficiency > 80
        and kinetics_dict[secondary].efficiency > 50
    )

    return DiauxicAnalysis(
        diauxic_detected=diauxic_detected,
        primary_substrate=primary,
        secondary_substrate=secondary,
        t_diauxic_lag=t_diauxic_lag,
        overlap_fraction=overlap_fraction,
        consumption_order=consumption_order,
        active_windows=active_windows,
    )
