"""Substrate consumption kinetics: rates, lag phase, depletion times."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# Limit of detection — values at or below this are treated as depleted
LOD = 0.001


@dataclass
class SubstrateKinetics:
    """All kinetic parameters for a single substrate."""
    substrate: str
    S0: float          # initial concentration (g/L)
    Se: float          # final concentration (g/L)
    delta_S: float     # total consumption (g/L)
    efficiency: float  # consumption efficiency (%)
    q_avg: float       # average consumption rate (g/L/h)
    q_max: float       # maximum consumption rate (g/L/h)
    t_qmax: float      # time of maximum rate (h)
    t_lag: float       # lag phase duration (h)
    t_depletion: Optional[float]  # time substrate depleted (h)
    t_50: Optional[float]  # time to 50% consumption (h)
    t_90: Optional[float]  # time to 90% consumption (h)
    t_active: Optional[float]  # active consumption period (h)
    q_active: Optional[float]  # active-phase rate (g/L/h)
    # Arrays for plotting
    times: np.ndarray        # time points
    means: np.ndarray        # mean concentrations
    rates: np.ndarray        # consumption rates (positive = consuming)


def _finite_difference_rates(t: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Compute dS/dt using non-uniform central finite differences.

    For interior points:
        h1 = t[i] - t[i-1], h2 = t[i+1] - t[i]
        dS/dt = -h2/(h1*(h1+h2))*S[i-1] + (h2-h1)/(h1*h2)*S[i] + h1/(h2*(h1+h2))*S[i+1]

    Boundary points use forward/backward differences.
    Returns consumption_rate = -dS/dt (positive when substrate is consumed).
    """
    n = len(t)
    dsdt = np.zeros(n)

    if n < 2:
        return dsdt

    # Forward difference for first point
    h = t[1] - t[0]
    if h > 0:
        dsdt[0] = (S[1] - S[0]) / h

    # Central difference for interior points
    for i in range(1, n - 1):
        h1 = t[i] - t[i - 1]
        h2 = t[i + 1] - t[i]
        if h1 > 0 and h2 > 0:
            dsdt[i] = (
                -h2 / (h1 * (h1 + h2)) * S[i - 1]
                + (h2 - h1) / (h1 * h2) * S[i]
                + h1 / (h2 * (h1 + h2)) * S[i + 1]
            )

    # Backward difference for last point
    h = t[-1] - t[-2]
    if h > 0:
        dsdt[-1] = (S[-1] - S[-2]) / h

    # Consumption rate: positive means substrate is being consumed
    return -dsdt


def _interpolate_time(
    t: np.ndarray, S: np.ndarray, target_S: float
) -> Optional[float]:
    """Find time at which concentration crosses target_S via linear interpolation.

    Searches for the first downward crossing (S going from above to below target).
    """
    for i in range(len(S) - 1):
        if S[i] >= target_S and S[i + 1] < target_S:
            # Linear interpolation
            frac = (S[i] - target_S) / (S[i] - S[i + 1])
            return t[i] + frac * (t[i + 1] - t[i])
    # Check if already below at start or never reaches
    if len(S) > 0 and S[-1] <= target_S:
        # Already depleted at last timepoint — return last time
        return float(t[np.argmax(S <= target_S)])
    return None


def _compute_lag(
    t: np.ndarray, S: np.ndarray, rates: np.ndarray, S0: float
) -> float:
    """Compute lag phase using tangent-line intersection method.

    Draws a tangent at the point of maximum consumption rate and finds
    where it intersects the S = S0 horizontal line.
    """
    if len(rates) == 0 or np.max(rates) <= 0:
        return 0.0

    i_max = np.argmax(rates)
    t_max = t[i_max]
    S_max = S[i_max]
    slope = -rates[i_max]  # dS/dt is negative during consumption

    if abs(slope) < 1e-10:
        return 0.0

    # Tangent line: S(t) = S_max + slope * (t - t_max)
    # Intersection with S = S0: S0 = S_max + slope * (t_lag - t_max)
    t_lag = t_max + (S0 - S_max) / slope

    # Lag can't be negative or before measurement start
    return max(0.0, max(t[0], t_lag))


def compute_kinetics(
    times: np.ndarray,
    means: np.ndarray,
    substrate: str,
    depletion_threshold: Optional[float] = None,
) -> SubstrateKinetics:
    """Compute all kinetic parameters for a substrate.

    Args:
        times: Sorted unique timepoints (h).
        means: Mean concentration at each timepoint (g/L).
        substrate: Substrate name.
        depletion_threshold: Concentration below which substrate is considered
            depleted. Default: max(0.5, 0.05 * S0).

    Returns:
        SubstrateKinetics dataclass with all parameters.
    """
    S0 = float(means[0])
    Se = float(means[-1])
    delta_S = S0 - Se
    efficiency = (delta_S / S0 * 100) if S0 > 0 else 0.0

    dt_total = times[-1] - times[0]
    q_avg = delta_S / dt_total if dt_total > 0 else 0.0

    # Consumption rates
    rates = _finite_difference_rates(times, means)

    # Maximum rate
    q_max = float(np.max(rates)) if len(rates) > 0 else 0.0
    i_qmax = int(np.argmax(rates))
    t_qmax = float(times[i_qmax])

    # Lag phase
    t_lag = _compute_lag(times, means, rates, S0)

    # Depletion threshold
    if depletion_threshold is None:
        depletion_threshold = max(0.5, 0.05 * S0)

    t_depletion = _interpolate_time(times, means, depletion_threshold)

    # t_50 and t_90
    S_50 = S0 - 0.5 * delta_S
    S_90 = S0 - 0.9 * delta_S
    t_50 = _interpolate_time(times, means, S_50) if delta_S > 0 else None
    t_90 = _interpolate_time(times, means, S_90) if delta_S > 0 else None

    # Active consumption period
    t_active = None
    q_active = None
    if t_depletion is not None and t_lag is not None:
        t_active = t_depletion - t_lag
        if t_active > 0:
            q_active = delta_S / t_active

    return SubstrateKinetics(
        substrate=substrate,
        S0=S0, Se=Se, delta_S=delta_S, efficiency=efficiency,
        q_avg=q_avg, q_max=q_max, t_qmax=t_qmax,
        t_lag=t_lag, t_depletion=t_depletion,
        t_50=t_50, t_90=t_90,
        t_active=t_active, q_active=q_active,
        times=times, means=means, rates=rates,
    )
