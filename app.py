"""HPLC Substrate Analysis — Streamlit Dashboard."""

import io
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from hplc_analysis.pipeline import (
    analyze_experiment,
    results_to_summary_df,
    results_to_timeseries_df,
    ExperimentResults,
)
from hplc_analysis.kinetics import SubstrateKinetics

# ──────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HPLC Substrate Analysis",
    page_icon="🧪",
    layout="wide",
)

COLORS = {'Glucose': '#2196F3', 'Xylose': '#FF9800'}
DATA_DIR = Path(__file__).parent / "data"

def _color(name: str) -> str:
    return COLORS.get(name, '#4CAF50')


# ──────────────────────────────────────────────────────────────
# Sidebar — data selection
# ──────────────────────────────────────────────────────────────
st.sidebar.title("Data Input")

source = st.sidebar.radio(
    "Select data source",
    ["Sample: 1:1 Glucose:Xylose", "Sample: 2:1 Glucose:Xylose", "Upload CSV"],
)

sample_files = {
    "Sample: 1:1 Glucose:Xylose": DATA_DIR / "1-to-1-GlucoseXyloseMicroplateGrowth (1).csv",
    "Sample: 2:1 Glucose:Xylose": DATA_DIR / "2-to-1-GlucoseXyloseMicroplateGrowth (1).csv",
}

filepath = None

if source.startswith("Sample"):
    filepath = str(sample_files[source])
else:
    uploaded = st.sidebar.file_uploader(
        "Upload HPLC CSV", type=["csv"],
        help="CSV with columns: Time (h), [Substrate] (g/L)",
    )
    if uploaded is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        tmp.write(uploaded.getvalue())
        tmp.flush()
        filepath = tmp.name

if filepath is None:
    st.title("HPLC Substrate Consumption Analysis")
    st.info("← Upload a CSV file or select a sample dataset from the sidebar.")
    st.stop()


# ──────────────────────────────────────────────────────────────
# Run analysis
# ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Analyzing…")
def run_analysis(path: str):
    return analyze_experiment(path)

results: ExperimentResults = run_analysis(filepath)
substrates = list(results.kinetics.keys())

st.title("HPLC Substrate Consumption Analysis")
label = Path(filepath).stem
st.caption(f"File: `{label}`  |  Substrates: {', '.join(substrates)}  |  "
           f"Timepoints: {results.metadata.n_timepoints}  |  "
           f"Time range: {results.metadata.time_range[0]:.1f} – {results.metadata.time_range[1]:.1f} h")


# ──────────────────────────────────────────────────────────────
# Tab layout
# ──────────────────────────────────────────────────────────────
tab_raw, tab_params, tab_viz, tab_formulas, tab_export = st.tabs([
    "📊 Raw Data", "📋 Parameters", "📈 Visualizations", "📐 Formulas", "💾 Export",
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — Raw Data
# ══════════════════════════════════════════════════════════════
with tab_raw:
    st.subheader("Input Data Preview")

    col_table, col_plot = st.columns([1, 2])

    with col_table:
        st.dataframe(results.raw_df, use_container_width=True, height=400)
        rep_counts = results.raw_df.groupby('Time (h)').size().reset_index(name='Replicates')
        st.markdown("**Replicates per timepoint**")
        st.dataframe(rep_counts, use_container_width=True, height=250)

    with col_plot:
        fig, ax = plt.subplots(figsize=(10, 5))
        for sub in substrates:
            col = f'{sub} (g/L)'
            ax.scatter(results.raw_df['Time (h)'], results.raw_df[col],
                       c=_color(sub), alpha=0.6, s=35, label=sub,
                       edgecolors='white', linewidth=0.4)
        ax.set_xlabel('Time (h)')
        ax.set_ylabel('Concentration (g/L)')
        ax.set_title('Raw Replicate Measurements')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ══════════════════════════════════════════════════════════════
# TAB 2 — Parameters
# ══════════════════════════════════════════════════════════════
with tab_params:
    st.subheader("Kinetic Parameters")

    # Summary table
    rows = []
    for sub, kin in results.kinetics.items():
        rows.append({
            'Substrate': sub,
            'S₀ (g/L)': round(kin.S0, 2),
            'Sₑ (g/L)': round(kin.Se, 3),
            'ΔS (g/L)': round(kin.delta_S, 2),
            'Efficiency (%)': round(kin.efficiency, 1),
            'q_avg (g/L/h)': round(kin.q_avg, 3),
            'q_max (g/L/h)': round(kin.q_max, 3),
            't_qmax (h)': round(kin.t_qmax, 1),
            't_lag (h)': round(kin.t_lag, 1),
            't₅₀ (h)': round(kin.t_50, 1) if kin.t_50 else '—',
            't₉₀ (h)': round(kin.t_90, 1) if kin.t_90 else '—',
            't_depletion (h)': round(kin.t_depletion, 1) if kin.t_depletion else '—',
            't_active (h)': round(kin.t_active, 1) if kin.t_active else '—',
            'q_active (g/L/h)': round(kin.q_active, 3) if kin.q_active else '—',
        })
    param_df = pd.DataFrame(rows).set_index('Substrate')
    st.dataframe(param_df.T.rename_axis('Parameter'), use_container_width=True)

    # Per-substrate metric cards
    st.markdown("---")
    cols = st.columns(len(substrates))
    for col, sub in zip(cols, substrates):
        kin = results.kinetics[sub]
        with col:
            st.markdown(f"### {sub}")
            m1, m2, m3 = st.columns(3)
            m1.metric("S₀", f"{kin.S0:.2f} g/L")
            m2.metric("Efficiency", f"{kin.efficiency:.1f}%")
            m3.metric("q_max", f"{kin.q_max:.3f} g/L/h")
            m4, m5, m6 = st.columns(3)
            m4.metric("t_lag", f"{kin.t_lag:.1f} h")
            m5.metric("t_depletion", f"{kin.t_depletion:.1f} h" if kin.t_depletion else "—")
            m6.metric("t_active", f"{kin.t_active:.1f} h" if kin.t_active else "—")

    # Diauxic analysis
    st.markdown("---")
    st.subheader("Diauxic Shift Analysis")
    d = results.diauxic
    dc1, dc2, dc3, dc4 = st.columns(4)
    dc1.metric("Detected", "Yes" if d.diauxic_detected else "No")
    dc2.metric("Primary", d.primary_substrate or "—")
    dc3.metric("Order", " → ".join(d.consumption_order))
    dc4.metric("Overlap", f"{d.overlap_fraction:.1%}" if d.overlap_fraction is not None else "—")

    # Quality
    st.markdown("---")
    st.subheader("Data Quality")
    qcols = st.columns(len(substrates))
    for qcol, sub in zip(qcols, substrates):
        qr = results.quality[sub]
        with qcol:
            st.markdown(f"**{sub}**")
            st.write(f"- Max CV: **{qr.max_cv:.1f}%** at t={qr.max_cv_time:.1f}h")
            st.write(f"- Monotonicity: **{qr.monotonicity_score:.2f}**")
            if qr.high_cv_times:
                st.warning(f"High CV (>20%) at: {[f'{t:.1f}h' for t in qr.high_cv_times]}")
            if qr.increasing_intervals:
                st.warning(f"Non-physical increases at: {[(f'{a:.1f}→{b:.1f}h', f'+{d:.3f}') for a, b, d in qr.increasing_intervals]}")

            ol = results.outliers[sub]
            if len(ol) > 0:
                st.write(f"Outliers flagged: {len(ol)}")
                st.dataframe(ol, use_container_width=True)
            else:
                st.write("No outliers flagged.")


# ══════════════════════════════════════════════════════════════
# TAB 3 — Visualizations
# ══════════════════════════════════════════════════════════════
with tab_viz:
    st.subheader("Visualizations")

    # ── Plot 1: Consumption curves ──
    st.markdown("#### 1. Substrate Consumption Curves")
    st.caption("Mean ± 95% CI with replicate scatter and depletion lines")
    fig1, ax1 = plt.subplots(figsize=(11, 5.5))
    for sub in substrates:
        c = _color(sub)
        st_df = results.stats[sub]
        kin = results.kinetics[sub]
        ax1.fill_between(st_df.index, st_df['ci95_lower'], st_df['ci95_upper'], alpha=0.2, color=c)
        ax1.plot(st_df.index, st_df['mean'], 'o-', color=c, linewidth=2, markersize=5, label=sub)
        ax1.scatter(results.raw_df['Time (h)'], results.raw_df[f'{sub} (g/L)'],
                    color=c, alpha=0.25, s=12, zorder=1)
        if kin.t_depletion:
            ax1.axvline(kin.t_depletion, color=c, linestyle='--', alpha=0.6,
                        label=f'{sub} depletion ({kin.t_depletion:.1f}h)')
    ax1.set_xlabel('Time (h)'); ax1.set_ylabel('Concentration (g/L)')
    ax1.set_title('Substrate Consumption Curves'); ax1.legend(); ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    st.pyplot(fig1); plt.close(fig1)

    st.markdown("---")

    # ── Plot 2 & 3 side by side ──
    col_r, col_n = st.columns(2)

    with col_r:
        st.markdown("#### 2. Consumption Rate")
        st.caption("rate = −dS/dt, star = q_max")
        fig2, ax2 = plt.subplots(figsize=(7, 4.5))
        for sub in substrates:
            kin = results.kinetics[sub]
            c = _color(sub)
            ax2.plot(kin.times, kin.rates, 'o-', color=c, linewidth=2, markersize=4, label=sub)
            ax2.plot(kin.t_qmax, kin.q_max, '*', color='red', markersize=14, zorder=5)
        ax2.axhline(0, color='gray', linewidth=0.5)
        ax2.set_xlabel('Time (h)'); ax2.set_ylabel('Rate (g/L/h)')
        ax2.legend(); ax2.grid(True, alpha=0.3); fig2.tight_layout()
        st.pyplot(fig2); plt.close(fig2)

    with col_n:
        st.markdown("#### 3. Normalized Consumption")
        st.caption("(S₀ − S(t)) / ΔS — 50% / 90% lines")
        fig3, ax3 = plt.subplots(figsize=(7, 4.5))
        for sub in substrates:
            kin = results.kinetics[sub]
            c = _color(sub)
            norm = np.clip((kin.S0 - kin.means) / kin.delta_S, 0, 1.05) if kin.delta_S > 0 else np.zeros_like(kin.means)
            ax3.plot(kin.times, norm, 'o-', color=c, linewidth=2, markersize=4, label=sub)
        ax3.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='50%')
        ax3.axhline(0.9, color='gray', linestyle='--', alpha=0.5, label='90%')
        ax3.set_xlabel('Time (h)'); ax3.set_ylabel('Fraction consumed')
        ax3.set_ylim(-0.05, 1.1); ax3.legend(); ax3.grid(True, alpha=0.3); fig3.tight_layout()
        st.pyplot(fig3); plt.close(fig3)

    st.markdown("---")

    # ── Plot 4 & 5 side by side ──
    col_d, col_cv = st.columns(2)

    with col_d:
        st.markdown("#### 4. Diauxic Analysis")
        fig4, ax4 = plt.subplots(figsize=(7, 4.5))
        for sub, kin in results.kinetics.items():
            c = _color(sub)
            ax4.plot(kin.times, kin.means, 'o-', color=c, linewidth=2, markersize=4, label=sub)
            if sub in results.diauxic.active_windows:
                t_s, t_e = results.diauxic.active_windows[sub]
                ax4.axvspan(t_s, t_e, alpha=0.08, color=c)
            if kin.t_depletion:
                ax4.axvline(kin.t_depletion, color=c, linestyle='--', alpha=0.4)
        status = "DETECTED" if results.diauxic.diauxic_detected else "Not detected"
        ax4.text(0.02, 0.98, f'Diauxic: {status}', transform=ax4.transAxes,
                 fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        ax4.set_xlabel('Time (h)'); ax4.set_ylabel('Conc. (g/L)')
        ax4.legend(); ax4.grid(True, alpha=0.3); fig4.tight_layout()
        st.pyplot(fig4); plt.close(fig4)

    with col_cv:
        st.markdown("#### 5. Replicate CV%")
        fig5, ax5 = plt.subplots(figsize=(7, 4.5))
        for sub, st_df in results.stats.items():
            ax5.plot(st_df.index, st_df['cv_pct'], 'o-', color=_color(sub), linewidth=2, markersize=4, label=sub)
        ax5.axhline(20, color='red', linestyle='--', alpha=0.5, label='20% threshold')
        ax5.set_xlabel('Time (h)'); ax5.set_ylabel('CV (%)')
        ax5.legend(); ax5.grid(True, alpha=0.3); fig5.tight_layout()
        st.pyplot(fig5); plt.close(fig5)

    st.markdown("---")

    # ── Plot 6: All-params annotated (one substrate selector) ──
    st.markdown("#### 6. All Parameters — Annotated View")
    sel_sub = st.selectbox("Select substrate", substrates, key="annotated_sub")
    kin = results.kinetics[sel_sub]
    st_df = results.stats[sel_sub]
    c = _color(sel_sub)

    fig6, ax6 = plt.subplots(figsize=(13, 6.5))
    ax6.fill_between(st_df.index, st_df['ci95_lower'], st_df['ci95_upper'], alpha=0.15, color=c)
    ax6.plot(kin.times, kin.means, 'o-', color=c, linewidth=2.5, markersize=6, zorder=3)

    # S0
    ax6.axhline(kin.S0, color='green', linestyle=':', alpha=0.6, linewidth=1.5)
    ax6.text(kin.times[-1] + 1, kin.S0, f'S₀ = {kin.S0:.1f}', fontsize=10, color='green', va='center')

    # Tangent at q_max
    slope_val = -kin.q_max
    i_qmax = int(np.argmin(np.abs(kin.times - kin.t_qmax)))
    S_at_qmax = kin.means[i_qmax]
    t_tang = np.linspace(max(kin.t_lag - 3, kin.times[0]), kin.t_qmax + 6, 80)
    S_tang = S_at_qmax + slope_val * (t_tang - kin.t_qmax)
    ax6.plot(t_tang, S_tang, '--', color='red', linewidth=2, alpha=0.7)

    # t_lag
    ax6.axvline(kin.t_lag, color='purple', linestyle='-.', alpha=0.5, linewidth=1.5)
    ax6.plot(kin.t_lag, kin.S0, 'D', color='purple', markersize=10, zorder=5)
    ax6.annotate(f't_lag = {kin.t_lag:.1f}h', xy=(kin.t_lag, kin.S0),
                 xytext=(kin.t_lag - 12, kin.S0 * 0.65),
                 fontsize=10, fontweight='bold', color='purple',
                 arrowprops=dict(arrowstyle='->', color='purple', lw=1.5))

    # q_max star
    ax6.plot(kin.t_qmax, S_at_qmax, '*', color='red', markersize=18, zorder=5)
    ax6.annotate(f'q_max = {kin.q_max:.3f}\n@ {kin.t_qmax:.0f}h',
                 xy=(kin.t_qmax, S_at_qmax), xytext=(kin.t_qmax + 7, S_at_qmax + kin.S0 * 0.12),
                 fontsize=10, fontweight='bold', color='red',
                 arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    # t_50
    if kin.t_50:
        S50 = kin.S0 - 0.5 * kin.delta_S
        ax6.axhline(S50, color='#9C27B0', linestyle=':', alpha=0.3)
        ax6.plot(kin.t_50, S50, 's', color='#9C27B0', markersize=9, zorder=5)
        ax6.annotate(f't₅₀ = {kin.t_50:.1f}h', xy=(kin.t_50, S50),
                     xytext=(kin.t_50 + 5, S50 + kin.S0 * 0.08),
                     fontsize=9, color='#9C27B0', fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color='#9C27B0'))

    # t_90
    if kin.t_90:
        S90 = kin.S0 - 0.9 * kin.delta_S
        ax6.plot(kin.t_90, S90, 's', color='#E91E63', markersize=9, zorder=5)
        ax6.annotate(f't₉₀ = {kin.t_90:.1f}h', xy=(kin.t_90, S90),
                     xytext=(kin.t_90 + 5, S90 + kin.S0 * 0.08),
                     fontsize=9, color='#E91E63', fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color='#E91E63'))

    # t_depletion
    if kin.t_depletion:
        dep_th = max(0.5, 0.05 * kin.S0)
        ax6.axvline(kin.t_depletion, color='red', linestyle='--', alpha=0.4)
        ax6.plot(kin.t_depletion, dep_th, 'v', color='red', markersize=12, zorder=5)
        ax6.annotate(f't_dep = {kin.t_depletion:.1f}h', xy=(kin.t_depletion, dep_th),
                     xytext=(kin.t_depletion + 4, dep_th + kin.S0 * 0.12),
                     fontsize=9, color='red', fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color='red'))

    # Active period bracket
    if kin.t_lag is not None and kin.t_depletion is not None and kin.t_active and kin.t_active > 0:
        y_br = -kin.S0 * 0.18
        ax6.annotate('', xy=(kin.t_lag, y_br), xytext=(kin.t_depletion, y_br),
                     arrowprops=dict(arrowstyle='<->', color='darkgreen', lw=2))
        ax6.text((kin.t_lag + kin.t_depletion) / 2, y_br - kin.S0 * 0.06,
                 f't_active = {kin.t_active:.1f}h  |  q_active = {kin.q_active:.3f} g/L/h',
                 ha='center', fontsize=10, color='darkgreen', fontweight='bold')

    ax6.set_xlabel('Time (h)'); ax6.set_ylabel(f'{sel_sub} (g/L)')
    ax6.set_title(f'{sel_sub} — All Parameters at a Glance', fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(-kin.S0 * 0.3, kin.S0 * 1.15)
    fig6.tight_layout()
    st.pyplot(fig6); plt.close(fig6)


# ══════════════════════════════════════════════════════════════
# TAB 4 — Formulas
# ══════════════════════════════════════════════════════════════
with tab_formulas:
    st.subheader("Calculation Methods")

    st.markdown("### Per-Substrate Parameters")
    st.markdown("""
| Parameter | Formula | Description |
|:---|:---|:---|
| **S₀** | First timepoint mean | Initial concentration |
| **Sₑ** | Last timepoint mean | Final concentration |
| **ΔS** | S₀ − Sₑ | Total substrate consumed |
| **Efficiency** | ΔS / S₀ × 100 | % of substrate consumed |
| **q_avg** | ΔS / Δt | Average consumption rate |
| **q_max** | max(−dS/dt) | Peak instantaneous consumption rate |
| **t_lag** | Tangent-line intersection | End of lag phase |
| **t_depletion** | Interpolation at threshold | Time substrate is depleted |
| **t₅₀, t₉₀** | Interpolation at 50%, 90% ΔS | Consumption milestone times |
| **t_active** | t_depletion − t_lag | Duration of active consumption |
| **q_active** | ΔS / t_active | Average rate during active phase |
""")

    st.markdown("---")
    st.markdown("### Consumption Rate — Non-uniform Finite Difference")
    st.latex(r"""
\frac{dS}{dt}\bigg|_i =
\frac{-h_2}{h_1(h_1+h_2)} S_{i-1}
+ \frac{h_2-h_1}{h_1 h_2} S_i
+ \frac{h_1}{h_2(h_1+h_2)} S_{i+1}
""")
    st.markdown(r"where $h_1 = t_i - t_{i-1}$, $h_2 = t_{i+1} - t_i$")
    st.info("Standard `np.diff` assumes uniform spacing. This data has intervals from 1.7h to 25.7h, "
            "so the weighted 3-point stencil is required for accuracy.")

    st.markdown("---")
    st.markdown("### Lag Phase — Tangent-Line Intersection")
    st.latex(r"""
\text{Tangent: } S(t) = S_{q_\max} + \left.\frac{dS}{dt}\right|_{q_\max} \cdot (t - t_{q_\max})
""")
    st.latex(r"""
t_{\text{lag}} = t_{q_\max} + \frac{S_0 - S_{q_\max}}{dS/dt|_{q_\max}}
""")
    st.markdown("The tangent at the steepest consumption point intersects the S₀ horizontal line → **lag time**.")

    st.markdown("---")
    st.markdown("### Depletion / t₅₀ / t₉₀ — Linear Interpolation")
    st.latex(r"""
t_{\text{target}} = t_i + \frac{S_i - S_{\text{target}}}{S_i - S_{i+1}} \cdot (t_{i+1} - t_i)
""")
    st.markdown("""
| Target | Value |
|:---|:---|
| t₅₀ | S₀ − 0.5 × ΔS |
| t₉₀ | S₀ − 0.9 × ΔS |
| t_depletion | max(0.5, 0.05 × S₀) |
""")

    st.markdown("---")
    st.markdown("### Replicate Statistics")
    st.latex(r"\bar{S} = \frac{1}{n}\sum S_i \qquad \sigma = \sqrt{\frac{\sum(S_i-\bar{S})^2}{n-1}}")
    st.latex(r"\text{SEM} = \frac{\sigma}{\sqrt{n}} \qquad \text{95\% CI} = \bar{S} \pm t_{0.975,\,n-1} \times \text{SEM}")
    st.latex(r"\text{CV (\%)} = \frac{\sigma}{\bar{S}} \times 100")

    st.markdown("---")
    st.markdown("### Outlier Detection — Modified Z-score (MAD)")
    st.latex(r"\text{MAD} = \text{median}(|S_i - \tilde{S}|)")
    st.latex(r"M_i = \frac{0.6745 \cdot (S_i - \tilde{S})}{\text{MAD}}")
    st.markdown("Flagged if |Mᵢ| > 3.5. Only applied when n ≥ 3. **No auto-removal** — flagging only.")

    st.markdown("---")
    st.markdown("### Diauxic Shift Analysis")
    st.markdown("""
- **Active window**: timepoints where rate > 10% × q_max
- **Primary substrate**: first to reach depletion
- **Diauxic lag**: gap between primary depletion and secondary active start
- **Overlap fraction**: simultaneous consumption time / total active time
""")


# ══════════════════════════════════════════════════════════════
# TAB 5 — Export
# ══════════════════════════════════════════════════════════════
with tab_export:
    st.subheader("Export Results")

    summary_df = results_to_summary_df(results, label)
    ts_df = results_to_timeseries_df(results, label)

    col_s, col_t = st.columns(2)

    with col_s:
        st.markdown("#### Summary CSV")
        st.caption("One row per experiment, all parameters")
        st.dataframe(summary_df.T.rename(columns={0: 'Value'}), use_container_width=True)
        csv_s = summary_df.to_csv(index=False)
        st.download_button("Download summary.csv", csv_s, file_name=f"{label}_summary.csv", mime="text/csv")

    with col_t:
        st.markdown("#### Timeseries CSV")
        st.caption("Per-timepoint detail: mean, SD, CI, rate, normalized consumption")
        st.dataframe(ts_df.head(20), use_container_width=True, height=400)
        csv_t = ts_df.to_csv(index=False)
        st.download_button("Download timeseries.csv", csv_t, file_name=f"{label}_timeseries.csv", mime="text/csv")
