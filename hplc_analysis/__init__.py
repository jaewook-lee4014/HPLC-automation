"""HPLC Substrate Analysis Automation Pipeline."""

from .loader import load_hplc_csv, DatasetMetadata
from .stats import compute_replicate_stats, detect_outliers, QualityReport
from .kinetics import compute_kinetics, SubstrateKinetics
from .diauxic import analyze_diauxic, DiauxicAnalysis
from .pipeline import analyze_experiment, batch_analyze, export_results
