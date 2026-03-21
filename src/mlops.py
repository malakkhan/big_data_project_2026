"""
MLOps, Metadata Tracking, and Data Drift Governance.

Tracks statistical shifts between training baselines and newly inferred serving data.
Implements non-parametric anomaly checks resolving to Kolmogorov-Smirnov Tests, Population 
Stability Index (PSI) mapping, and Kullback-Leibler bounds.

Classes:
    DriftMonitor: Analyzes vector deviations to flag pipeline staleness.
"""

import sys
import logging
import json
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, entropy
from pathlib import Path

from src import config

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

class DriftMonitor:
    """
    Executes drift analysis simulating advanced continuous integration ML pipelines.

    Attributes:
        continuous_features (list): Strict structural schemas checked for shifts.
    """

    def __init__(self):
        """
        Identifies structural target signatures targeted for statistical anomaly testing.
        """
        self.continuous_features = ["runtimeMinutes", "numVotes"]
        
    def calculate_psi(self, expected, actual, buckets=10):
        """
        Calculates Population Stability Index (PSI) between two dynamic numerical sets.

        Args:
            expected (pd.Series): The baseline feature space from historical training.
            actual (pd.Series): The newly intercepted temporal slice.
            buckets (int, optional): The uniform quantiles mapped to density splits. Defaults to 10.

        Returns:
            float: Population Stability Index signifying drift magnitude.
        """
        def build_buckets(x):
            breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
            quants = np.percentile(x, breakpoints)
            quants += np.random.uniform(-1e-6, 1e-6, size=quants.shape)
            quants[0], quants[-1] = -np.inf, np.inf
            return quants

        if len(expected) == 0 or len(actual) == 0:
            return 0.0

        expected_binned, bins = pd.qcut(expected, q=buckets, retbins=True, duplicates='drop')
        expected_percents = expected_binned.value_counts(normalize=True).sort_index()

        actual_binned = pd.cut(actual, bins=bins, include_lowest=True)
        actual_percents = actual_binned.value_counts(normalize=True).sort_index()

        expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
        actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)

        psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
        return psi

    def calculate_kl_divergence(self, expected, actual, bins=10):
        """
        Calculates Kullback-Leibler (KL) Divergence explicitly on continuous intervals.

        Used specifically to assert that abstract semantic bounds outputted from prior SVD 
        dimensionalities continue matching the baseline probability geometry.

        Args:
            expected (pd.Series): Abstract baseline SVD feature mapping.
            actual (pd.Series): New semantic SVD intercept space.
            bins (int, optional): Histographic density divisions. Defaults to 10.

        Returns:
            float: KL asymmetry metric predicting vector rot.
        """
        hist_expected, bin_edges = np.histogram(expected, bins=bins, density=True)
        hist_actual, _ = np.histogram(actual, bins=bin_edges, density=True)

        hist_expected = np.maximum(hist_expected, 1e-8)
        hist_actual = np.maximum(hist_actual, 1e-8)

        kl_div = entropy(pk=hist_actual, qk=hist_expected)
        return kl_div

    def detect_drift(self, train_df, serving_df):
        """
        Triggers the continuous integration checks calculating schema integrity warnings.

        Args:
            train_df (pd.DataFrame): The base frame from the original training iteration.
            serving_df (pd.DataFrame): Incoming tabular frame mapped dynamically.

        Returns:
            bool: Signals whether the calculated decay indexes exceed safe thresholds.
        """
        drift_report = {}
        retrain_required = False
        
        logger.info("Executing Kolmogorov-Smirnov (K-S) Tests...")
        for col in self.continuous_features:
            if col in train_df.columns and col in serving_df.columns:
                t_data = train_df[col].dropna()
                s_data = serving_df[col].dropna()
                
                if len(t_data) > 0 and len(s_data) > 0:
                    stat, p_value = ks_2samp(t_data, s_data)
                    drift_report[f"ks_{col}"] = {"stat": float(stat), "p_value": float(p_value)}
                    
                    if p_value < 0.05:
                        logger.warning(f"K-S Test: {col} indicates statistically significant drift (p={p_value:.4f}).")
                        retrain_required = True

        logger.info("Executing KL Divergence on Textual Embeddings...")
        svd_cols = [c for c in train_df.columns if "svd" in c]
        for col in svd_cols:
            if col in serving_df.columns:
                t_data = train_df[col].dropna()
                s_data = serving_df[col].dropna()
                
                if len(t_data) > 0 and len(s_data) > 0:
                    kl_val = self.calculate_kl_divergence(t_data, s_data)
                    drift_report[f"kl_{col}"] = float(kl_val)
                    
                    if kl_val > 0.5:
                        logger.warning(f"KL Divergence: {col} indicates semantic drift (KL={kl_val:.4f}).")
                        retrain_required = True
                        
        logger.info("Executing Population Stability Index (PSI) calculations...")
        if "startYear" in train_df.columns and "startYear" in serving_df.columns:
             t_data = train_df["startYear"].dropna()
             s_data = serving_df["startYear"].dropna()
             psi_val = self.calculate_psi(t_data, s_data, buckets=5)
             drift_report["psi_startYear"] = float(psi_val)
             
             if psi_val > 0.20:
                 logger.warning(f"PSI: startYear indicates significant population shift (PSI={psi_val:.4f}).")
                 retrain_required = True

        passport_path = config.OUTPUT_DIR / "data_passport.json"
        with open(passport_path, "w") as f:
            json.dump({
                "drift_report": drift_report,
                "retrain_recommended": retrain_required,
                "serving_dataset_size": len(serving_df)
            }, f, indent=4)
            
        logger.info(f"Drift Analysis Complete. Data Passport saved to {passport_path}")
        return retrain_required

    def run(self):
        """
        Simulates checking a new validation artifact via the metadata checks mapped earlier.

        Args:
            None

        Returns:
            None
        """
        try:
            train_features = pd.read_parquet(config.PARQUET_DIR / "imputed_features.parquet")
        except:
            logger.error("Imputed features not found. Run previous stages first.")
            return
            
        logger.info("Monitoring logic initialized. Normally requires serving data inputs.")

if __name__ == "__main__":
    monitor = DriftMonitor()
    monitor.run()
