import logging
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from src import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CovarianceAnalyzer")

def analyze_collinearity(correlation_threshold=0.8):
    logger.info("Initializing Covariance and Collinearity Analysis Engine...")
    
    input_parquet = config.PARQUET_DIR / "imputed_features.parquet"
    if not input_parquet.exists():
        logger.error(f"FATAL: Missing feature matrix at {input_parquet}. Have you executed prepare_data.py?")
        return
        
    logger.info(f"Loading feature matrix from {input_parquet}")
    df = pd.read_parquet(input_parquet)
    
    # Mirror XGBoost's exact pre-processing exclusions
    columns_to_drop = [
        "isAdult", "primaryTitle", "originalTitle", "tconst",
        "startYear", "endYear", "tmdb_fetched_at", "tmdb_primary_genre"
    ]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    # Isolate targets if present
    if "averageRating" in df.columns:
        df = df.drop(columns=["averageRating"])
        
    logger.info(f"Analyzed feature space comprises {df.shape[1]} dimensions.")
    
    # Numeric enforcement (Categoricals -> numerical proxy representations)
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            df[col] = df[col].astype('category').cat.codes
            
    # Compute Spearman (Rank) Correlation structurally robust against non-normal distributions
    logger.info("Computing Spearman Rank Correlation Matrix...")
    corr_matrix = df.corr(method="spearman")
    
    # Extract highest absolute correlations
    corr_unstacked = corr_matrix.abs().unstack()
    
    # Filter out self-correlations (1.0) and sort
    mask = (corr_unstacked < 1.0)
    high_corr = corr_unstacked[mask].sort_values(ascending=False)
    
    # Drop duplicates (A-B and B-A)
    seen = set()
    unique_high_corr = []
    
    for (f1, f2), val in high_corr.items():
        pair = frozenset([f1, f2])
        if pair not in seen:
            seen.add(pair)
            if val >= correlation_threshold:
                unique_high_corr.append((f1, f2, val))
                
    logger.info("="*60)
    if not unique_high_corr:
        logger.info(f"✅ Great news! No feature pairs exhibit collinearity >= {correlation_threshold}")
    else:
        logger.warning(f"⚠️ HIGH COLLINEARITY DETECTED (Threshold >= {correlation_threshold}):")
        for f1, f2, val in unique_high_corr:
            logger.warning(f"     -> {val:.4f} | {f1} <--> {f2}")
            
    logger.info("="*60)
    
    # Generate Heatmap figure
    analysis_dir = config.OUTPUT_DIR / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    out_img = analysis_dir / "feature_correlation_heatmap.png"
    
    logger.info(f"Rendering full correlation heatmap to {out_img}...")
    
    plt.figure(figsize=(16, 12))
    # We plot the raw matrix allowing positive/negative differentiation
    sns_plot = sns.heatmap(
        corr_matrix, 
        cmap='coolwarm', 
        vmin=-1, vmax=1, 
        center=0,
        square=True, 
        linewidths=.5, 
        cbar_kws={"shrink": .75}
    )
    plt.title("XGBoost Feature Spearman Correlation Matrix")
    plt.tight_layout()
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Analysis Concluded Successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate XGBoost pipeline feature collinearity")
    parser.add_argument("--threshold", type=float, default=0.75, help="Absolute correlation threshold to flag as highly collinear")
    args = parser.parse_args()
    
    analyze_collinearity(correlation_threshold=args.threshold)
