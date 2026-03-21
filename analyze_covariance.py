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
    
    input_parquet = config.PARQUET_DIR / "enriched_features.parquet"
    if not input_parquet.exists():
        logger.error(f"FATAL: Missing feature matrix at {input_parquet}. Have you executed prepare_data.py?")
        return
        
    logger.info(f"Loading feature matrix from {input_parquet}")
    df = pd.read_parquet(input_parquet)
    
    # Mirror XGBoost's exact pre-processing exclusions
    columns_to_drop = [
        "isAdult", "primaryTitle", "originalTitle", "tconst",
        "startYear", "endYear"
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


def analyze_genre_label_association():
    """
    Examine the relationship between tmdb_primary_genre and the binary label.

    Produces:
      1. A grouped bar chart of label distribution per genre.
      2. A chi-squared test of independence with Cramér's V effect size.
    """
    from scipy.stats import chi2_contingency

    logger.info("=" * 60)
    logger.info("Genre–Label Association Analysis")
    logger.info("=" * 60)

    input_parquet = config.PARQUET_DIR / "enriched_features.parquet"
    if not input_parquet.exists():
        logger.error(f"FATAL: Missing feature matrix at {input_parquet}.")
        return

    df = pd.read_parquet(input_parquet)

    if "tmdb_primary_genre" not in df.columns or "label" not in df.columns:
        logger.warning("Skipping genre–label analysis: required columns not found.")
        return

    df["label"] = df["label"].astype(int)

    # Build contingency table
    ct = pd.crosstab(df["tmdb_primary_genre"], df["label"], margins=False)
    ct.columns = ["label=0", "label=1"]

    logger.info("Genre × Label contingency table:")
    for genre_name in ct.index:
        row = ct.loc[genre_name]
        total = row.sum()
        pct_positive = 100 * row["label=1"] / total if total > 0 else 0
        logger.info(f"  {genre_name:30s}  n={total:5d}  label=1 rate={pct_positive:5.1f}%")

    # Chi-squared test
    chi2, p_value, dof, expected = chi2_contingency(ct)

    # Cramér's V (effect size for categorical association)
    n = ct.values.sum()
    k = min(ct.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * k)) if k > 0 else 0.0

    logger.info("-" * 60)
    logger.info(f"Chi-squared statistic : {chi2:.2f}")
    logger.info(f"Degrees of freedom    : {dof}")
    logger.info(f"p-value               : {p_value:.6f}")
    logger.info(f"Cramér's V            : {cramers_v:.4f}")

    if p_value < 0.05:
        logger.info("→ Genre and label are STATISTICALLY DEPENDENT (p < 0.05).")
    else:
        logger.info("→ No significant association detected between genre and label.")
    logger.info("=" * 60)

    # --- Visualisation: grouped bar chart ---
    analysis_dir = config.OUTPUT_DIR / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Compute positive-label rate per genre, sorted
    genre_stats = (
        df.groupby("tmdb_primary_genre")["label"]
        .agg(["count", "mean"])
        .rename(columns={"count": "n", "mean": "positive_rate"})
        .sort_values("positive_rate", ascending=True)
    )
    # Filter to genres with at least 10 samples for readable chart
    genre_stats = genre_stats[genre_stats["n"] >= 10]

    fig, ax = plt.subplots(figsize=(12, max(6, len(genre_stats) * 0.4)))
    bars = ax.barh(
        genre_stats.index,
        genre_stats["positive_rate"],
        color=plt.cm.RdYlGn(genre_stats["positive_rate"]),
        edgecolor="grey",
        linewidth=0.5,
    )

    # Annotate bars with sample count
    for bar, (genre, row) in zip(bars, genre_stats.iterrows()):
        ax.text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"n={int(row['n'])}", va="center", fontsize=8, color="grey"
        )

    ax.set_xlabel("Positive Label Rate (label=1)")
    ax.set_title(
        f"Label Distribution by TMDB Genre\n"
        f"(χ²={chi2:.1f}, p={p_value:.4f}, Cramér's V={cramers_v:.3f})"
    )
    ax.set_xlim(0, min(1.0, genre_stats["positive_rate"].max() + 0.1))
    ax.axvline(x=df["label"].mean(), color="navy", linestyle="--", linewidth=1, label=f"Overall rate ({df['label'].mean():.2f})")
    ax.legend(loc="lower right")
    plt.tight_layout()

    out_img = analysis_dir / "genre_label_association.png"
    plt.savefig(out_img, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Genre–label chart saved to {out_img}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate XGBoost pipeline feature collinearity")
    parser.add_argument("--threshold", type=float, default=0.75, help="Absolute correlation threshold to flag as highly collinear")
    args = parser.parse_args()
    
    analyze_collinearity(correlation_threshold=args.threshold)
    analyze_genre_label_association()
