"""
Data Preparation Orchestrator.

Executes the foundational data engineering phases:
  Phase 1: PySpark Ingestion + MAD Cleaning
  Phase 2: TMDB API Enrichment + Genre Encoding
  Phase 3: Graph Feature Computation (bipartite centralities + synergies)

Run this script ONCE per fresh dataset independently of downstream experiments.

Design:
    A single SparkSession is created here and shared across all phases to
    avoid redundant JVM overhead and ensure consistent configuration.
    Legacy Parquet format is enabled so downstream Pandas consumers can
    read Spark-written Parquet without PyArrow compatibility issues.
"""

import logging
import sys
import time
import argparse

from pyspark.sql import SparkSession

from src import config
from src.ingestion import PySparkIngestor
from src.tmdb_enrichment import TMDBEnrichment
from src.graph_features import GraphFeatureExtractor

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger("DataPreparation")

def _build_spark() -> SparkSession:
    """
    Build the shared SparkSession with legacy Parquet format enabled.
    
    The legacy format writes Parquet files compatible with all PyArrow
    versions, eliminating the need for .toPandas() conversions when
    downstream consumers read with pd.read_parquet().
    """
    return SparkSession.builder \
        .appName(config.SPARK_APP_NAME) \
        .master(config.SPARK_MASTER) \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.sql.ansi.enabled", "false") \
        .config("spark.sql.parquet.writeLegacyFormat", "true") \
        .getOrCreate()

def main():
    parser = argparse.ArgumentParser(description="Big Data Preparation Orchestrator")
    parser.add_argument("--phase1", action="store_true", help="Phase 1: PySpark Ingestion + MAD Cleaning")
    parser.add_argument("--phase2", action="store_true", help="Phase 2: TMDB Enrichment + Genre Encoding")
    parser.add_argument("--phase3", action="store_true", help="Phase 3: Graph Feature Computation")
    args = parser.parse_args()
    
    run_phase1 = args.phase1
    run_phase2 = args.phase2
    run_phase3 = args.phase3
    
    # If no flags -> run all phases sequentially
    if not run_phase1 and not run_phase2 and not run_phase3:
        run_phase1 = True
        run_phase2 = True
        run_phase3 = True
        
    startTime = time.time()
    config.create_directories()
    
    # Single shared SparkSession for all phases
    spark = _build_spark()

    if run_phase1:
        logger.info("=" * 60)
        logger.info("--- Phase 1: PySpark Ingestion + MAD Cleaning ---")
        logger.info("=" * 60)
        ingestor = PySparkIngestor(pipeline="imdb", spark=spark)
        ingestor.run_with_cleaning()

    if run_phase2:
        logger.info("=" * 60)
        logger.info("--- Phase 2: TMDB Enrichment + Genre Encoding ---")
        logger.info("=" * 60)
        enricher = TMDBEnrichment(spark=spark)
        enricher.run()

    if run_phase3:
        logger.info("=" * 60)
        logger.info("--- Phase 3: Graph Feature Computation ---")
        logger.info("=" * 60)
        graphExtractor = GraphFeatureExtractor(spark=spark)
        graphExtractor.run()

    elapsed = time.time() - startTime
    logger.info(f"Data Preparation Concluded in {elapsed:.2f} seconds.")
    logger.info("Foundation built. You may now run `python run_experiments.py`.")

if __name__ == "__main__":
    main()
