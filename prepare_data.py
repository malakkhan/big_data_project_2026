"""
Data Preparation Orchestrator.

Executes the heavy-duty foundational data engineering phases:
1. PySpark Data Ingestion & JSON Flattening
2. Bipartite Graph Extraction & TMDB API Hydration

Run this script ONCE per fresh dataset independently of downstream experiments.
"""

import logging
import sys
import time
import argparse

from src import config
from src.ingestion import PySparkIngestor
from src.graph_features import GraphFeatureExtractor

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger("DataPreparation")

def main():
    parser = argparse.ArgumentParser(description="Big Data Preparation Orchestrator")
    parser.add_argument("--phase1", action="store_true", help="Run Phase 1: Distributed PySpark Ingestion")
    parser.add_argument("--phase2", action="store_true", help="Run Phase 2: Graph Extraction & TMDB API Hydration")
    args = parser.parse_args()
    
    # If no explicit phase limits are dictated by the user, default to executing the entire pipeline globally.
    run_phase1 = args.phase1
    run_phase2 = args.phase2
    
    if not run_phase1 and not run_phase2:
        run_phase1 = True
        run_phase2 = True
        
    startTime = time.time()
    
    config.create_directories()

    if run_phase1:
        logger.info("Initializing Big Data Preparation Phase 1...")
        logger.info("--- Phase 1: Distributed PySpark Ingestion ---")
        ingestor = PySparkIngestor(pipeline="imdb")  # explicit — IMDb-specific repairs enabled
        ingestor.run()

    if run_phase2:
        logger.info("Initializing Big Data Preparation Phase 2...")
        logger.info("--- Phase 2: Graph Extraction & TMDB API Hydration ---")
        graphExtractor = GraphFeatureExtractor()
        graphExtractor.run()

    elapsed = time.time() - startTime
    logger.info(f"Data Preparation Concluded in {elapsed:.2f} seconds.")
    logger.info("Foundation sequentially built. You may now run `python run_experiments.py`.")

if __name__ == "__main__":
    main()
