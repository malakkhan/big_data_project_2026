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
    start_time = time.time()
    logger.info("Initializing Big Data Preparation (Phase 1 & 2)...")
    
    config.create_directories()

    logger.info("--- Phase 1: Distributed PySpark Ingestion ---")
    ingestor = PySparkIngestor()
    ingestor.run()
    
    logger.info("--- Phase 2: Graph Extraction & TMDB API Hydration ---")
    graph_extractor = GraphFeatureExtractor()
    graph_extractor.run()
    
    elapsed = time.time() - start_time
    logger.info(f"Data Preparation Concluded in {elapsed:.2f} seconds.")
    logger.info("Foundation built! You may now aggressively loop `python run_experiments.py`.")

if __name__ == "__main__":
    main()
