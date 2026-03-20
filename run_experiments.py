"""
Global Experimentation Engine.

Iterates through macroscopic structural configurations (MAD multipliers and 
deep learning imputer dimensions) alongside micro-algorithmic XGBoost tuning.
"""

import sys
import logging
import itertools
from pathlib import Path

from src import config
from src.duckdb_processor import DuckDBFeatureEngineer
from src.imputation import DeepImputer
from src.modeling import XGBoostModeler

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger("GlobalExperimentRunner")

def run_experiments():
    """
    Executes a comprehensive grid search across macro pipeline parameters.
    
    This function temporarily modifies `config.py` variables for each 
    combination of MAD thresholds and Imputer variables. It triggers 
    DuckDB parsing, ML Imputation, and XGBoost Modeling. Individual 
    Optuna trials inside XGBoost will capture the ROC plots.

    Returns:
        None
    """
    config.create_directories()
    
    # 1. Define hyperparameter macro-grid
    mad_multipliers = [3.0, 4.0]
    imputer_epochs = [5, 10]
    imputer_batch_sizes = [64, 128]
    imputer_lrs = [1e-3, 5e-3]
    
    combinations = list(itertools.product(
        mad_multipliers, imputer_epochs, imputer_batch_sizes, imputer_lrs
    ))
    
    logger.info(f"Initialized Global Experiment Runner. Testing {len(combinations)} parameter topologies.")
    
    for idx, (mad, epochs, bs, lr) in enumerate(combinations):
        logger.info("="*60)
        logger.info(f"🚀 RUNNING EXPERIMENT BLOCK {idx+1}/{len(combinations)}")
        logger.info(f"Parameters: MAD={mad}, Epochs={epochs}, BatchSize={bs}, LR={lr}")
        logger.info("="*60)
        
        # Inject macro-parameters into runtime config dynamically
        config.MAD_THRESHOLD_MULTIPLIER = mad
        config.IMPUTER_EPOCHS = epochs
        config.IMPUTER_BATCH_SIZE = bs
        config.IMPUTER_LEARNING_RATE = lr
        
        # Prefix for trials in this specific branch
        macro_prefix = f"exp_mad{mad}_ep{epochs}_bs{bs}_lr{lr}"
        
        # 1. DuckDB Phase (Outliers)
        logger.info("-> Executing DuckDB Filtration")
        duckdb_processor = DuckDBFeatureEngineer()
        duckdb_processor.run()
        
        # 2. Imputation Phase
        logger.info("-> Executing Neural Imputation")
        imputer = DeepImputer()
        imputer.run()
        
        # 3. XGBoost & Optuna Bayesian Phase 
        # (Optuna will internally save ROC curves into `experiment_results`)
        logger.info("-> Executing XGBoost Search Space")
        modeler = XGBoostModeler(experiment_prefix=macro_prefix)
        modeler.run()
        
    logger.info("🎉 All Global Experiments Concluded Successfully!")

if __name__ == "__main__":
    run_experiments()
