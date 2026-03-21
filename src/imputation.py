"""
Deep Learning Multivariate Imputation Module.

Models missing states through surrogate data approximations derived from DataWig's logic.
Applies stochastic Multi-Layer Perceptrons connected to sub-network layers evaluating
character n-grams and standardized scale numerics.

Classes:
    DeepImputer: High-performance missing entity resolving component.
"""

import sys
import logging
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor, MLPClassifier

from src import config

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

class DeepImputer:
    """
    Executes Neural Network logic to probabilistically guess missing states based on
    adjacent correlative attributes within highly heterogenous matrices.

    Attributes:
        numeric_features (list): Predefined interval schemas.
        text_features (list): Abstract string schemas mapping to character hashes.
        categorical_features (list): Enumerable schema states evaluating to OH matrices.
    """

    def __init__(self):
        """
        Maps out deterministic pipeline boundaries required for the Scikit wrappers.
        """
        self.numeric_features = ["startYear", "endYear", "tmdb_popularity", "tmdb_vote_average", "tmdb_budget", "tmdb_revenue", "tmdb_runtime"]
        self.text_features = ["tmdb_production_company"]
        self.categorical_features = ["tmdb_primary_genre"]

    def build_feature_extractor(self, input_columns):
        """
        Dynamically builds a Scikit-Learn feature union evaluating heterogenous nodes.

        Args:
            input_columns (list): String identifiers tracking the column vectors visible.

        Returns:
            sklearn.compose.ColumnTransformer: A compiled multi-path execution graph mapping
                                               inputs directly into normalized perceptron scales.
        """
        transformers = []
        for col in input_columns:
            if col in self.text_features:
                transformers.append((
                    f"hash_{col}", 
                    HashingVectorizer(analyzer='char_wb', ngram_range=(2, 4), n_features=256), 
                    col
                ))
            elif col in self.categorical_features:
                transformers.append((
                    f"ohe_{col}",
                    Pipeline([
                        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("ohe", OneHotEncoder(handle_unknown='ignore'))
                    ]),
                    [col]
                ))
            elif col in self.numeric_features or pd.api.types.is_numeric_dtype(col):
                transformers.append((
                    f"num_{col}",
                    Pipeline([
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())
                    ]),
                    [col]
                ))
                
        return ColumnTransformer(transformers, remainder="drop")

    def impute_column(self, df, target_col):
        """
        Trains a predictive model spanning non-missing datasets to impute null gaps.

        Implements adaptive fallback protocols if cardinal complexities blow up neural
        topologies. Maps numeric logic to regressions and labels to complex classifications.

        Args:
            df (pd.DataFrame): Context matrix holding the signals.
            target_col (str): Entity attribute column targeted for restoration.

        Returns:
            pd.DataFrame: Imputed copy of the matrix devoid of specified null anomalies.
        """
        missing_mask = df[target_col].isna()
        if not missing_mask.any() or missing_mask.all():
            logger.warning(f"Target {target_col} is completely null or completely dense. Skipping Deep Imputation.")
            return df
            
        # Protect HashingVectorizer from crashing on native Python NoneTypes
        for col in self.text_features:
            if col in df.columns and col != target_col:
                df[col] = df[col].fillna("")
                
        input_cols = [c for c in df.columns if c != target_col and c not in ["tconst", "label", "synthetic_index"]]
        
        train_df = df[~missing_mask]
        predict_df = df[missing_mask]
        
        # Purge entirely hollow columns from the context map to prevent Scikit-Learn StandardScaler dimensional collapse
        input_cols = [c for c in input_cols if train_df[c].notna().any()]
        
        if len(train_df) < 50:
            fallback = df[target_col].mode()[0] if df[target_col].dtype == 'O' else df[target_col].median()
            df[target_col] = df[target_col].fillna(fallback)
            return df

        is_numeric = pd.api.types.is_numeric_dtype(train_df[target_col])
        is_high_cardinality = not is_numeric and train_df[target_col].nunique() > 1000
        
        if is_high_cardinality:
             df[target_col] = df[target_col].fillna("Unknown")
             return df
             
        feature_extractor = self.build_feature_extractor(input_cols)
        
        if is_numeric:
            model = MLPRegressor(
                hidden_layer_sizes=(64, 32),
                learning_rate_init=config.IMPUTER_LEARNING_RATE,
                max_iter=config.IMPUTER_EPOCHS * 10,
                batch_size=config.IMPUTER_BATCH_SIZE,
                random_state=42,
                early_stopping=True
            )
        else:
            model = MLPClassifier(
                hidden_layer_sizes=(64, 32),
                learning_rate_init=config.IMPUTER_LEARNING_RATE,
                max_iter=config.IMPUTER_EPOCHS * 10,
                batch_size=config.IMPUTER_BATCH_SIZE,
                random_state=42,
                early_stopping=True
            )
            
        pipeline = Pipeline([
            ("features", feature_extractor),
            ("mlp", model)
        ])
        
        pipeline.fit(train_df, train_df[target_col])
        predicted_values = pipeline.predict(predict_df)
        
        # Prevent aggressive Pandas PyArrow strict casting failures (e.g., float regressions into Int32 blocks)
        predicted_series = pd.Series(predicted_values, index=predict_df.index)
        if is_numeric:
            df[target_col] = df[target_col].astype(float)
        else:
            df[target_col] = df[target_col].astype(object)
            
        df.loc[missing_mask, target_col] = predicted_series
        
        return df

    def run(self):
        """
        Drives iterative sequential imputation protocols across susceptible fields.

        Args:
            None

        Returns:
            None
        """
        input_parquet = config.OUTPUT_DIR / "parquet" / "duckdb_features.parquet"
        
        try:
            df = pd.read_parquet(input_parquet)
        except Exception as e:
            return
            
        targets = ["runtimeMinutes", "numVotes", "startYear"]
        
        for tgt in targets:
            if tgt in df.columns:
                df = self.impute_column(df, tgt)
                
        output_path = config.OUTPUT_DIR / "parquet" / "imputed_features.parquet"
        df.to_parquet(output_path, index=False)

if __name__ == "__main__":
    imputer = DeepImputer()
    imputer.run()
