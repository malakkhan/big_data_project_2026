"""
DuckDB & Scikit-Learn Feature Engineering Module.

Executes zero-copy Parquet ingestion, computes Robust Univariate Outlier Detection
(Median Absolute Deviation), and generates semantic dimensionality reduction sequences
via Term Frequency-Inverse Document Frequency (TF-IDF) mapping into SVD components.

Classes:
    DuckDBFeatureEngineer: Handles analytics and reduction via DuckDB querying.
"""

import sys
import logging
import duckdb
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from src import config

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

class DuckDBFeatureEngineer:
    """
    Executes an analytical memory pass over massive data clusters using DuckDB.

    Attributes:
        con (duckdb.DuckDBPyConnection): The mapped in-memory transactional database.
    """

    def __init__(self):
        """
        Initializes an ephemeral DuckDB transactional memory layout over the cluster.
        """
        self.con = duckdb.connect(database=':memory:')

    def apply_mad_filter(self, table_name, column, multiplier=3.0, constant=1.4826):
        """
        Filters out severe numerical outliers using the Hampel X84 filter criteria.

        Uses robust analytical DuckDB window functions (`percentile_cont`) to capture
        the Median Absolute Deviation without degrading under massive skew.

        Args:
            table_name (str): The DuckDB relation identifier to query.
            column (str): The numeric variable target.
            multiplier (float, optional): Tuning severity to eject extreme signals. Defaults to 3.0.
            constant (float, optional): Normal distribution scaling factor. Defaults to 1.4826.

        Returns:
            str: The identifier name of the newly created filtered relational view.
        """
        logger.info(f"Executing robust outlier detection (MAD) on {column}")
        
        median_query = f"SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY {column}) FROM {table_name} WHERE {column} IS NOT NULL"
        median_val = self.con.execute(median_query).fetchone()[0]
        
        if median_val is None:
            return table_name
            
        mad_query = f"""
            SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY abs({column} - {median_val}))
            FROM {table_name}
            WHERE {column} IS NOT NULL
        """
        mad_val = self.con.execute(mad_query).fetchone()[0]
        
        if mad_val is None or mad_val == 0:
             return table_name
             
        mad_val = float(mad_val)
        median_val = float(median_val)
        
        threshold = multiplier * constant * mad_val
        lower_bound = median_val - threshold
        upper_bound = median_val + threshold
        
        filtered_table = f"{table_name}_{column}_filtered"
        filter_sql = f"""
            CREATE TABLE {filtered_table} AS 
            SELECT * FROM {table_name}
            WHERE ({column} IS NULL) OR ({column} BETWEEN {lower_bound} AND {upper_bound})
        """
        self.con.execute(filter_sql)
        return filtered_table

    def apply_tfidf_svd(self, df: pd.DataFrame, text_columns, n_components=10, is_inference=False):
        """
        Extracts complex latent structures from text using algebraic geometry (SVD).

        Bypasses the curse of dimensionality native to one-hot encoding across NLP maps
        by generating fixed n-feature dense embeddings.

        Args:
            df (pd.DataFrame): In-memory analytical matrix.
            text_columns (list): String-target identities (e.g. ['primaryTitle']).
            n_components (int, optional): Semantic rank matrices extracted from TF-IDF. Defaults to 10.

        Returns:
            pd.DataFrame: Augmented matrix infused with compressed lexical signatures.
        """
        import numpy as np
        import joblib
        from src import config
        for col in text_columns:
            if col not in df.columns:
                continue
                
            text_data = df[col].fillna("")
            vectorizer_path = config.OUTPUT_DIR / "models" / f"{col}_vectorizer.joblib"
            svd_path = config.OUTPUT_DIR / "models" / f"{col}_svd.joblib"
            
            if not is_inference:
                vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(text_data)
                
                svd = TruncatedSVD(n_components=n_components, random_state=42)
                dense_embeddings = svd.fit_transform(tfidf_matrix)
                
                config.OUTPUT_DIR.joinpath("models").mkdir(parents=True, exist_ok=True)
                joblib.dump(vectorizer, vectorizer_path)
                joblib.dump(svd, svd_path)
            else:
                try:
                    vectorizer = joblib.load(vectorizer_path)
                    svd = joblib.load(svd_path)
                    tfidf_matrix = vectorizer.transform(text_data)
                    dense_embeddings = svd.transform(tfidf_matrix)
                except Exception as e:
                    logger.warning(f"Failed to load SVD dependencies. Filling {col} with 0: {e}")
                    dense_embeddings = np.zeros((len(df), n_components))
            
            for i in range(n_components):
                df[f"{col}_svd_{i}"] = dense_embeddings[:, i]
                
        return df

    def run(self, is_inference=False):
        """
        Drives DuckDB outlier isolation and ML dimensionality logic end-to-end.

        Args:
            is_inference (bool): Flag to strictly bypass outlier filtration 
                                 ensuring no inference rows are permanently deleted.

        Returns:
            None
        """
        input_parquet = str(config.PARQUET_DIR / "featured_graph.parquet")
        
        self.con.execute(f"CREATE TABLE movies AS SELECT * FROM read_parquet('{input_parquet}')")
        
        if not is_inference:
            current_table = self.apply_mad_filter("movies", "runtimeMinutes", config.MAD_THRESHOLD_MULTIPLIER)
            current_table = self.apply_mad_filter(current_table, "numVotes", config.MAD_THRESHOLD_MULTIPLIER)
        else:
            current_table = "movies"
            
        result_df = self.con.execute(f"SELECT * FROM {current_table}").fetchdf()
        
        result_df = self.apply_tfidf_svd(result_df, ["tmdb_primary_genre"], n_components=5, is_inference=is_inference)
        
        output_path = str(config.PARQUET_DIR / "duckdb_features.parquet")
        result_df.to_parquet(output_path, index=False)

if __name__ == "__main__":
    engineer = DuckDBFeatureEngineer()
    engineer.run()
