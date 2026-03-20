"""
PySpark-based Data Ingestion and Transformation Module.

Responsible for reading raw CSV and JSON files, correcting structural
anomalies, handling mojibake text corruption, and standardizing arbitrary nulls.

Classes:
    PySparkIngestor: Orchestrates initial data staging and normalization pipelines.
"""

import sys
import logging
import json
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import StringType
from pathlib import Path

from src import config

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

class PySparkIngestor:
    """
    Handles robust distributed ingestion of noisy tabular and JSON formats.

    Attributes:
        spark (SparkSession): PySpark engine session mapping over Arrow memory.
    """

    def __init__(self):
        """
        Instantiates Spark sessions forcing dense vector mapping representations 
        and high-availability driver memory structures.
        """
        self.spark = SparkSession.builder \
            .appName(config.SPARK_APP_NAME) \
            .master(config.SPARK_MASTER) \
            .config("spark.driver.memory", "4g") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .config("spark.sql.ansi.enabled", "false") \
            .getOrCreate()
        
    def read_and_clean_csv(self, file_path_pattern):
        """
        Sequentially rectifies corrupt header structures and structural row 
        drifts inside comma-separated text repositories.

        Args:
            file_path_pattern (str): Glob structure matching the CSV files.

        Returns:
            pyspark.sql.DataFrame: Cleansed, strictly cast PySpark dataframe 
                                   with temporally consistent values.
        """
        logger.info(f"Ingesting CSV data from: {file_path_pattern}")
        
        lines_df = self.spark.read.text(file_path_pattern)
        header_line = lines_df.first()[0]
        actual_headers = header_line.split(",")
        
        df = self.spark.read.option("header", "false") \
                            .option("escape", '"') \
                            .csv(file_path_pattern)
        
        num_header_cols = len(actual_headers)
        num_data_cols = len(df.columns)
        
        if num_data_cols > num_header_cols:
            adjusted_headers = ["synthetic_index"] + [h.strip() for h in actual_headers]
            for i, c in enumerate(df.columns):
                df = df.withColumnRenamed(c, adjusted_headers[i] if i < len(adjusted_headers) else f"extra_{i}")
        else:
            adjusted_headers = [h.strip() for h in actual_headers]
            for i, c in enumerate(df.columns):
                df = df.withColumnRenamed(c, adjusted_headers[i] if i < len(adjusted_headers) else f"extra_{i}")
        
        df = df.filter(F.col(adjusted_headers[-1]) != actual_headers[-1])
        
        null_markers = ['\\N', ',,', '', ' ']
        for col_name in df.columns:
            df = df.withColumn(
                col_name,
                F.when(F.trim(F.col(col_name)).isin(null_markers), F.lit(None)).otherwise(F.col(col_name))
            )
        
        if "startYear" in df.columns and "endYear" in df.columns:
            df = df.withColumn(
                "temp_start",
                F.when(
                    F.col("startYear").isNull() & 
                    F.col("endYear").rlike(r"^\d{4}$"), 
                    F.col("endYear")
                ).otherwise(F.col("startYear"))
            ).withColumn(
                "temp_end",
                F.when(
                    F.col("startYear").isNull() & 
                    F.col("endYear").rlike(r"^\d{4}$"), 
                    F.lit(None)
                ).otherwise(F.col("endYear"))
            )
            df = df.drop("startYear", "endYear") \
                   .withColumnRenamed("temp_start", "startYear") \
                   .withColumnRenamed("temp_end", "endYear")
        
        # Comprehensive mapping of diacritics to canonical ASCII characters natively within JVM execution
        diacritics = "ÀÁÂÃÄÅàáâãäåÒÓÔÕÖØòóôõöøÈÉÊËèéêëÇçÌÍÎÏìíîïÙÚÛÜùúûüÿÑñ"
        ascii_repl = "AAAAAAaaaaaaOOOOOOooooooEEEEeeeeCcIIIIiiiiUUUUuuuuyNn"
        
        if "primaryTitle" in df.columns:
            df = df.withColumn("primaryTitle", F.lower(F.translate(F.col("primaryTitle"), diacritics, ascii_repl)))
        if "originalTitle" in df.columns:
            df = df.withColumn("originalTitle", F.lower(F.translate(F.col("originalTitle"), diacritics, ascii_repl)))

        if "runtimeMinutes" in df.columns:
            df = df.withColumn("runtimeMinutes", F.col("runtimeMinutes").cast("int"))
        if "numVotes" in df.columns:
            df = df.withColumn("numVotes", F.col("numVotes").cast("float").cast("int"))
        if "label" in df.columns:
            df = df.withColumn("label", 
                F.when(F.lower(F.col("label")).isin("true", "1", "1.0", "yes"), F.lit(1))
                 .when(F.lower(F.col("label")).isin("false", "0", "0.0", "no"), F.lit(0))
                 .otherwise(F.col("label").cast("int"))
            )
            
        return df

    def read_json_relational(self, file_path, role="writer"):
        """
        Parses array-of-objects JSON structural layouts into Spark bindings.

        Args:
            file_path (str): Target nested JSON architecture file link.
            role (str, optional): Identity key representing the individual node role 
                                  (e.g., "writer" or "director"). Defaults to "writer".

        Returns:
            pyspark.sql.DataFrame: Rectilinear tabular format binding roles to targets.
        """
        import pandas as pd
        logger.info(f"Ingesting JSON from: {file_path}")
        
        # Leverage Pandas to immediately flatten dynamic nested dictionary strings 
        # seamlessly into row vectors, avoiding massive PySpark StructType explosions.
        pdf = pd.read_json(file_path)
        
        if "movie" in pdf.columns:
            pdf.rename(columns={"movie": "tconst"}, inplace=True)
        if role in pdf.columns:
            pdf.rename(columns={role: "nconst"}, inplace=True)
            
        # Push cleanly into the JVM leveraging Arrow bindings memory representations
        df = self.spark.createDataFrame(pdf)
            
        return df


    def run(self):
        """
        Executes sequence flows fetching and translating all heterogeneous raw models.

        Args:
            None

        Returns:
            None
        """
        csv_path = str(config.DATA_DIR / "train-*.csv")
        directing_path = str(config.DATA_DIR / "directing.json")
        writing_path = str(config.DATA_DIR / "writing.json")

        movies_df = self.read_and_clean_csv(csv_path)
        
        if Path(directing_path).exists():
            directors_df = self.read_json_relational(directing_path, role="director")
            directors_df.write.mode("overwrite").parquet(str(config.OUTPUT_DIR / "parquet" / "directing.parquet"))
            
        if Path(writing_path).exists():
            writers_df = self.read_json_relational(writing_path, role="writer")
            writers_df.write.mode("overwrite").parquet(str(config.OUTPUT_DIR / "parquet" / "writing.parquet"))

        if movies_df:
            movies_df.write.mode("overwrite").parquet(str(config.OUTPUT_DIR / "parquet" / "movies_cleaned.parquet"))
        
        logger.info("Ingestion completed. Files exported to Parquet.")
        
if __name__ == "__main__":
    ingestor = PySparkIngestor()
    ingestor.run()
