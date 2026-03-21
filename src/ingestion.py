"""
PySpark-based Data Ingestion and Transformation Module.

Responsible for reading raw CSV and JSON files, correcting structural
anomalies, handling mojibake text corruption, and standardizing arbitrary nulls.

Schema-on-read philosophy:
    Column classification (int / float / text) is performed by Pandas on a
    sampled slice of the first matching file, entirely on the driver and without
    triggering any Spark jobs. The full dataset is then read by Spark, which
    receives the pre-computed schema as input and applies per-class
    transformations in parallel across the cluster.

    This avoids the collect() round-trip that schema-on-read inside Spark would
    require, while keeping all heavy cleaning work on the distributed engine.

    Sampling ceiling: pd.read_csv(..., nrows=SAMPLE_SIZE) reads only the first
    N rows of the first glob-matched file. This is safe because:
        a) glob files are expected to share a schema, and
        b) classification only needs a representative slice, not the full dataset.
    If the files are small enough to fit in driver memory, SAMPLE_SIZE can be
    raised to cover the full file without any code changes.

Pipeline modes:
    "general"   Apply only schema-agnostic transformations (null standardisation,
                type classification, text normalisation, numeric casting).

    "imdb"      All of the above, plus two IMDb-specific steps applied before
                classification:
                    - startYear/endYear structural swap repair.
                    - "label" boolean-string vocabulary mapping ("yes"/"no" → 1/0).
                These are named exceptions: their semantics cannot be inferred
                from column content alone and must be declared explicitly.

Classes:
    PySparkIngestor: Orchestrates initial data staging and normalization pipelines.
"""

import sys
import glob
import logging
from typing import Literal

import pandas as pd
from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F
from pyspark.sql.types import StringType, IntegerType, DoubleType
from pathlib import Path

from src import config

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Pipeline mode type alias — extend this union when new modes are added.
PipelineMode = Literal["general", "imdb"]

# ---------------------------------------------------------------------------
# Transliteration tables
# ---------------------------------------------------------------------------

# Characters that map 1-to-1 (source char → single ASCII char).
# F.translate() is a Catalyst-native, codegen-friendly character substitution
# that operates at the JVM level with no Python serialization overhead.
# It cannot express 1-to-N expansions (e.g. ß → ss), so those are handled
# separately below via regexp_replace before this mapping is applied.
_DIACRITICS = "ÀÁÂÃÄÅàáâãäåÒÓÔÕÖØòóôõöøÈÉÊËèéêëÇçÌÍÎÏìíîïÙÚÛÜùúûüÿÑñ"
_ASCII_REPL  = "AAAAAAaaaaaaOOOOOOooooooEEEEeeeeCcIIIIiiiiUUUUuuuuyNn"

# Characters that expand to more than one ASCII character.
# F.translate() cannot handle these (it is strictly 1-to-1), so we resolve
# them first with regexp_replace calls, which are also Catalyst-native.
# Thanks to whole-stage code generation, Spark pipelines these passes over
# the column rather than re-scanning it once per entry.
#
# Sorted longest-key-first as a defensive measure; all current entries are
# single characters, but maintaining the ordering convention protects against
# future multi-char additions causing partial-match shadowing.
_MULTI_CHAR_MAP = {
    "ß": "ss",   # German sharp-S — the canonical example of a 1-to-2 expansion
    "æ": "ae",   # Latin ae ligature (common in Danish/Norwegian/Old English)
    "Æ": "AE",
    "œ": "oe",   # Latin oe ligature (common in French)
    "Œ": "OE",
    "þ": "th",   # Thorn (Old English / Icelandic)
    "Þ": "Th",
    "ð": "d",    # Eth (Icelandic)
    "Ð": "D",
    "ł": "l",    # Polish barred-L
    "Ł": "L",
}

# ---------------------------------------------------------------------------
# Schema-on-read classification (Pandas-side, driver-only)
# ---------------------------------------------------------------------------

# Minimum fraction of sampled non-null values that must parse as a given
# numeric type before the column is promoted to that type. Set conservatively
# so that text columns with incidental numeric values (e.g. a "rank" field
# that sometimes contains "N/A") are not misclassified.
_NUMERIC_THRESHOLD = 0.95

# Maximum number of rows read from the sample file for classification.
# Pandas reads only this many rows — the rest of the file is never loaded.
# Raise this value if your smallest file is larger and you want a fuller sample.
_SAMPLE_SIZE = 500


def _is_int_like(series: pd.Series) -> bool:
    """
    Return True if ≥ threshold fraction of non-null values parse as bare integers.

    Accepts strings that parse as integers directly, or strings that end in 
    trailing zeros (e.g., "7.0") which are commonly produced by float-casting 
    integer columns in upstream systems.
    """
    non_null = series.dropna().astype(str).str.strip()
    non_null = non_null[non_null != ""]
    if non_null.empty:
        return False

    def _parseable(v: str) -> bool:
        # Strip trailing .0, .00, etc. to treat float-formatted ints as ints
        v_clean = v.split('.')[0] if "." in v and v.split('.')[-1].replace('0', '') == '' else v
        if "e" in v_clean.lower():
            return False
        try:
            int(v_clean)
            return True
        except ValueError:
            return False

    return non_null.map(_parseable).mean() >= _NUMERIC_THRESHOLD


def _is_float_like(series: pd.Series) -> bool:
    """
    Return True if ≥ threshold fraction of non-null values parse as floats.

    Accepts integers, decimals, and scientific notation. Applied only after
    _is_int_like has already returned False.
    """
    non_null = series.dropna().astype(str).str.strip()
    non_null = non_null[non_null != ""]
    if non_null.empty:
        return False

    def _parseable(v: str) -> bool:
        try:
            float(v)
            return True
        except ValueError:
            return False

    return non_null.map(_parseable).mean() >= _NUMERIC_THRESHOLD


def _classify_columns_from_sample(sample_path: str, null_markers: list[str]) -> dict[str, str]:
    """
    Classify each column in the sample file as "int", "float", or "text".

    Reads only the first _SAMPLE_SIZE rows of a single file using Pandas —
    no Spark jobs are triggered. Classification rules mirror those used for
    Spark casting in Stage 6, so the inferred type is always consistent with
    what the cast will actually produce.

    Null standardisation is applied to the sample before classification so
    that residual sentinel values (e.g. empty strings) do not skew the
    numeric fraction calculation.

    Args:
        sample_path:  Absolute path to the first file in the glob (used as
                      the classification representative).
        null_markers: List of string values to treat as null before classifying.

    Returns:
        Dict mapping column name → "int" | "float" | "text".
    """
    # Read only the first SAMPLE_SIZE rows — the rest of the file stays on disk.
    sample_df = pd.read_csv(
        sample_path,
        nrows=_SAMPLE_SIZE,
        dtype=str,          # Keep everything as strings; we classify, not infer.
        keep_default_na=False,  # Suppress Pandas' own NA inference — we handle nulls explicitly.
    )

    # Mirror the null standardisation that Spark will apply to the full dataset,
    # so the classification sees the same effective values.
    sample_df.replace(null_markers, pd.NA, inplace=True)
    sample_df = sample_df.apply(lambda col: col.str.strip() if col.dtype == object else col)
    sample_df.replace("", pd.NA, inplace=True)

    classification: dict[str, str] = {}
    for col in sample_df.columns:
        if _is_int_like(sample_df[col]):
            classification[col] = "int"
        elif _is_float_like(sample_df[col]):
            classification[col] = "float"
        else:
            classification[col] = "text"

    return classification


# ---------------------------------------------------------------------------
# Spark-side text normalisation
# ---------------------------------------------------------------------------

def _apply_text_normalisation(df: DataFrame, col_name: str) -> DataFrame:
    """
    Apply the full text normalisation pipeline to a single column, in order:

        1. Strip leading/trailing whitespace.
        2. Collapse runs of internal whitespace to a single space.
        3. Multi-char diacritic expansion via regexp_replace (e.g. ß → ss).
        4. 1-to-1 diacritic substitution via F.translate().
        5. Lowercase the result.

    Steps 3–5 use Catalyst-native functions only; no UDFs are introduced.
    Spark's whole-stage codegen fuses the regexp_replace chain and the final
    translate+lower into a single JVM pass over the column.

    Args:
        df:       Input DataFrame.
        col_name: Name of the StringType column to normalise.

    Returns:
        DataFrame with the column replaced by its normalised form.
    """
    # Steps 1 & 2: structural whitespace cleanup.
    # trim() handles leading/trailing; regexp_replace collapses internal runs.
    df = df.withColumn(
        col_name,
        F.regexp_replace(F.trim(F.col(col_name)), r"\s{2,}", " ")
    )

    # Step 3: multi-char expansions — must run before translate() so the source
    # characters are still present when the 1-to-1 map is applied.
    for src, repl in sorted(_MULTI_CHAR_MAP.items(), key=lambda kv: len(kv[0]), reverse=True):
        df = df.withColumn(col_name, F.regexp_replace(F.col(col_name), src, repl))

    # Steps 4 & 5: bulk 1-to-1 diacritic substitution, then lowercase —
    # expressed as a single fused Catalyst expression.
    df = df.withColumn(
        col_name,
        F.lower(F.translate(F.col(col_name), _DIACRITICS, _ASCII_REPL))
    )

    return df


# ---------------------------------------------------------------------------
# Spark-side numeric casting
# ---------------------------------------------------------------------------

def _apply_int_cast(df: DataFrame, col_name: str) -> DataFrame:
    """
    Cast a string column to IntegerType.

    Strips trailing decimals (e.g. "1234.0") before casting so that values
    written as floats by upstream exporters are handled correctly. A float
    intermediate would silently round scientific notation like "1.2e6".
    """
    return df.withColumn(
        col_name,
        F.regexp_replace(F.col(col_name), r"\.\d+$", "").cast("int")
    )


def _apply_float_cast(df: DataFrame, col_name: str) -> DataFrame:
    """
    Cast a string column to DoubleType.

    A direct cast is sufficient; Spark handles standard float string formats
    including scientific notation and sign prefixes.
    """
    return df.withColumn(col_name, F.col(col_name).cast("double"))


# ---------------------------------------------------------------------------
# IMDb-specific pre-classification repairs
# ---------------------------------------------------------------------------

def _repair_temporal_swap(df: DataFrame) -> DataFrame:
    """
    Repair the IMDb-specific startYear/endYear column swap.

    Some IMDb export variants write a standalone year into endYear when
    startYear is absent (e.g. one-off specials without a date range). Detect
    this pattern and move the value to startYear so downstream year
    comparisons remain consistent.

    Must run before classification: the classifier must see values in their
    correct columns to infer the int type accurately.
    """
    if "startYear" not in df.columns or "endYear" not in df.columns:
        return df

    df = df.withColumn(
        "temp_start",
        F.when(
            F.col("startYear").isNull() & F.col("endYear").rlike(r"^\d{4}$"),
            F.col("endYear")
        ).otherwise(F.col("startYear"))
    ).withColumn(
        "temp_end",
        F.when(
            F.col("startYear").isNull() & F.col("endYear").rlike(r"^\d{4}$"),
            F.lit(None)
        ).otherwise(F.col("endYear"))
    )

    return df.drop("startYear", "endYear") \
             .withColumnRenamed("temp_start", "startYear") \
             .withColumnRenamed("temp_end", "endYear")


def _map_label_column(df: DataFrame) -> DataFrame:
    """
    Map IMDb boolean-string label vocabulary to "1"/"0" string values.

    "label" carries vocabulary ("yes"/"no"/etc.) that a generic numeric
    classifier would misread (e.g. "yes" → null after int cast). Mapping to
    string "1"/"0" before classification ensures the classifier sees int-like
    values and promotes the column to int, which is then cast in Stage 6
    without any special-cased logic.

    Unexpected values are set to null explicitly rather than passed through,
    making schema violations visible downstream.
    """
    if "label" not in df.columns:
        return df

    return df.withColumn(
        "label",
        F.when(F.lower(F.col("label")).isin("true",  "1", "1.0", "yes"), F.lit("1"))
         .when(F.lower(F.col("label")).isin("false", "0", "0.0", "no"),  F.lit("0"))
         .otherwise(F.lit(None))
    )


# ---------------------------------------------------------------------------
# Main ingestor class
# ---------------------------------------------------------------------------

class PySparkIngestor:
    """
    Handles robust distributed ingestion of noisy tabular and JSON formats.

    Args:
        pipeline: Controls which domain-specific pre-classification repairs
                  are applied before schema-on-read type inference.

                  "general" — schema-agnostic only (null standardisation,
                              classification, text normalisation, numeric casting).

                  "imdb"    — all of the above, plus:
                                  - startYear/endYear structural swap repair.
                                  - "label" boolean-string vocabulary mapping.

    Attributes:
        spark    (SparkSession): Active PySpark engine session.
        pipeline (PipelineMode): Active pipeline mode.
    """

    def __init__(self, pipeline: PipelineMode = "general"):
        if pipeline not in ("general", "imdb"):
            raise ValueError(f"Unknown pipeline mode '{pipeline}'. Expected 'general' or 'imdb'.")
        self.pipeline = pipeline

        self.spark = SparkSession.builder \
            .appName(config.SPARK_APP_NAME) \
            .master(config.SPARK_MASTER) \
            .config("spark.driver.memory", "4g") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .config("spark.sql.ansi.enabled", "false") \
            .getOrCreate()

    def read_and_clean_csv(self, file_path_pattern: str) -> DataFrame:
        """
        Reads a glob of CSV files, repairs structural anomalies, standardises
        null representations, classifies columns by inferred type, and applies
        appropriate transformations without requiring hardcoded column names
        (except for pipeline-mode-gated named exceptions).

        Pipeline stages:
            1.  Header extraction and column renaming (Spark).
            2.  Null standardisation (Spark).
            3.  [imdb only] Temporal column swap repair (Spark).
            4.  [imdb only] Label vocabulary mapping (Spark).
            5.  Column classification via Pandas sample (driver-only, no Spark job).
            6a. Text normalisation for text-like columns (Spark).
            6b. Int casting for int-like columns (Spark).
            6c. Float casting for float-like columns (Spark).

        Args:
            file_path_pattern: Glob pattern matching one or more CSV files.

        Returns:
            Cleaned PySpark DataFrame with inferred types applied.
        """
        logger.info(f"[{self.pipeline.upper()}] Ingesting CSV data from: {file_path_pattern}")

        # -----------------------------------------------------------------------
        # Stage 1 — Header extraction and column renaming
        # -----------------------------------------------------------------------

        # Read as raw text first to extract the true header line before Spark
        # has a chance to infer (or misalign) the schema.
        lines_df = self.spark.read.text(file_path_pattern)
        header_line = lines_df.first()[0]
        
        def to_camel_case(s: str) -> str:
            """Standardize snake_case or PascalCase to camelCase."""
            s = s.strip().replace('"', '')
            if not s: return s
            parts = s.split('_')
            if len(parts) == 1:
                return s[0].lower() + s[1:] if s else s
            return parts[0].lower() + "".join(p.capitalize() for p in parts[1:])

        actual_headers = [to_camel_case(h) for h in header_line.split(",")]

        # Read without header so every row — including the header row — becomes
        # a data row. This lets us detect and handle the synthetic index column
        # that some IMDb export variants prepend.
        df = self.spark.read \
            .option("header", "false") \
            .option("escape", '"') \
            .option("nullValue", "\\N") \
            .csv(file_path_pattern)
            # nullValue at read time covers IMDb's \N sentinel. Doing it here
            # avoids an extra withColumn pass and ensures the null is typed
            # correctly from the moment it enters Spark.

        num_header_cols = len(actual_headers)
        num_data_cols   = len(df.columns)

        logger.info(
            f"   -> [STRUCTURAL]: Schema alignment — "
            f"header fields: {num_header_cols} | data columns: {num_data_cols}"
        )


        if num_data_cols > num_header_cols:
            # Extra leading column indicates a synthetic row index prepended by
            # the exporter. Prepend a placeholder name to keep the zip aligned.
            adjusted_headers = ["syntheticIndex"] + [h.strip() for h in actual_headers]
        else:
            adjusted_headers = [h.strip() for h in actual_headers]

        for i, c in enumerate(df.columns):
            df = df.withColumnRenamed(
                c,
                adjusted_headers[i] if i < len(adjusted_headers) else f"extraCol{i}"
            )

        # Drop rows where the last column still contains the header value —
        # these are duplicate header rows baked in by some export tools.
        df = df.filter(F.col(adjusted_headers[-1]) != actual_headers[-1])

        # -----------------------------------------------------------------------
        # Stage 2 — Null standardisation
        # -----------------------------------------------------------------------
        logger.info("   -> [CLEANING]: Standardising null representations.")

        # nullValue="\\N" covers IMDb's primary sentinel. This pass catches
        # secondary patterns (empty string, bare whitespace, stray commas) that
        # the CSV reader does not treat as nulls by default.
        null_markers = [",,", "", " "]
        for col_name in df.columns:
            df = df.withColumn(
                col_name,
                F.when(F.trim(F.col(col_name)).isin(null_markers), F.lit(None))
                 .otherwise(F.col(col_name))
            )

        # -----------------------------------------------------------------------
        # Stages 3 & 4 — IMDb-specific pre-classification repairs (gated)
        # -----------------------------------------------------------------------

        if self.pipeline == "imdb":
            logger.info("   -> [IMDB]: Applying domain-specific pre-classification repairs.")
            logger.info("   -> [IMDB]: Dropping primaryTitle and originalTitle sequentially.")
            df = df.drop("primaryTitle", "originalTitle")

            # Stage 3: temporal swap — must precede classification so that year
            # values are in the correct columns when the int classifier runs.
            df = _repair_temporal_swap(df)

            # Stage 4: label mapping — converts "yes"/"no" vocabulary to "1"/"0"
            # strings so the classifier promotes "label" to int, not text.
            df = _map_label_column(df)

        # -----------------------------------------------------------------------
        # Stage 5 — Schema-on-read classification (Pandas, driver-only)
        # -----------------------------------------------------------------------
        logger.info(
            f"   -> [CLASSIFICATION]: Sampling up to {_SAMPLE_SIZE} rows via Pandas "
            f"(threshold: {_NUMERIC_THRESHOLD:.0%}, no Spark job triggered)."
        )

        # Resolve the glob to find the first file for the Pandas sample.
        # All files in a glob are expected to share a schema, so sampling one
        # is representative of the whole set.
        matched_files = sorted(glob.glob(file_path_pattern))
        if not matched_files:
            raise FileNotFoundError(f"No files matched pattern: {file_path_pattern}")
        sample_file = matched_files[0]

        # Pandas reads only _SAMPLE_SIZE rows — the rest of the file stays on disk.
        # null_markers mirrors Stage 2 so the classifier sees the same effective values.
        classification = _classify_columns_from_sample(sample_file, null_markers)

        int_cols   = [c for c, t in classification.items() if t == "int"   and c in df.columns]
        float_cols = [c for c, t in classification.items() if t == "float" and c in df.columns]
        text_cols  = [c for c, t in classification.items() if t == "text"  and c in df.columns]

        logger.info(f"      int-like   ({len(int_cols)}):  {int_cols}")
        logger.info(f"      float-like ({len(float_cols)}): {float_cols}")
        logger.info(f"      text-like  ({len(text_cols)}):  {text_cols}")

        # -----------------------------------------------------------------------
        # Stage 6 — Per-class Spark transformations
        # -----------------------------------------------------------------------

        # Text: whitespace cleanup → multi-char expansion → translate → lower.
        logger.info("   -> [TEXT]: Normalising text-like columns.")
        for col_name in text_cols:
            df = _apply_text_normalisation(df, col_name)

        # Int: strip trailing decimals, then cast to IntegerType.
        logger.info("   -> [INT]: Casting int-like columns.")
        for col_name in int_cols:
            df = _apply_int_cast(df, col_name)

        # Float: direct cast to DoubleType.
        logger.info("   -> [FLOAT]: Casting float-like columns.")
        for col_name in float_cols:
            df = _apply_float_cast(df, col_name)

        return df

    def read_json_relational(self, file_path: str, role: str = "writer") -> DataFrame:
        """
        Parses an array-of-objects JSON file into a flat Spark DataFrame.

        Pandas is used for the initial read because Spark's built-in JSON reader
        would infer a StructType for nested fields, requiring an explicit schema
        or a costly schema-inference pass. For the flat key-value structures
        expected here, Pandas + Arrow conversion is simpler and equivalent in
        performance at this file size.

        Args:
            file_path: Path to the JSON file.
            role:      Key name identifying the person's role (e.g. "writer",
                       "director"). Renamed to "nconst" in the output schema.

        Returns:
            DataFrame with columns [tconst, nconst].
        """
        logger.info(f"Ingesting JSON from: {file_path}")

        pdf = pd.read_json(file_path)

        # Normalise column names to the shared tconst / nconst vocabulary used
        # across all relational tables so joins downstream are schema-agnostic.
        if "movie" in pdf.columns:
            pdf.rename(columns={"movie": "tconst"}, inplace=True)
        if role in pdf.columns:
            pdf.rename(columns={role: "nconst"}, inplace=True)

        # Arrow is enabled on the session, so createDataFrame uses zero-copy
        # columnar transfer from Pandas to JVM memory where possible.
        return self.spark.createDataFrame(pdf)

    def run(self, target_pattern: str = "train-*.csv") -> None:
        """
        Orchestrates ingestion of all heterogeneous source files and writes
        cleaned outputs to Parquet.

        Args:
            target_pattern: Filename wildcard identifying the main CSV tables.
        """
        csv_path       = str(config.DATA_DIR / target_pattern)
        directing_path = str(config.DATA_DIR / "directing.json")
        writing_path   = str(config.DATA_DIR / "writing.json")

        movies_df = self.read_and_clean_csv(csv_path)

        if Path(directing_path).exists():
            directors_df = self.read_json_relational(directing_path, role="director")
            directors_df.write.mode("overwrite").parquet(
                str(config.PARQUET_DIR / "directing.parquet")
            )

        if Path(writing_path).exists():
            writers_df = self.read_json_relational(writing_path, role="writer")
            writers_df.write.mode("overwrite").parquet(
                str(config.PARQUET_DIR / "writing.parquet")
            )

        # DataFrames are lazily evaluated; the write action triggers the full
        # DAG execution. Checking truthiness on a DataFrame is not meaningful —
        # guard on the path pattern instead if conditional writes are needed.
        movies_df.write.mode("overwrite").parquet(
            str(config.PARQUET_DIR / "movies_cleaned.parquet")
        )

        logger.info("Ingestion complete. Files exported to Parquet.")


if __name__ == "__main__":
    ingestor = PySparkIngestor(pipeline="imdb")
    ingestor.run()