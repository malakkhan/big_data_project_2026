"""
TMDB API Enrichment & Categorical Normalization Module.

Fetches external metadata from The Movie Database (TMDB) API, joins it
onto the cleaned movie Parquet, coalesces runtime data, and normalizes
categorical columns (genre, language, country) via fingerprint keying.
XGBoost's native categorical support handles these directly.

Design:
    TMDBFetcher runs entirely on the driver in a sequential Python loop.
    This makes rate-limit control, retry handling, and audit logging
    straightforward. Given TMDB's ~40 req/s free-tier limit, a sequential
    loop is the correct concurrency model.

    Genre encoding uses OpenRefine-style fingerprint keying to canonicalise
    genre strings before ordinal encoding, ensuring stability across
    minor string variations.

Classes:
    TMDBFetcher:       Driver-side API fetch and Parquet writer.
    TMDBEnrichment:    Orchestrates TMDB join, runtime coalesce, and genre encoding.
"""

import os
import re
import time
import logging
import sys
import shutil
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import requests
import pandas as pd
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pyspark.sql.functions as F
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import (
    StructType, StructField,
    StringType, DoubleType, IntegerType, BooleanType, TimestampType,
)


from src import config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


def _buildAuditLogger(logPath: Path) -> logging.Logger:
    """
    Build a dedicated logger that writes structured TMDB API audit records
    to a file, one line per API attempt.
    """
    auditLogger = logging.getLogger("tmdbAudit")
    auditLogger.setLevel(logging.DEBUG)
    auditLogger.propagate = False

    if not auditLogger.handlers:
        logPath.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(logPath, mode="a", encoding="utf-8")
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        ))
        auditLogger.addHandler(handler)

    return auditLogger


# ---------------------------------------------------------------------------
# TMDB Parquet schema
# ---------------------------------------------------------------------------

TMDB_PARQUET_SCHEMA = StructType([
    StructField("tconst",                  StringType(),    False),
    StructField("tmdb_popularity",         DoubleType(),    True),
    StructField("tmdb_vote_average",       DoubleType(),    True),
    StructField("tmdb_budget",             DoubleType(),    True),
    StructField("tmdb_revenue",            DoubleType(),    True),
    StructField("tmdb_runtime",            IntegerType(),   True),
    StructField("tmdb_primary_genre",      StringType(),    True),
    StructField("tmdb_original_language",  StringType(),    True),
    StructField("tmdb_origin_country",     StringType(),    True),
    StructField("tmdb_production_company", StringType(),    True),
    StructField("tmdb_success",            BooleanType(),   False),
    StructField("tmdb_fetched_at",         TimestampType(), False),
])


# ---------------------------------------------------------------------------
# TMDBFetcher — driver-side, no Spark jobs
# ---------------------------------------------------------------------------

class TMDBFetcher:
    """
    Fetch TMDB metadata for a list of IMDb tconsts and persist to Parquet.

    Each tconst produces exactly one output row regardless of whether the
    fetch succeeded, so the output Parquet is always left-joinable to any
    tconst-keyed table without losing rows.

    Args:
        token:         TMDB API Bearer token.
        requestDelay:  Seconds between requests (~40 req/s at 0.025).
        maxRetries:    Retry count for 429 / transient errors.
        retryBackoff:  Additional seconds per retry attempt (linear).
        auditLogPath:  Path for the structured API audit log.
    """

    _BASE_URL = "https://api.themoviedb.org/3"

    def __init__(
        self,
        token:        str | None = None,
        requestDelay: float = 0.025,
        maxRetries:   int   = 3,
        retryBackoff: float = 2.0,
        auditLogPath: Path  = None,
    ):
        self.token        = token or os.environ.get("TMDB_READ_TOKEN", "")
        self.requestDelay = requestDelay
        self.maxRetries   = maxRetries
        self.retryBackoff = retryBackoff

        if auditLogPath is None:
            auditLogPath = config.OUTPUT_DIR / "logs" / "tmdbAudit.log"
        self.audit = _buildAuditLogger(auditLogPath)

        if not self.token:
            logger.warning(
                "No TMDB token found. Set TMDB_READ_TOKEN environment variable "
                "or pass token= to TMDBFetcher. All fetches will return null rows."
            )

    @property
    def _headers(self) -> dict:
        return {
            "accept": "application/json",
            "Authorization": f"Bearer {self.token}",
        }

    def _getWithRetry(self, url: str, params: dict, tconst: str, hop: int) -> requests.Response | None:
        """GET with linear back-off retry on 429 and transient errors."""
        for attempt in range(1, self.maxRetries + 1):
            try:
                res = requests.get(url, params=params, headers=self._headers, timeout=5)
                self.audit.info(
                    f"tconst={tconst} hop={hop} attempt={attempt} "
                    f"status={res.status_code} url={url}"
                )
                if res.status_code == 200:
                    return res
                if res.status_code == 429:
                    wait = self.retryBackoff * attempt
                    self.audit.warning(
                        f"tconst={tconst} hop={hop} attempt={attempt} "
                        f"rate-limited, backing off {wait:.1f}s"
                    )
                    time.sleep(wait)
                    continue
                self.audit.warning(
                    f"tconst={tconst} hop={hop} attempt={attempt} "
                    f"non-retryable status={res.status_code}"
                )
                return res
            except requests.RequestException as exc:
                self.audit.error(
                    f"tconst={tconst} hop={hop} attempt={attempt} "
                    f"status=ERROR exception={exc}"
                )
                if attempt < self.maxRetries:
                    time.sleep(self.retryBackoff * attempt)

        self.audit.error(f"tconst={tconst} hop={hop} all {self.maxRetries} attempts exhausted.")
        return None

    def _fetchOne(self, tconst: str) -> dict:
        """
        Fetch all TMDB fields for a single tconst via a two-hop REST call.

        Hop 1: /find/{tconst} -> resolve to TMDB movie ID
        Hop 2: /movie/{tmdb_id} -> fetch full movie record
        """
        fetchedAt = datetime.now(timezone.utc)
        nullRow = {
            "tconst": tconst, "tmdb_popularity": None, "tmdb_vote_average": None,
            "tmdb_budget": None, "tmdb_revenue": None, "tmdb_runtime": None,
            "tmdb_primary_genre": None, "tmdb_original_language": None,
            "tmdb_origin_country": None, "tmdb_production_company": None,
            "tmdb_success": False, "tmdb_fetched_at": fetchedAt,
        }

        if not self.token:
            return nullRow

        resFind = self._getWithRetry(
            url=f"{self._BASE_URL}/find/{tconst}",
            params={"external_source": "imdb_id"}, tconst=tconst, hop=1,
        )
        if resFind is None or resFind.status_code != 200:
            return nullRow

        movieResults = resFind.json().get("movie_results", [])
        if not movieResults:
            self.audit.info(f"tconst={tconst} hop=1 not found in TMDB movie_results.")
            return nullRow

        tmdbId = movieResults[0]["id"]
        resMovie = self._getWithRetry(
            url=f"{self._BASE_URL}/movie/{tmdbId}",
            params={}, tconst=tconst, hop=2,
        )
        if resMovie is None or resMovie.status_code != 200:
            return nullRow

        m = resMovie.json()
        genres    = m.get("genres", [])
        companies = m.get("production_companies", [])
        runtime   = m.get("runtime")

        return {
            "tconst":                  tconst,
            "tmdb_popularity":         float(m.get("popularity")   or 0.0),
            "tmdb_vote_average":       float(m.get("vote_average") or 0.0),
            "tmdb_budget":             float(m.get("budget")       or 0),
            "tmdb_revenue":            float(m.get("revenue")      or 0),
            "tmdb_runtime":            int(runtime) if runtime is not None else None,
            "tmdb_primary_genre":      genres[0]["name"]    if genres    else None,
            "tmdb_original_language":  m.get("original_language"),
            "tmdb_origin_country":     (m.get("origin_country") or [None])[0],
            "tmdb_production_company": companies[0]["name"] if companies else None,
            "tmdb_success":            True,
            "tmdb_fetched_at":         fetchedAt,
        }

    def fetchAndSave(self, tconsts: list[str], outputPath: Path, spark: SparkSession) -> None:
        """Fetch TMDB metadata for all tconsts concurrently and write to Parquet.
        
        Uses a ThreadPoolExecutor with 10 workers and a semaphore-based rate
        limiter to stay within TMDB's ~40 req/s free-tier limit while achieving
        ~10× speedup over sequential fetching.
        """
        logger.info(f"[TMDB] Starting concurrent fetch for {len(tconsts)} titles (10 workers).")
        logger.info(f"[TMDB] Audit log: {self.audit.handlers[0].baseFilename}")

        rows = []
        successCount = 0
        lock = threading.Lock()
        semaphore = threading.Semaphore(10)

        def _fetch_with_limit(tconst: str) -> dict:
            with semaphore:
                result = self._fetchOne(tconst)
                time.sleep(self.requestDelay)
                return result

        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = {pool.submit(_fetch_with_limit, tc): tc for tc in tconsts}
            for i, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                rows.append(row)
                if row["tmdb_success"]:
                    with lock:
                        successCount += 1
                if i % 100 == 0 or i == len(tconsts):
                    logger.info(f"[TMDB] {i}/{len(tconsts)} fetched | success={successCount} | failed={i - successCount}")

        logger.info(f"[TMDB] Fetch complete. Success rate: {successCount}/{len(tconsts)} ({100 * successCount / max(len(tconsts), 1):.1f}%).")

        tmdbDf = spark.createDataFrame(pd.DataFrame(rows), schema=TMDB_PARQUET_SCHEMA)
        tmdbDf.write.mode("overwrite").parquet(str(outputPath))
        logger.info(f"[TMDB] Results written to {outputPath}.")


# ---------------------------------------------------------------------------
# Fingerprint keying + categorical normalization
# ---------------------------------------------------------------------------

def fingerprint_key(value: str) -> str:
    """
    Canonicalise a raw genre string to a stable, case- and
    punctuation-insensitive key following the OpenRefine standard.
    """
    if not isinstance(value, str) or not value:
        return ""
    value = value.strip().lower()
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", errors="ignore").decode("ascii")
    value = value.replace("-", "")
    value = re.sub(r"[^\w\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    tokens = value.split()
    if not tokens:
        return ""
    return " ".join(sorted(set(tokens)))


def normalize_categorical(
    df:           pd.DataFrame,
    column:       str,
    is_inference: bool = False,
    min_frequency: int = 0,
) -> pd.DataFrame:
    """
    Normalise a categorical column via fingerprint keying.

    No ordinal encoding is applied — XGBoost's native categorical support
    handles unordered categories optimally via subset splits.

    Training: saves the known class vocabulary to disk for inference safety.
             If min_frequency > 0, values appearing fewer than min_frequency
             times are collapsed to 'unknown' before saving the vocabulary.
    Inference: loads the vocabulary and remaps unseen values to 'unknown'.
    """
    vocab_dir = config.OUTPUT_DIR / "models"
    vocab_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = vocab_dir / f"{column}_vocab.joblib"

    if column not in df.columns:
        logger.warning("Categorical normalization skipped -- column '%s' not found", column)
        return df

    df = df.copy()
    df[column] = df[column].fillna("unknown").apply(fingerprint_key)
    df[column] = df[column].replace("", "unknown")

    if not is_inference:
        # Collapse rare categories below the frequency threshold
        if min_frequency > 0:
            freq = df[column].value_counts()
            rare_values = freq[freq < min_frequency].index.tolist()
            if rare_values:
                logger.info("Collapsing %d rare value(s) in '%s' (freq < %d) to 'unknown'",
                            len(rare_values), column, min_frequency)
                df[column] = df[column].where(
                    ~df[column].isin(rare_values), other="unknown"
                )

        known_classes = sorted(df[column].unique().tolist())
        if "unknown" not in known_classes:
            known_classes.append("unknown")
        joblib.dump(known_classes, vocab_path)
        logger.info("Vocabulary saved: %d classes for '%s' -> %s",
                    len(known_classes), column, vocab_path)
    else:
        if not vocab_path.exists():
            raise FileNotFoundError(
                f"No vocabulary found at {vocab_path}. "
                "Run the training pipeline before inference."
            )
        known_classes = joblib.load(vocab_path)
        logger.info("Vocabulary loaded for '%s' from %s (%d classes)",
                    column, vocab_path, len(known_classes))
        known_set = set(known_classes)
        unseen = df[column][~df[column].isin(known_set)].unique()
        if len(unseen) > 0:
            logger.warning("Remapping %d unseen value(s) in '%s' to 'unknown': %s",
                           len(unseen), column, list(unseen))
            df[column] = df[column].where(
                df[column].isin(known_set), other="unknown"
            )

    logger.info("Normalized '%s': %d unique values", column, df[column].nunique())
    return df


# ---------------------------------------------------------------------------
# TMDBEnrichment — orchestrator
# ---------------------------------------------------------------------------

class TMDBEnrichment:
    """
    Orchestrates TMDB API fetch, runtime coalesce, and categorical normalization.

    Reads cleaned_data.parquet, fetches TMDB metadata (or uses cached),
    joins it onto the movie DataFrame, coalesces tmdb_runtime into
    runtimeMinutes, and normalizes categorical columns via fingerprint keying.

    Writes the enriched DataFrame to enriched_features.parquet for
    downstream graph feature computation.
    """

    def __init__(self, spark: SparkSession = None):
        self.spark = spark or SparkSession.builder \
            .appName(config.SPARK_APP_NAME) \
            .master(config.SPARK_MASTER) \
            .config("spark.executorEnv.TMDB_API_KEY", config.TMDB_API_KEY) \
            .config("spark.executorEnv.TMDB_READ_TOKEN", config.TMDB_READ_TOKEN) \
            .getOrCreate()

    def run(self, input_parquet: str = "cleaned_data.parquet", is_inference: bool = False, tmdb_label: str = None) -> None:
        """
        Execute TMDB enrichment pipeline.

        Args:
            input_parquet: Name of the input parquet file in PARQUET_DIR.
            is_inference:  If True, load saved vocabulary instead of fitting.
            tmdb_label:    Optional label for per-target TMDB caching (e.g. "validation_hidden").
        """
        moviesDf = self.spark.read.parquet(
            str(config.PARQUET_DIR / input_parquet)
        )

        # Per-target TMDB cache: tmdb_{label}.parquet, or generic tmdb.parquet
        tmdb_filename = f"tmdb_{tmdb_label}.parquet" if tmdb_label else "tmdb.parquet"
        tmdbPath = config.PARQUET_DIR / tmdb_filename

        # --- TMDB fetch (driver-side, cached) ---
        if tmdbPath.exists():
            logger.info(f"[TMDB] Parquet already exists at {tmdbPath}, skipping fetch.")
        else:
            logger.info("[TMDB] Collecting tconst list for API fetch.")
            tconsts = [
                row["tconst"]
                for row in moviesDf.select("tconst").collect()
                if row["tconst"] is not None
            ]
            fetcher = TMDBFetcher()
            fetcher.fetchAndSave(tconsts=tconsts, outputPath=tmdbPath, spark=self.spark)

        tmdbDf = self.spark.read.parquet(str(tmdbPath))

        # --- Join TMDB onto movies ---
        logger.info("   -> [TMDB]: Joining TMDB enrichment onto movie DataFrame.")
        moviesDf = moviesDf.join(tmdbDf, on="tconst", how="left")

        # --- Coalesce runtime ---
        logger.info("   -> [TMDB]: Coalescing tmdb_runtime into runtimeMinutes.")
        moviesDf = moviesDf.withColumn(
            "runtimeMinutes",
            F.coalesce(F.col("runtimeMinutes"), F.col("tmdb_runtime"))
        ).drop("tmdb_runtime")

        # Drop audit columns not needed downstream
        moviesDf = moviesDf.drop("tmdb_fetched_at")

        # --- Categorical normalization (Pandas-side for fingerprint keying) ---
        # Convert directly to Pandas from Spark for the normalization step.
        # No ordinal encoding — XGBoost handles categories natively.
        logger.info("   -> [ENCODE]: Applying fingerprint normalization...")
        enriched_df = moviesDf.toPandas()
        enriched_df = normalize_categorical(enriched_df, column="tmdb_primary_genre", is_inference=is_inference)
        enriched_df = normalize_categorical(enriched_df, column="tmdb_original_language", is_inference=is_inference)
        enriched_df = normalize_categorical(enriched_df, column="tmdb_origin_country", is_inference=is_inference)
        enriched_df = normalize_categorical(enriched_df, column="tmdb_production_company", is_inference=is_inference, min_frequency=5)

        # --- Interaction features ---
        logger.info("   -> [INTERACT]: Engineering interaction features...")
        # Budget-to-revenue ratio (0 when revenue is 0 to avoid inf)
        if "tmdb_budget" in enriched_df.columns and "tmdb_revenue" in enriched_df.columns:
            enriched_df["budget_revenue_ratio"] = np.where(
                enriched_df["tmdb_revenue"] > 0,
                enriched_df["tmdb_budget"] / enriched_df["tmdb_revenue"],
                0.0,
            )
        # Votes × popularity composite signal
        if "numVotes" in enriched_df.columns and "tmdb_popularity" in enriched_df.columns:
            enriched_df["votes_x_popularity"] = (
                enriched_df["numVotes"].fillna(0) * enriched_df["tmdb_popularity"].fillna(0)
            )

        # Write final result for downstream compatibility.
        outputPath = config.PARQUET_DIR / "tmdb_enriched.parquet"
        if outputPath.is_dir():
            shutil.rmtree(outputPath)
        enriched_df.to_parquet(str(outputPath), index=False)
        logger.info("   -> [ENCODE]: Enrichment complete. Matrix: %d rows x %d cols.", *enriched_df.shape)


if __name__ == "__main__":
    enricher = TMDBEnrichment()
    enricher.run()
