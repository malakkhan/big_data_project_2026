"""
Graph & Network Topological Feature Engineering.

Calculates degree centralities, collaborative frequencies (team synergies),
and enriches the feature set with external TMDB API metadata.

Design overview:
    The module is split into two cooperating classes:

    TMDBFetcher
        Runs entirely on the driver. Reads the tconst list from the cleaned
        movie Parquet, calls the TMDB REST API in a sequential Python loop
        with proper rate-limit handling, and writes the results to a dedicated
        TMDB Parquet file. Each row records the fetched fields alongside a
        success flag and a fetch timestamp, making the audit trail queryable
        as ordinary DataFrame columns.

        A structured audit log (tmdbAudit.log) is also written to the output
        directory, with one line per API attempt so failures can be reviewed
        independently of the main pipeline log.

    GraphFeatureExtractor
        Runs Spark jobs. Reads the cleaned movie Parquet, the crew relation
        Parquets, and the TMDB Parquet written by TMDBFetcher, then computes
        graph-derived features and joins everything into a single feature
        matrix keyed on tconst.

Why no UDFs for the API calls:
    Executing HTTP requests inside a Spark UDF scatters the fetch work across
    executors, making rate-limit control, retry logic, and audit logging all
    but impossible. Running the fetch loop on the driver gives full control
    over concurrency, back-off, and observability at the cost of losing
    parallelism for the I/O — an acceptable tradeoff given TMDB's rate limits
    (~40 req/s) already constrain throughput more than a single-threaded loop.

Feature families produced:
    Degree centrality          — experience score per person, aggregated per movie.
    Writer-writer synergy      — co-credit frequency of writer pairs.
    Writer-director synergy    — co-credit frequency of writer/director pairs.
    TMDB enrichment            — budget, revenue, popularity, genre, success flag,
                                 and fetch timestamp.

Classes:
    TMDBFetcher:           Driver-side API fetch and Parquet writer.
    GraphFeatureExtractor: Spark-side graph feature computation and final join.
"""

import os
import time
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
import pandas as pd

import pyspark.sql.functions as F
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import (
    StructType, StructField,
    StringType, DoubleType, IntegerType, BooleanType, TimestampType,
)

from src import config

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

# Main pipeline logger — writes to stdout at INFO level, consistent with the
# rest of the pipeline.
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


def _buildAuditLogger(logPath: Path) -> logging.Logger:
    """
    Build a dedicated logger that writes structured TMDB API audit records to
    a file, one line per API attempt.

    The audit logger is separate from the main pipeline logger so that API
    call history can be reviewed and queried independently without filtering
    through unrelated pipeline messages.

    Log format per line:
        <ISO timestamp> | <level> | tconst=<id> hop=<1|2> status=<code|ERROR> msg=<detail>

    Args:
        logPath: Absolute path to the audit log file. Created if absent;
                 appended to if it already exists (so reruns accumulate history).

    Returns:
        A Logger instance with a FileHandler attached.
    """
    auditLogger = logging.getLogger("tmdbAudit")
    auditLogger.setLevel(logging.DEBUG)
    auditLogger.propagate = False  # Don't duplicate records into the root logger.

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

# Declared at module level so both TMDBFetcher (writer) and GraphFeatureExtractor
# (reader) share the same schema definition without either importing the other.
# Column names use snake_case because they are Parquet/DataFrame field names,
# not Python identifiers — keeping them lowercase with underscores matches
# the convention of the rest of the feature matrix.
TMDB_PARQUET_SCHEMA = StructType([
    StructField("tconst",                  StringType(),    False),
    StructField("tmdb_popularity",         DoubleType(),    True),
    StructField("tmdb_vote_average",       DoubleType(),    True),
    StructField("tmdb_budget",             DoubleType(),    True),
    StructField("tmdb_revenue",            DoubleType(),    True),
    StructField("tmdb_runtime",            IntegerType(),   True),
    StructField("tmdb_primary_genre",      StringType(),    True),
    StructField("tmdb_production_company", StringType(),    True),
    StructField("tmdb_success",            BooleanType(),   False),
    StructField("tmdb_fetched_at",         TimestampType(), False),
])


# ---------------------------------------------------------------------------
# TMDBFetcher — driver-side, no Spark jobs
# ---------------------------------------------------------------------------

class TMDBFetcher:
    """
    Fetch TMDB metadata for a list of IMDb tconsts and persist the results
    to Parquet.

    Runs entirely on the driver in a sequential Python loop. This makes
    rate-limit control, retry handling, and audit logging straightforward
    at the cost of not parallelising the HTTP I/O. Given TMDB's ~40 req/s
    free-tier limit, a sequential loop is the correct concurrency model.

    Each tconst produces exactly one output row regardless of whether the
    fetch succeeded, so the output Parquet is always left-joinable to any
    tconst-keyed table without losing rows.

    Args:
        token:         TMDB API Bearer token. Defaults to the TMDB_READ_TOKEN
                       environment variable if not provided explicitly.
        requestDelay:  Seconds to wait between requests. At 0.025 s the loop
                       runs at ~40 req/s, matching the TMDB free-tier limit.
                       Increase this if you observe 429 responses.
        maxRetries:    Number of times to retry a 429 or transient error
                       before recording the attempt as failed.
        retryBackoff:  Additional seconds added per retry attempt (linear).
        auditLogPath:  Path for the structured API audit log file.
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
        """
        Execute a GET request with linear back-off retry on 429 and transient errors.

        Each attempt is recorded in the audit log with its HTTP status code
        (or "ERROR" on exception). Only the final outcome after all retries
        is considered for the success flag.

        Args:
            url:    Full request URL.
            params: Query parameters dict.
            tconst: IMDb ID being fetched (for audit log context only).
            hop:    1 (find) or 2 (movie detail), for audit log context only.

        Returns:
            The final Response object, or None if all retries were exhausted
            or an unrecoverable exception occurred.
        """
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
                    # Rate limited — back off and retry.
                    wait = self.retryBackoff * attempt
                    self.audit.warning(
                        f"tconst={tconst} hop={hop} attempt={attempt} "
                        f"rate-limited, backing off {wait:.1f}s"
                    )
                    time.sleep(wait)
                    continue

                # Any other non-200 status is treated as a permanent failure
                # for this attempt — no point retrying a 404 or 401.
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

        self.audit.error(
            f"tconst={tconst} hop={hop} all {self.maxRetries} attempts exhausted."
        )
        return None

    def _fetchOne(self, tconst: str) -> dict:
        """
        Fetch all TMDB fields for a single tconst via a two-hop REST call.

        Hop 1 — /find/{tconst}?external_source=imdb_id
            Resolves the IMDb tconst to a TMDB internal movie ID.

        Hop 2 — /movie/{tmdb_id}
            Fetches the full movie record (budget, revenue, genres, etc.).

        Returns a dict with all TMDB_PARQUET_SCHEMA fields populated.
        tmdb_success is True only if both hops completed with status 200
        and a movie result was found. tmdb_fetched_at is always set to the
        time this function was called, regardless of success, so the audit
        trail is complete even for failed attempts.

        Args:
            tconst: IMDb title identifier (e.g. "tt0111161").

        Returns:
            Dict with keys matching TMDB_PARQUET_SCHEMA field names.
        """
        fetchedAt = datetime.now(timezone.utc)

        # Base null row — returned on any failure path. tconst and timestamp
        # are always populated so the row is joinable and auditable.
        nullRow = {
            "tconst":                  tconst,
            "tmdb_popularity":         None,
            "tmdb_vote_average":       None,
            "tmdb_budget":             None,
            "tmdb_revenue":            None,
            "tmdb_runtime":            None,
            "tmdb_primary_genre":      None,
            "tmdb_production_company": None,
            "tmdb_success":            False,
            "tmdb_fetched_at":         fetchedAt,
        }

        if not self.token:
            return nullRow

        # Hop 1: resolve tconst → TMDB movie ID.
        resFind = self._getWithRetry(
            url    = f"{self._BASE_URL}/find/{tconst}",
            params = {"external_source": "imdb_id"},
            tconst = tconst,
            hop    = 1,
        )
        if resFind is None or resFind.status_code != 200:
            return nullRow

        movieResults = resFind.json().get("movie_results", [])
        if not movieResults:
            # Title exists in IMDb but not in TMDB — not an error, just absent.
            self.audit.info(f"tconst={tconst} hop=1 not found in TMDB movie_results.")
            return nullRow

        tmdbId = movieResults[0]["id"]

        # Hop 2: fetch the full movie record.
        resMovie = self._getWithRetry(
            url    = f"{self._BASE_URL}/movie/{tmdbId}",
            params = {},
            tconst = tconst,
            hop    = 2,
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
            "tmdb_production_company": companies[0]["name"] if companies else None,
            "tmdb_success":            True,
            "tmdb_fetched_at":         fetchedAt,
        }

    def fetchAndSave(
        self,
        tconsts:    list[str],
        outputPath: Path,
        spark:      SparkSession,
    ) -> None:
        """
        Fetch TMDB metadata for all tconsts and write the results to Parquet.

        The fetch loop runs sequentially on the driver. Progress is logged to
        the main pipeline logger every 100 titles. All per-request events are
        written to the audit log file by _fetchOne / _getWithRetry.

        After the loop, the collected rows are written to Parquet via Spark
        (rather than Pandas) so the schema is enforced against TMDB_PARQUET_SCHEMA
        and the file is immediately readable by GraphFeatureExtractor without
        any schema coercion.

        Args:
            tconsts:    List of IMDb tconst strings to fetch.
            outputPath: Destination Parquet path (overwritten if it exists).
            spark:      Active SparkSession used for schema-enforced Parquet write.
        """
        logger.info(f"[TMDB] Starting fetch loop for {len(tconsts)} titles.")
        logger.info(f"[TMDB] Audit log: {self.audit.handlers[0].baseFilename}")

        rows = []
        successCount = 0

        for i, tconst in enumerate(tconsts, start=1):
            row = self._fetchOne(tconst)
            rows.append(row)

            if row["tmdb_success"]:
                successCount += 1

            # Pace requests to stay within the TMDB rate limit.
            time.sleep(self.requestDelay)

            if i % 100 == 0 or i == len(tconsts):
                logger.info(
                    f"[TMDB] {i}/{len(tconsts)} fetched | "
                    f"success={successCount} | "
                    f"failed={i - successCount}"
                )

        logger.info(
            f"[TMDB] Fetch complete. "
            f"Success rate: {successCount}/{len(tconsts)} "
            f"({100 * successCount / max(len(tconsts), 1):.1f}%)."
        )

        # Write via Spark with the declared schema so field types are enforced.
        # Using Pandas as an intermediate here is fine: the rows are already
        # in driver memory, and the Parquet write is a single-partition job.
        tmdbDf = spark.createDataFrame(pd.DataFrame(rows), schema=TMDB_PARQUET_SCHEMA)
        tmdbDf.write.mode("overwrite").parquet(str(outputPath))
        logger.info(f"[TMDB] Results written to {outputPath}.")


# ---------------------------------------------------------------------------
# GraphFeatureExtractor — Spark-side, no HTTP calls
# ---------------------------------------------------------------------------

class GraphFeatureExtractor:
    """
    Transforms bipartite crew-movie edges into tabular numerical features
    and joins them with the TMDB-enriched metadata into a single feature matrix.

    Attributes:
        spark (SparkSession): Active PySpark engine session.
    """

    def __init__(self):
        """
        Acquire or create a Spark session.

        Uses getOrCreate() so that the extractor shares an existing session
        when called from the same process as the ingestor, avoiding the
        overhead of spinning up a second JVM.
        """
        self.spark = SparkSession.builder \
            .appName(config.SPARK_APP_NAME) \
            .master(config.SPARK_MASTER) \
            .config("spark.executorEnv.TMDB_API_KEY", config.TMDB_API_KEY) \
            .config("spark.executorEnv.TMDB_READ_TOKEN", config.TMDB_READ_TOKEN) \
            .getOrCreate()

    # -----------------------------------------------------------------------
    # Graph feature methods
    # -----------------------------------------------------------------------

    def computeBipartiteFeatures(self, relationsDf: DataFrame, rolePrefix: str) -> DataFrame:
        """
        Derive degree-centrality features from a bipartite person-movie graph.

        Degree centrality is the number of distinct movies a person has been
        credited on — a proxy for industry experience. The per-person degree
        is aggregated back to the movie level:

            {prefix}_avg_centrality — average experience scale of the team.
            {prefix}_count          — raw size of the credited crew block.

        Args:
            relationsDf: DataFrame with columns [tconst, nconst].
            rolePrefix:  String prepended to output column names (e.g. "director").

        Returns:
            DataFrame keyed on tconst with three centrality feature columns.
        """
        # Step 1: compute each person's global degree across the whole dataset.
        personDegree = relationsDf \
            .groupBy("nconst") \
            .count() \
            .withColumnRenamed("count", f"{rolePrefix}_degree")

        # Step 2: join the global degree back onto the edge list so every
        # (tconst, nconst) row carries that person's experience count.
        relationsWithDegree = relationsDf.join(personDegree, on="nconst", how="left")

        # Step 3: aggregate to the movie level.
        # avg captures typical crew experience; max captures the most experienced
        # individual (e.g. a veteran director paired with a first-time crew).
        return relationsWithDegree \
            .groupBy("tconst") \
            .agg(
                F.avg(f"{rolePrefix}_degree").alias(f"{rolePrefix}_avg_centrality"),
                F.count("nconst").alias(f"{rolePrefix}_count"),
            )

    def computeSameRoleCollabWeight(self, relDf: DataFrame, rolePrefix: str) -> DataFrame:
        """
        Measure the strength of established partnerships within a single role.

        Two people are a pair if they share a tconst credit. Their collaboration
        weight is the number of times that specific pair has co-appeared across
        all titles. The maximum weight among all pairs on a given movie is used
        as the movie-level feature — capturing whether the film benefits from a
        well-established creative partnership.

        Complexity note:
            The self-join produces O(k²) pairs per movie, where k is the number
            of people credited in this role. For typical films (k ≤ 10) this is
            negligible. For anomalous records with very high k (anthology projects,
            data errors), the intermediate table grows quadratically. Consider a
            pre-filter on groupBy("tconst").count() if this is a concern.

        Args:
            relDf:      DataFrame with columns [tconst, nconst].
            rolePrefix: Used to name the output column (e.g. "writer" →
                        "writer_writer_max_collab_weight").

        Returns:
            DataFrame keyed on tconst with one collaboration weight column.
        """
        colName = f"{rolePrefix}_{rolePrefix}_max_collab_weight"

        # Alias both sides of the self-join; the < filter ensures each pair
        # is counted once rather than twice — (A, B) and (B, A) are the same
        # partnership within the same role.
        p1 = relDf.withColumnRenamed("nconst", "person_1")
        p2 = relDf.withColumnRenamed("nconst", "person_2")

        pairs = p1.join(p2, on="tconst", how="inner") \
                  .filter(F.col("person_1") < F.col("person_2"))

        # Global pair frequency across all titles.
        pairFreq = pairs \
            .groupBy("person_1", "person_2") \
            .count() \
            .withColumnRenamed("count", "collaboration_weight")

        moviesWithPairs = pairs.join(pairFreq, on=["person_1", "person_2"], how="left")

        return moviesWithPairs \
            .groupBy("tconst") \
            .agg(F.max("collaboration_weight").alias(colName))

    def computeCrossRoleCollabWeight(
        self,
        roleADf:     DataFrame,
        roleBDf:     DataFrame,
        roleAPrefix: str,
        roleBPrefix: str,
    ) -> DataFrame:
        """
        Measure the strength of established partnerships between two different roles.

        Structurally identical to computeSameRoleCollabWeight but operates on two
        heterogeneous edge sets (e.g. writers and directors) rather than a self-join
        on one. A pair here is one person from role A and one from role B who share
        a tconst credit.

        This generalises to any two role combinations without code duplication.
        Current usage: writer-director pairs, capturing whether a film uses a
        writer-director team with an established working relationship.

        No < filter is applied here because (writer_X, director_Y) and
        (writer_Y, director_X) are genuinely distinct pairs when the roles differ,
        unlike the same-role case where both orderings represent the same partnership.

        Args:
            roleADf:     DataFrame with columns [tconst, nconst] for role A.
            roleBDf:     DataFrame with columns [tconst, nconst] for role B.
            roleAPrefix: Name of role A (e.g. "writer").
            roleBPrefix: Name of role B (e.g. "director").

        Returns:
            DataFrame keyed on tconst with one collaboration weight column named
            {roleAPrefix}_{roleBPrefix}_max_collab_weight.
        """
        colName = f"{roleAPrefix}_{roleBPrefix}_max_collab_weight"

        # Rename nconst to role-specific names before the join to avoid
        # ambiguous column references in the pair frequency aggregation.
        a = roleADf.withColumnRenamed("nconst", roleAPrefix)
        b = roleBDf.withColumnRenamed("nconst", roleBPrefix)

        # Inner join on tconst: every (roleA_person, roleB_person) pair that
        # has shared a credit on any title.
        pairs = a.join(b, on="tconst", how="inner")

        # Global pair frequency across all titles.
        pairFreq = pairs \
            .groupBy(roleAPrefix, roleBPrefix) \
            .count() \
            .withColumnRenamed("count", "collaboration_weight")

        moviesWithPairs = pairs.join(
            pairFreq, on=[roleAPrefix, roleBPrefix], how="left"
        )

        return moviesWithPairs \
            .groupBy("tconst") \
            .agg(F.max("collaboration_weight").alias(colName))

    # -----------------------------------------------------------------------
    # Orchestration
    # -----------------------------------------------------------------------

    def run(self) -> None:
        """
        Execute the full graph feature extraction pipeline.

        Stage 1 — TMDB fetch (driver-side):
            TMDBFetcher reads the tconst list, calls the API, and writes a
            TMDB Parquet. This stage is skipped if the TMDB Parquet already
            exists, allowing reruns without re-fetching.

        Stage 2 — Graph feature computation (Spark):
            Degree centralities and collaborative weights are computed from
            the crew relation Parquets.

        Stage 3 — Join and write (Spark):
            All feature DataFrames are left-joined onto the movie base table
            and the result is written to featured_graph.parquet.

        Output schema (in addition to all columns from movies_cleaned.parquet):
            director_avg_centrality           — average director degree centrality
            director_count                    — size of director bloc
            writer_avg_centrality             — average writer degree centrality
            writer_count                      — size of writer bloc
            writer_writer_max_collab_weight   — max writer-writer pair frequency
            writer_director_max_collab_weight — max writer-director pair frequency
            tmdb_popularity                   — TMDB popularity score
            tmdb_vote_average                 — TMDB audience vote average
            tmdb_budget                       — production budget in USD
            tmdb_revenue                      — box office revenue in USD
            tmdb_primary_genre                — nominal primary catalog genre from TMDB
            tmdb_production_company           — first-listed production company
            tmdb_success                      — True if TMDB fetch succeeded
            tmdb_fetched_at                   — UTC timestamp of fetch attempt
        """
        moviesDf = self.spark.read.parquet(
            str(config.PARQUET_DIR / "movies_cleaned.parquet")
        )

        dirPath  = config.PARQUET_DIR / "directing.parquet"
        writPath = config.PARQUET_DIR / "writing.parquet"
        tmdbPath = config.PARQUET_DIR / "tmdb.parquet"

        # -------------------------------------------------------------------
        # Stage 1 — TMDB fetch (driver-side, no Spark jobs)
        # -------------------------------------------------------------------

        if tmdbPath.exists():
            # Allow reruns without re-fetching. Delete the file manually to
            # force a refresh (e.g. after the dataset grows or the API changes).
            logger.info(f"[TMDB] Parquet already exists at {tmdbPath}, skipping fetch.")
        else:
            logger.info("[TMDB] Collecting tconst list from movies Parquet for API fetch.")

            # Collect only the tconst column — no need to pull the full table
            # to the driver for what is just a list of IDs.
            tconsts = [
                row["tconst"]
                for row in moviesDf.select("tconst").collect()
                if row["tconst"] is not None
            ]

            fetcher = TMDBFetcher()
            fetcher.fetchAndSave(
                tconsts    = tconsts,
                outputPath = tmdbPath,
                spark      = self.spark,
            )

        tmdbDf = self.spark.read.parquet(str(tmdbPath))

        # -------------------------------------------------------------------
        # Stage 2 — Graph feature computation
        # -------------------------------------------------------------------

        finalDf     = moviesDf
        directorsDf = None
        writersDf   = None

        if dirPath.exists():
            logger.info("   -> [GRAPH]: Computing degree centralities for directors.")
            directorsDf = self.spark.read.parquet(str(dirPath))
            dirFeatures = self.computeBipartiteFeatures(directorsDf, "director")
            finalDf = finalDf.join(dirFeatures, on="tconst", how="left")

        if writPath.exists():
            logger.info("   -> [GRAPH]: Computing degree centralities for writers.")
            writersDf   = self.spark.read.parquet(str(writPath))
            writFeatures = self.computeBipartiteFeatures(writersDf, "writer")
            finalDf = finalDf.join(writFeatures, on="tconst", how="left")

            logger.info("   -> [SYNERGY]: Computing writer-writer collaboration weights.")
            wwCollab = self.computeSameRoleCollabWeight(writersDf, "writer")
            finalDf = finalDf.join(wwCollab, on="tconst", how="left")

        if writersDf is not None and directorsDf is not None:
            logger.info("   -> [SYNERGY]: Computing writer-director collaboration weights.")
            # Cross-role collaboration: one person from each edge set, joined on tconst.
            # Movies with a single writer or no director will receive null here —
            # this is correct, as no cross-role pair can be formed.
            wdCollab = self.computeCrossRoleCollabWeight(
                roleADf     = writersDf,
                roleBDf     = directorsDf,
                roleAPrefix = "writer",
                roleBPrefix = "director",
            )
            finalDf = finalDf.join(wdCollab, on="tconst", how="left")

        # -------------------------------------------------------------------
        # Stage 3 — TMDB join and write
        # -------------------------------------------------------------------

        logger.info("   -> [TMDB]: Joining TMDB enrichment onto feature matrix.")
        # Left join: all movies are preserved. tmdb_success=False rows represent
        # titles for which the API call failed or returned no match.
        finalDf = finalDf.join(tmdbDf, on="tconst", how="left")

        logger.info("   -> [TMDB]: Coalescing tmdb_runtime into runtimeMinutes.")
        finalDf = finalDf.withColumn(
            "runtimeMinutes",
            F.coalesce(F.col("runtimeMinutes"), F.col("tmdb_runtime"))
        ).drop("tmdb_runtime")

        outputPath = str(config.PARQUET_DIR / "featured_graph.parquet")
        logger.info(f"   -> [WRITE]: Writing featured graph matrix to {outputPath}.")
        finalDf.write.mode("overwrite").parquet(outputPath)
        logger.info("   -> [COMPLETE]: Featured graph matrix written successfully.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    extractor = GraphFeatureExtractor()
    extractor.run()