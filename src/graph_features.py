"""
Graph & Network Topological Feature Engineering.

Computes bipartite degree centralities and collaboration weights from
crew-movie edge relations, producing numerical features that capture
team experience and established creative partnerships.

Feature families produced:
    Degree centrality          -- experience score per person, aggregated per movie.
    Writer-writer synergy      -- co-credit frequency of writer pairs.
    Writer-director synergy    -- co-credit frequency of writer/director pairs.

Classes:
    GraphFeatureExtractor: Spark-side graph feature computation.
"""

import logging
import shutil
import sys
from pathlib import Path

import pyspark.sql.functions as F
from pyspark.sql import SparkSession, DataFrame

from src import config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GraphFeatureExtractor
# ---------------------------------------------------------------------------

class GraphFeatureExtractor:
    """
    Transforms bipartite crew-movie edges into tabular numerical features.

    Reads crew relation Parquets (directing.parquet, writing.parquet) and
    the TMDB-enriched movie Parquet, then computes graph-derived features
    and joins everything into a single feature matrix keyed on tconst.

    Attributes:
        spark (SparkSession): Active PySpark engine session.
    """

    def __init__(self, spark: SparkSession = None):
        self.spark = spark or SparkSession.builder \
            .appName(config.SPARK_APP_NAME) \
            .master(config.SPARK_MASTER) \
            .getOrCreate()

    # -----------------------------------------------------------------------
    # Graph feature methods
    # -----------------------------------------------------------------------

    def computeBipartiteFeatures(self, relationsDf: DataFrame, rolePrefix: str) -> DataFrame:
        """
        Derive degree-centrality features from a bipartite person-movie graph.

        Args:
            relationsDf: DataFrame with columns [tconst, nconst].
            rolePrefix:  String prepended to output column names.

        Returns:
            DataFrame keyed on tconst with centrality feature columns.
        """
        personDegree = relationsDf \
            .groupBy("nconst") \
            .count() \
            .withColumnRenamed("count", f"{rolePrefix}_degree")

        relationsWithDegree = relationsDf.join(personDegree, on="nconst", how="left")

        return relationsWithDegree \
            .groupBy("tconst") \
            .agg(
                F.avg(f"{rolePrefix}_degree").alias(f"{rolePrefix}_avg_centrality"),
                F.count("nconst").alias(f"{rolePrefix}_count"),
            )

    def computeSameRoleCollabWeight(self, relDf: DataFrame, rolePrefix: str) -> DataFrame:
        """
        Measure partnership strength within a single role via co-credit frequency.

        Args:
            relDf:      DataFrame with columns [tconst, nconst].
            rolePrefix: Used to name the output column.

        Returns:
            DataFrame keyed on tconst with one collaboration weight column.
        """
        colName = f"{rolePrefix}_{rolePrefix}_max_collab_weight"

        # Repartition by tconst to co-locate crew for the same movie,
        # reducing shuffle volume for the self-join.
        relDf = relDf.repartition("tconst")

        p1 = relDf.withColumnRenamed("nconst", "person_1")
        p2 = relDf.withColumnRenamed("nconst", "person_2")

        pairs = p1.join(p2, on="tconst", how="inner") \
                  .filter(F.col("person_1") < F.col("person_2"))

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
        Measure partnership strength between two different roles.

        Args:
            roleADf:     DataFrame with [tconst, nconst] for role A.
            roleBDf:     DataFrame with [tconst, nconst] for role B.
            roleAPrefix: Name of role A.
            roleBPrefix: Name of role B.

        Returns:
            DataFrame keyed on tconst with one collaboration weight column.
        """
        colName = f"{roleAPrefix}_{roleBPrefix}_max_collab_weight"

        a = roleADf.withColumnRenamed("nconst", roleAPrefix)
        b = roleBDf.withColumnRenamed("nconst", roleBPrefix)

        pairs = a.join(b, on="tconst", how="inner")

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

    def run(self, input_parquet: str = "tmdb_enriched.parquet") -> None:
        """
        Compute graph features and join them onto the enriched movie DataFrame.

        Reads the TMDB-enriched parquet (from tmdb_enrichment.py), computes
        bipartite centralities and collaboration weights from crew relations,
        and writes the final enriched_features.parquet.

        Args:
            input_parquet: Name of the input parquet file in PARQUET_DIR.
        """
        moviesDf = self.spark.read.parquet(
            str(config.PARQUET_DIR / input_parquet)
        )

        dirPath  = config.PARQUET_DIR / "directing.parquet"
        writPath = config.PARQUET_DIR / "writing.parquet"

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
            writersDf = self.spark.read.parquet(str(writPath))
            writFeatures = self.computeBipartiteFeatures(writersDf, "writer")
            finalDf = finalDf.join(writFeatures, on="tconst", how="left")

            logger.info("   -> [SYNERGY]: Computing writer-writer collaboration weights.")
            wwCollab = self.computeSameRoleCollabWeight(writersDf, "writer")
            finalDf = finalDf.join(wwCollab, on="tconst", how="left")

        if writersDf is not None and directorsDf is not None:
            logger.info("   -> [SYNERGY]: Computing writer-director collaboration weights.")
            wdCollab = self.computeCrossRoleCollabWeight(
                roleADf     = writersDf,
                roleBDf     = directorsDf,
                roleAPrefix = "writer",
                roleBPrefix = "director",
            )
            finalDf = finalDf.join(wdCollab, on="tconst", how="left")

        outputPath = config.PARQUET_DIR / "enriched_features.parquet"
        logger.info(f"   -> [WRITE]: Writing enriched feature matrix to {outputPath}.")
        # Remove any stale Parquet directory at this path first.
        if outputPath.is_dir():
            shutil.rmtree(outputPath)
        # Write via Pandas at the Spark→Pandas boundary so downstream
        # consumers (modeling.py, analyze_covariance.py) can read with
        # pd.read_parquet() without PyArrow version incompatibilities.
        finalDf.toPandas().to_parquet(str(outputPath), index=False)
        logger.info("   -> [COMPLETE]: Enriched feature matrix written successfully.")


if __name__ == "__main__":
    extractor = GraphFeatureExtractor()
    extractor.run()