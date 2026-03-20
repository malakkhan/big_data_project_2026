"""
Graph & Network Topological Engineering.

Calculates degree centralities, collaborative frequencies (team synergies),
and bridges independent domain APIs representing external context parameters.

Classes:
    GraphFeatureExtractor: Distills relational databases mapping edges into nodes.
"""

import logging
import sys
import requests
import json
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, DoubleType, StructType, StructField, StringType
from pathlib import Path

from src import config

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

class GraphFeatureExtractor:
    """
    Transforms bipartite edges (e.g., Movie <--> Director) into tabular numerical metrics.

    Attributes:
        spark (SparkSession): Unified execution mapping over the JVM backend.
    """

    def __init__(self):
        """
        Instantiates JVM mappings necessary for PySpark DAG configurations.
        """
        self.spark = SparkSession.builder \
            .appName(config.SPARK_APP_NAME) \
            .master(config.SPARK_MASTER) \
            .getOrCreate()
            
    def compute_bipartite_features(self, relations_df, role_prefix):
        """
        Computes analytical centralities across distinct crew relationships.

        Translates independent film connections into degree measurements indicating 
        individual industry prominence, subsequently reducing back to the context film.

        Args:
            relations_df (pyspark.sql.DataFrame): PySpark relation mapping identifying edges.
            role_prefix (str): Textual identifier resolving conflicts (e.g., "director").

        Returns:
            pyspark.sql.DataFrame: Summarized node representations mapped strictly to 'tconst'.
        """
        person_degree = relations_df.groupBy("nconst").count() \
                                    .withColumnRenamed("count", f"{role_prefix}_degree")
                                    
        relations_with_degree = relations_df.join(person_degree, on="nconst", how="left")
        
        movie_crew_experience = relations_with_degree.groupBy("tconst") \
            .agg(
                F.avg(f"{role_prefix}_degree").alias(f"{role_prefix}_avg_centrality"),
                F.max(f"{role_prefix}_degree").alias(f"{role_prefix}_max_centrality"),
                F.count("nconst").alias(f"{role_prefix}_count")
            )
            
        return movie_crew_experience

    def compute_collaborative_weight(self, writers_df):
        """
        Measures the synergy between unique writing pairings sharing the same title matrix.

        Args:
            writers_df (pyspark.sql.DataFrame): Standard execution edges linking movies to staff.

        Returns:
            pyspark.sql.DataFrame: Computed feature indicating maximal collaborative frequency.
        """
        w1 = writers_df.withColumnRenamed("nconst", "writer_1")
        w2 = writers_df.withColumnRenamed("nconst", "writer_2")
        
        pairs = w1.join(w2, "tconst", "inner") \
                  .filter(F.col("writer_1") < F.col("writer_2"))
                  
        pair_freq = pairs.groupBy("writer_1", "writer_2").count() \
                         .withColumnRenamed("count", "collaboration_weight")
                         
        movies_with_pairs = pairs.join(pair_freq, ["writer_1", "writer_2"], "left")
        
        movie_collab = movies_with_pairs.groupBy("tconst") \
                                        .agg(F.max("collaboration_weight").alias("max_team_collaboration_weight"))
                                        
        return movie_collab
        
tmdb_schema = StructType([
    StructField("tmdb_popularity", DoubleType(), True),
    StructField("tmdb_vote_average", DoubleType(), True),
    StructField("tmdb_budget", DoubleType(), True),
    StructField("tmdb_revenue", DoubleType(), True),
    StructField("tmdb_runtime", IntegerType(), True),
    StructField("tmdb_primary_genre", StringType(), True),
    StructField("tmdb_production_company", StringType(), True)
])

@F.udf(returnType=tmdb_schema)
def fetch_tmdb_api(tconst):
    """
    Triggers deterministic two-hop REST API payloads.
    
    1. Fetches TMDB internal ID using the IMDB tconst.
    2. Queries the /movie/ endpoint fetching complex sub-graphs (budgets, crews).
    """
    import requests
    import time
    if not tconst:
        return (None, None, None, None, None, None, None)
        
    url_find = f"https://api.themoviedb.org/3/find/{tconst}?external_source=imdb_id"
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {config.TMDB_READ_TOKEN}"
    }
    
    try:
        res = requests.get(url_find, headers=headers, timeout=5)
        if res.status_code == 200:
            data = res.json()
            results = data.get("movie_results", [])
            if results:
                tmdb_id = results[0].get("id")
                
                # Double hop to detailed endpoint
                url_movie = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
                res_movie = requests.get(url_movie, headers=headers, timeout=5)
                
                if res_movie.status_code == 200:
                    m = res_movie.json()
                    pop = float(m.get("popularity", 0.0))
                    vote = float(m.get("vote_average", 0.0))
                    budget = float(m.get("budget", 0))
                    revenue = float(m.get("revenue", 0))
                    
                    runtime = m.get("runtime", None)
                    runtime = int(runtime) if runtime is not None else None
                    
                    genres = m.get("genres", [])
                    genre = genres[0]["name"] if genres else None
                    
                    pcs = m.get("production_companies", [])
                    company = pcs[0]["name"] if pcs else None
                    
                    return (pop, vote, budget, revenue, runtime, genre, company)
    except Exception:
        pass
        
    return (None, None, None, None, None, None, None)

class GraphFeatureExtractor:
    """
    Transforms bipartite edges (e.g., Movie <--> Director) into tabular numerical metrics.

    Attributes:
        spark (SparkSession): Unified execution mapping over the JVM backend.
    """

    def __init__(self):
        """
        Instantiates JVM mappings necessary for PySpark DAG configurations.
        """
        self.spark = SparkSession.builder \
            .appName(config.SPARK_APP_NAME) \
            .master(config.SPARK_MASTER) \
            .getOrCreate()
            
    def compute_bipartite_features(self, relations_df, role_prefix):
        """
        Computes analytical centralities across distinct crew relationships.

        Translates independent film connections into degree measurements indicating 
        individual industry prominence, subsequently reducing back to the context film.

        Args:
            relations_df (pyspark.sql.DataFrame): PySpark relation mapping identifying edges.
            role_prefix (str): Textual identifier resolving conflicts (e.g., "director").

        Returns:
            pyspark.sql.DataFrame: Summarized node representations mapped strictly to 'tconst'.
        """
        person_degree = relations_df.groupBy("nconst").count() \
                                    .withColumnRenamed("count", f"{role_prefix}_degree")
                                    
        relations_with_degree = relations_df.join(person_degree, on="nconst", how="left")
        
        movie_crew_experience = relations_with_degree.groupBy("tconst") \
            .agg(
                F.avg(f"{role_prefix}_degree").alias(f"{role_prefix}_avg_centrality"),
                F.max(f"{role_prefix}_degree").alias(f"{role_prefix}_max_centrality"),
                F.count("nconst").alias(f"{role_prefix}_count")
            )
            
        return movie_crew_experience

    def compute_collaborative_weight(self, writers_df):
        """
        Measures the synergy between unique writing pairings sharing the same title matrix.

        Args:
            writers_df (pyspark.sql.DataFrame): Standard execution edges linking movies to staff.

        Returns:
            pyspark.sql.DataFrame: Computed feature indicating maximal collaborative frequency.
        """
        w1 = writers_df.withColumnRenamed("nconst", "writer_1")
        w2 = writers_df.withColumnRenamed("nconst", "writer_2")
        
        pairs = w1.join(w2, "tconst", "inner") \
                  .filter(F.col("writer_1") < F.col("writer_2"))
                  
        pair_freq = pairs.groupBy("writer_1", "writer_2").count() \
                         .withColumnRenamed("count", "collaboration_weight")
                         
        movies_with_pairs = pairs.join(pair_freq, ["writer_1", "writer_2"], "left")
        
        movie_collab = movies_with_pairs.groupBy("tconst") \
                                        .agg(F.max("collaboration_weight").alias("max_team_collaboration_weight"))
                                        
        return movie_collab

    def run(self):
        """
        Drives calculation sequences persisting network characteristics efficiently.

        Args:
            None

        Returns:
            None
        """
        movies_df = self.spark.read.parquet(str(config.OUTPUT_DIR / "parquet" / "movies_cleaned.parquet"))
        
        dir_path = config.OUTPUT_DIR / "parquet" / "directing.parquet"
        writ_path = config.OUTPUT_DIR / "parquet" / "writing.parquet"
        
        final_df = movies_df
        
        if dir_path.exists():
            directors_df = self.spark.read.parquet(str(dir_path))
            dir_features = self.compute_bipartite_features(directors_df, "director")
            final_df = final_df.join(dir_features, on="tconst", how="left")
            
        if writ_path.exists():
            writers_df = self.spark.read.parquet(str(writ_path))
            writ_features = self.compute_bipartite_features(writers_df, "writer")
            collab_features = self.compute_collaborative_weight(writers_df)
            
            final_df = final_df.join(writ_features, on="tconst", how="left") \
                               .join(collab_features, on="tconst", how="left")
                               
        # Apply massive external structural fetch. 
        # WARNING: In massive production tables, limit API mappings or pre-cache.
        final_df = final_df.withColumn("tmdb_struct", fetch_tmdb_api(F.col("tconst")))
        
        final_df = final_df.select(
            "*",
            "tmdb_struct.tmdb_popularity",
            "tmdb_struct.tmdb_vote_average",
            "tmdb_struct.tmdb_budget",
            "tmdb_struct.tmdb_revenue",
            "tmdb_struct.tmdb_runtime",
            "tmdb_struct.tmdb_primary_genre",
            "tmdb_struct.tmdb_production_company"
        ).drop("tmdb_struct")
        
        output_path = str(config.OUTPUT_DIR / "parquet" / "featured_graph.parquet")
        final_df.write.mode("overwrite").parquet(output_path)

if __name__ == "__main__":
    extractor = GraphFeatureExtractor()
    extractor.run()
