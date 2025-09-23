from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# --------------------------------------------------
# 1. Spark session
# --------------------------------------------------
spark = SparkSession.builder.appName("MusicInsights").getOrCreate()

# --------------------------------------------------
# 2. Load data
# --------------------------------------------------
logs = spark.read.csv("listening_logs.csv", header=True, inferSchema=True)
songs = spark.read.csv("songs_metadata.csv", header=True, inferSchema=True)

# Ensure timestamp is a timestamp type
logs = logs.withColumn("timestamp", F.to_timestamp("timestamp"))

# --------------------------------------------------
# 3. Join logs with song metadata to get genres
# --------------------------------------------------
df = logs.join(songs, on="song_id", how="inner")

# --------------------------------------------------
# 4. Genre Loyalty Scores
#    Percentage of a user’s listens for each genre
# --------------------------------------------------
total_by_user = df.groupBy("user_id") \
                  .agg(F.count("*").alias("total_listens"))

genre_by_user = df.groupBy("user_id", "genre") \
                  .agg(F.count("*").alias("genre_listens"))

genre_loyalty_scores = genre_by_user.join(total_by_user, "user_id") \
    .withColumn(
        "loyalty_score",
        (F.col("genre_listens") / F.col("total_listens")).cast("double")
    )

# Save as a Parquet directory
genre_loyalty_scores.write.mode("overwrite").parquet("genre_loyalty_scores/")

# --------------------------------------------------
# 5. Night Owl Users
#    Active between 10 PM and 4 AM
# --------------------------------------------------
night_owl_users = (
    df.withColumn("hour", F.hour("timestamp"))
      .filter((F.col("hour") >= 22) | (F.col("hour") < 4))
      .groupBy("user_id")
      .agg(F.count("*").alias("night_play_count"))
)

night_owl_users.write.mode("overwrite").parquet("night_owl_users/")

# Optional: show quick preview
print("=== Genre Loyalty Scores ===")
genre_loyalty_scores.show(5, truncate=False)

print("=== Night Owl Users ===")
night_owl_users.show(5, truncate=False)
