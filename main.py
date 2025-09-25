from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window

spark = SparkSession.builder.appName("MusicAnalysis").getOrCreate()

# -------------------------------
# Load datasets
# -------------------------------
logs = spark.read.csv("listening_logs.csv", header=True, inferSchema=True)
songs = spark.read.csv("songs_metadata.csv", header=True, inferSchema=True)

# Join logs with songs to access genre info
df = logs.join(songs, on="song_id", how="inner")

# -------------------------------
# Task 1: User Favorite Genres
# -------------------------------
user_genre_counts = df.groupBy("user_id", "genre").count()

# Find max genre count per user
window_spec = Window.partitionBy("user_id").orderBy(desc("count"))
user_fav_genres = user_genre_counts.withColumn(
    "rank", row_number().over(window_spec)
).filter(col("rank") == 1).drop("rank")

user_fav_genres.write.mode("overwrite").csv("output/user_favorite_genres/")

# -------------------------------
# Task 2: Average Listen Time per Song
# -------------------------------
avg_listen_time = logs.groupBy("song_id") \
    .agg(avg("duration_sec").alias("avg_duration_sec"))

avg_listen_time.write.mode("overwrite").csv("output/avg_listen_time_per_song/")

# -------------------------------
# Task 3: Genre Loyalty Scores
# -------------------------------
# Total plays per user
user_total_plays = df.groupBy("user_id").count().withColumnRenamed("count", "total_plays")

# Most listened genre per user
user_genre_max = user_genre_counts.join(
    user_fav_genres.select("user_id", "genre"), on=["user_id", "genre"], how="inner"
)

# Loyalty score = max_genre_count / total_plays
loyalty = user_genre_max.join(user_total_plays, on="user_id") \
    .withColumn("loyalty_score", col("count") / col("total_plays")) \
    .filter(col("loyalty_score") <= 0.75)

loyalty.write.mode("overwrite").csv("output/genre_loyalty_scores/")

# -------------------------------
# Task 4: Night Owl Users (12 AM – 5 AM)
# -------------------------------
night_owl_users = logs.withColumn("hour", hour(to_timestamp("timestamp"))) \
    .filter((col("hour") >= 0) & (col("hour") < 5)) \
    .select("user_id").distinct()

night_owl_users.write.mode("overwrite").csv("output/night_owl_users/")
