# main.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window

# 1. Start SparkSession
spark = SparkSession.builder.appName("MusicAnalysis").getOrCreate()

# 2. Load datasets
logs = spark.read.option("header", True).csv("listening_logs.csv", inferSchema=True)
songs = spark.read.option("header", True).csv("songs_metadata.csv", inferSchema=True)

# Join logs with songs metadata
df = logs.join(songs, on="song_id", how="inner")

# =====================================================
# Task 1: User Favorite Genres
# =====================================================
genre_counts = df.groupBy("user_id", "genre").count()

# Window to rank genres by count per user
w = Window.partitionBy("user_id").orderBy(desc("count"))
user_fav_genre = genre_counts.withColumn("rank", row_number().over(w)) \
                             .filter(col("rank") == 1) \
                             .drop("rank")

user_fav_genre.write.mode("overwrite").csv("output/user_favorite_genres", header=True)

# =====================================================
# Task 2: Average Listen Time per Song
# =====================================================
avg_listen_time = df.groupBy("song_id", "title", "artist", "genre") \
                    .agg(avg("duration_sec").alias("avg_duration_sec"))

avg_listen_time.write.mode("overwrite").csv("output/avg_listen_time_per_song", header=True)

# =====================================================
# Task 3: Genre Loyalty Scores
# =====================================================
# Total plays per user
total_plays = df.groupBy("user_id").count().withColumnRenamed("count", "total_plays")

# Plays per user + genre
plays_per_genre = df.groupBy("user_id", "genre").count()

# Find favorite genre play count
fav_play_count = plays_per_genre.join(user_fav_genre, ["user_id", "genre"], "inner") \
                                .select("user_id", col("count").alias("fav_genre_plays"))

# Loyalty score = fav_genre_plays / total_plays
loyalty = fav_play_count.join(total_plays, "user_id") \
                        .withColumn("loyalty_score", col("fav_genre_plays") / col("total_plays")) \
                        .filter(col("loyalty_score") > 0.8)

loyalty.write.mode("overwrite").csv("output/genre_loyalty_scores", header=True)

# =====================================================
# Task 4: Identify users who listen between 12 AM and 5 AM
# =====================================================
# Extract hour from timestamp
df = df.withColumn("hour", hour(to_timestamp("timestamp", "yyyy-MM-dd HH:mm:ss")))

night_owls = df.filter((col("hour") >= 0) & (col("hour") <= 5)) \
               .select("user_id").distinct()

night_owls.write.mode("overwrite").csv("output/night_owl_users", header=True)

# Stop Spark
spark.stop()
