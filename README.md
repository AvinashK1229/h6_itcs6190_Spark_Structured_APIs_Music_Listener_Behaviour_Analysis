# Music Streaming Analysis Using Spark Structured APIs

## Overview
This project analyzes user listening behavior and music trends using *Apache Spark Structured APIs* as part of Cloud Computing for Data Analysis (ITCS 6190/8190, Fall 2025).  
It demonstrates joins, aggregations, window functions, and time-based filtering.

*Highlights*
- Find each user’s favorite genre (by total listening duration; ties resolved by play count, then genre).
- Compute average/median listen time per user and per genre (plus a global summary).
- Calculate a *genre loyalty score* for each user.
- Identify *night-owl* users (listening between *12 AM and 5 AM*).

---

## Dataset Description

### listening_logs.csv
- user_id – Unique user ID  
- song_id – Unique song ID  
- timestamp – Play time (e.g., 2025-03-23 14:05:00)  
- duration_sec – Listening duration in seconds

### songs_metadata.csv
- song_id – Unique song ID  
- title – Song title  
- artist – Artist name  
- genre – Genre (e.g., Pop, Rock, Jazz, …)  
- mood – Mood (e.g., Happy, Sad, Energetic, Chill)

> Ensure at least 100 total records across the logs for meaningful results.

---

## Repository Structure
~~~
.
├── main.py                  # Spark analysis script (uses DataFrame API + windows)
├── listening_logs.csv       # Input: user listening logs
├── songs_metadata.csv       # Input: songs metadata
├── out/                     # Output directory (Spark writes Parquet part files)
│   ├── favorite_genre_per_user/
│   ├── avg_listen_per_user/
│   ├── avg_listen_per_genre/
│   ├── genre_loyalty_scores/
│   └── night_owl_users/
└── README.md
~~~

---

## Output Directory Structure
Spark writes each result as a folder containing one or more *Parquet* part files:
~~~
out/
├── favorite_genre_per_user/      # Task 1 results
├── avg_listen_per_user/          # Task 2a results (per user)
├── avg_listen_per_genre/         # Task 2b results (per genre)
├── genre_loyalty_scores/         # Task 3 results
└── night_owl_users/              # Task 4 results
~~~
> Note: Global averages/medians are printed to console in this implementation (not written to disk).

---

## Tasks and Outputs

### Task 1: User Favorite Genres
*Goal:* For each user, choose the genre with the highest total listening duration (tie-breakers: higher play count, then alphabetical genre).  
*Output:* out/favorite_genre_per_user/  
*Key fields:* user_id, favorite_genre, total_duration_sec, plays

### Task 2: Average Listen Time
*2a. Per User* → out/avg_listen_per_user/  
*2b. Per Genre* → out/avg_listen_per_genre/  
*2c. Global Summary* → printed to console  
*Key fields:* avg_duration_sec, median_duration_sec, plays

### Task 3: Genre Loyalty Scores
*Goal:* For each user, compute  
loyalty = (duration in favorite genre) / (total duration by user)  
*Output:* out/genre_loyalty_scores/  
*Key fields:* user_id, favorite_genre, user_total_duration_sec, favorite_genre_duration_sec, genre_loyalty_score

### Task 4: Night-Owl Users (12 AM–5 AM)
*Goal:* Identify users with plays in hours 0..4 based on parsed timestamps.  
*Output:* out/night_owl_users/  
*Key fields:* user_id, night_plays, night_duration_sec

---

## Execution Instructions
## Prerequisites

Before starting the assignment, ensure you have the following software installed and properly configured on your machine:

1. Python 3.x  
   - Download: https://www.python.org/downloads/  
   - Verify:
     ~~~bash
     python3 --version
     ~~~

2. PySpark  
   - Install:
     ~~~bash
     pip install pyspark
     ~~~

3. Apache Spark  
   - Download: https://spark.apache.org/downloads.html  
   - Verify:
     ~~~bash
     spark-submit --version
     ~~~

> If you’re in Codespaces or a preconfigured environment, PySpark/Spark may already be available.

### 2. Running the Analysis Tasks

####  Running Locally
1. *Ensure inputs are present* (listening_logs.csv, songs_metadata.csv) in the project root.  
2. *Run the analysis*:
   ~~~bash
   spark-submit main.py
   ~~~
3. *Verify outputs*:
   ~~~bash
   ls -R out/
   ~~~

---

## Errors and Resolutions

- *Multiple part files in output*  
  Spark runs in parallel and writes results as multiple part-*.parquet files. This is expected.

- *“nothing to commit, working tree clean” (Git)*  
  Your changes are already committed; push them:  
  ~~~bash
  git push -u origin main
  ~~~

- *403 while pushing to GitHub (HTTPS)*  
  Re-auth with GitHub CLI or use SSH. For SSH:  
  ~~~bash
  ssh-keygen -t ed25519 -C "you@example.com"
  eval "$(ssh-agent -s)"
  ssh-add ~/.ssh/id_ed25519
  gh ssh-key add ~/.ssh/id_ed25519.pub -t "codespaces-$(date +%F)"
  git remote set-url origin git@github.com:<your-username>/<your-repo>.git
  git push -u origin main
  ~~~

---

### Notes
- This project uses Spark SQL/DataFrame APIs, window functions (row_number over user_id), and time functions (to_timestamp, hour).
- Outputs are stored under out/ to keep the repository organized and aligned with assignment requirements.
