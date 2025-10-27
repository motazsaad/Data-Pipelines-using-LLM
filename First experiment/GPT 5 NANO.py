# Databricks Python ETL: Open-Meteo Air Quality -> Bronze/Silver Delta with quality checks
# Notes:
# - Uses serverless-compatible PySpark + Delta (append, partitioned by ingestion_date)
# - Extracts via requests, transforms into structured Spark rows, loads Bronze, runs quality checks,
#   and writes clean rows to Silver

# 1) Extract Data (API call)
import requests
import datetime
from functools import reduce

# Open-Meteo API (past 31 days + 1 forecast day)
LAT = 40.3548
LON = 18.1724
URL = (
    f"https://air-quality-api.open-meteo.com/v1/air-quality"
    f"?latitude={LAT}&longitude={LON}"
    f"&hourly=pm10,pm2_5,carbon_monoxide,carbon_dioxide,nitrogen_dioxide,sulphur_dioxide,ozone"
    f"&past_days=31&forecast_days=1"
)

print(f"Fetching data from API: {URL}")
resp = requests.get(URL, timeout=60)
resp.raise_for_status()  # fail fast if request fails
data = resp.json()

hourly = data.get("hourly", {})
times = hourly.get("time", [])
pollutant_keys = ["pm10", "pm2_5", "carbon_monoxide", "carbon_dioxide",
                  "nitrogen_dioxide", "sulphur_dioxide", "ozone"]

# Build structured rows (one row per hourly timestamp)
rows = []
n_times = len(times)
for i in range(n_times):
    row = {"timestamp": times[i]}
    for key in pollutant_keys:
        vals = hourly.get(key, [])
        row[key] = vals[i] if i < len(vals) else None
    # Ingestion date (date only)
    row["ingestion_date"] = datetime.date.today()
    rows.append(row)

# 2) Transform and Load (Bronze)
# Create DataFrame from Python rows
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType, DateType
from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp, col

# If you are in a Databricks notebook, `spark` is already available
# Create DataFrame with implicit schema (will infer types)
bronze_df = spark.createDataFrame(rows)

# Cast types to desired types
bronze_df = bronze_df \
    .withColumn("timestamp", to_timestamp(col("timestamp"))) \
    .withColumn("ingestion_date", col("ingestion_date").cast("date"))

for k in pollutant_keys:
    bronze_df = bronze_df.withColumn(k, col(k).cast("double"))

# Persist to Delta bronze table (partitioned by ingestion_date)
bronze_table = "air_quality_bronze"
bronze_df.write.format("delta").mode("append").partitionBy("ingestion_date").saveAsTable(bronze_table)

print(f"Bronze write complete: {bronze_df.count()} rows partitioned by ingestion_date into {bronze_table}.")

# 3) Data Quality Checks (on Bronze data just loaded)
# Define pollutant columns
pollutant_cols = pollutant_keys

# Null checks: per-column nulls and rows with any nulls
nulls_by_column = {}
for c in pollutant_cols:
    nulls_by_column[c] = bronze_df.filter(col(c).isNull()).count()

any_null_condition = reduce(lambda a, b: a | b, [F.col(c).isNull() for c in pollutant_cols])
rows_with_any_nulls = bronze_df.filter(any_null_condition).count()

# Duplicate checks: duplicates by timestamp (keep first occurrence)
from pyspark.sql.window import Window
dup_window = Window.partitionBy("timestamp").orderBy("ingestion_date")
bronze_with_rn = bronze_df.withColumn("rn", F.row_number().over(dup_window))
duplicate_count = bronze_with_rn.filter(F.col("rn") > 1).count()
bronze_dedup = bronze_with_rn.filter(F.col("rn") == 1).drop("rn")

# Build a quality report
quality_report = {
    "nulls_by_column": nulls_by_column,
    "rows_with_any_nulls": rows_with_any_nulls,
    "duplicate_timestamp_count": duplicate_count,
    "bronze_total_rows": bronze_df.count(),
    "bronze_dedup_rows": bronze_dedup.count()
}

# Print quality report
print("=== Quality Report (Bronze) ===")
print(f"Total Bronze rows loaded: {quality_report['bronze_total_rows']}")
print("Nulls by column:")
for c, v in nulls_by_column.items():
    print(f" - {c}: {v} nulls")
print(f"Rows with any pollutant nulls: {rows_with_any_nulls}")
print(f"Duplicate timestamps detected: {duplicate_count} (kept first occurrence)")
print(f"Bronze rows after dedup: {quality_report['bronze_dedup_rows']}")

# 4) Final Save (Silver) - keep only clean rows (no nulls in pollutants, after dedup)
# Define non-null condition for all pollutant columns
non_null_condition = None
for c in pollutant_cols:
    if non_null_condition is None:
        non_null_condition = F.col(c).isNotNull()
    else:
        non_null_condition = non_null_condition & F.col(c).isNotNull()

clean_rows = bronze_dedup.filter(non_null_condition)

# Persist clean data to Silver (Delta), append mode
silver_table = "air_quality_silver"
clean_rows.write.format("delta").mode("append").partitionBy("ingestion_date").saveAsTable(silver_table)

print(f"Silver write complete: {clean_rows.count()} clean rows saved to {silver_table} (partitioned by ingestion_date).")

# Optional: final summary
print("ETL pipeline finished.")
