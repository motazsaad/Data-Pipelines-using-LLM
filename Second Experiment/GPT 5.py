# Databricks Python code - corrected monthly aggregation (no Window import needed)

import time
import requests
from pyspark.sql import functions as F

# ----------------------------
# 1) Extract Data
# ----------------------------
endpoint = (
    "https://air-quality-api.open-meteo.com/v1/air-quality"
    "?latitude=40.3548&longitude=18.1724"
    "&hourly=pm10,pm2_5,carbon_monoxide,carbon_dioxide,nitrogen_dioxide,sulphur_dioxide,ozone"
    "&start_date=2025-03-01&end_date=2025-08-31"
)

pollutants = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "carbon_dioxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
]

print("Starting data extraction from API...")
start_time = time.time()

try:
    resp = requests.get(endpoint, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    print("API request successful.")
except Exception as e:
    print(f"ERROR during API request: {e}")
    raise

fetch_time = time.time() - start_time
print(f"Data retrieval completed in {fetch_time:.2f} seconds.")

# ----------------------------
# 2) Transform Data
# ----------------------------
print("Transforming JSON into Spark DataFrame...")
trans_start = time.time()

hourly = data.get("hourly", {})
times = hourly.get("time", [])

# Build a list of dict records: one row per hour
records = []
for idx, ts in enumerate(times):
    row = {"timestamp": ts}
    for p in pollutants:
        arr = hourly.get(p, [])
        row[p] = arr[idx] if idx < len(arr) else None
    records.append(row)

# Create Spark DataFrame
df = spark.createDataFrame(records)

# Convert timestamp string to actual Timestamp type
df = df.withColumn("timestamp", F.to_timestamp(F.col("timestamp"), "yyyy-MM-dd'T'HH:mm"))

# Extract year and month for grouping
df = (
    df
    .withColumn("year", F.year(F.col("timestamp")))
    .withColumn("month", F.month(F.col("timestamp")))
    .withColumn("ingestion_date", F.current_date())
)

row_count = df.count()
print(f"Transformed DataFrame has {row_count} rows and {len(df.columns)} columns.")
trans_time = time.time() - trans_start
print(f"Transformation completed in {trans_time:.2f} seconds.")

# ----------------------------
# 3) Monthly Aggregation
# ----------------------------
print("Computing monthly averages for all pollutants (aligned to existing Delta table schema)...")
agg_start = time.time()

# Averages for each pollutant, named to match the target 'avg_<pollutant>' columns
agg_exprs = [
    F.avg("pm10").alias("avg_pm10"),
    F.avg("pm2_5").alias("avg_pm2_5"),
    F.avg("carbon_monoxide").alias("avg_carbon_monoxide"),
    F.avg("carbon_dioxide").alias("avg_carbon_dioxide"),
    F.avg("nitrogen_dioxide").alias("avg_nitrogen_dioxide"),
    F.avg("sulphur_dioxide").alias("avg_sulphur_dioxide"),
    F.avg("ozone").alias("avg_ozone"),
]

monthly = (
    df
    .groupBy("year", "month")
    .agg(*agg_exprs, F.count("*").alias("total_records"))
    .withColumn("processing_date", F.current_timestamp())
    .orderBy("year", "month")
)

agg_rows = monthly.count()
print(f"Monthly aggregated rows: {agg_rows}")
agg_time = time.time() - agg_start
print(f"Aggregation completed in {agg_time:.2f} seconds.")

# ----------------------------
# 4) Save Results
# ----------------------------
print("Saving results to Delta table air_quality_monthly_avg (append with schema merge)...")
write_start = time.time()

try:
    monthly.write.format("delta").mode("append") \
        .option("mergeSchema", "true") \
        .saveAsTable("air_quality_monthly_avg")
    print("Data successfully written to Delta table air_quality_monthly_avg.")
except Exception as e:
    print(f"ERROR while writing to Delta (mergeSchema): {e}")
    print("Attempting fallback: write to a separate table to avoid blocking the pipeline.")
    try:
        monthly.write.format("delta").mode("append").saveAsTable("air_quality_monthly_avg_v2")
        print("Data written to Delta table air_quality_monthly_avg_v2 as a fallback.")
    except Exception as e2:
        print(f"Fallback write failed as well: {e2}")
        raise

write_time = time.time() - write_start

total_time = time.time() - start_time
print(f"Total pipeline time: {total_time:.2f} seconds.")

# ----------------------------
# 5) Results Visualization
# ----------------------------
print("\nSample of monthly averages:")
monthly.show(5, truncate=False)

print("\nSummary statistics for monthly averages:")
monthly.describe().show(truncate=False)

# End of script
