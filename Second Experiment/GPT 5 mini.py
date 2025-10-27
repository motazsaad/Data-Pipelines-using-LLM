# Databricks notebook - Python

import requests
import json
import time
import logging
from datetime import datetime
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp, year, month, current_timestamp, col, avg
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType, IntegerType

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("air_quality_ingest")

spark = SparkSession.builder.getOrCreate()

# ---- Configuration ----
API_URL = ("https://air-quality-api.open-meteo.com/v1/air-quality"
           "?latitude=40.3548&longitude=18.1724"
           "&hourly=pm10,pm2_5,carbon_monoxide,carbon_dioxide,nitrogen_dioxide,sulphur_dioxide,ozone"
           "&start_date=2025-03-01&end_date=2025-08-31")

DELTA_TABLE_NAME = "air_quality_monthly_avg"

# Pollutant keys expected in hourly block
POLLUTANTS = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "carbon_dioxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone"
]

# ---- Utility timing helper ----
def now_ts():
    return datetime.utcnow().isoformat() + "Z"

# ---- 1) Extract Data ----
t0 = time.perf_counter()
logger.info("Starting extraction from API at %s", now_ts())

try:
    resp = requests.get(API_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    logger.info("API request successful. HTTP status: %s", resp.status_code)
except requests.RequestException as e:
    logger.exception("Request to air-quality API failed: %s", e)
    raise

t1 = time.perf_counter()
logger.info("Extraction completed in %.2f seconds", t1 - t0)

# ---- Validate and find hourly data ----
if "hourly" not in data:
    raise ValueError("API response does not contain 'hourly' field. Response keys: %s" % list(data.keys()))

hourly = data["hourly"]

# Check required keys present
expected_keys = set(POLLUTANTS + ["time"])
missing = [k for k in (["time"] + POLLUTANTS) if k not in hourly]
if missing:
    raise ValueError(f"Missing expected keys in hourly payload: {missing}")

times = hourly["time"]
n = len(times)
logger.info("Number of hourly timestamps received: %d", n)

# Ensure pollutant arrays match length (if not, we'll handle gracefully)
for p in POLLUTANTS:
    if not isinstance(hourly.get(p), list):
        raise ValueError(f"Hourly pollutant '{p}' is not a list in response.")
    if len(hourly[p]) != n:
        # Option: align to shortest length
        logger.warning("Length mismatch for '%s' (%d) vs time (%d). Aligning to min length.", p, len(hourly[p]), n)

min_len = min([len(hourly[k]) for k in (["time"] + POLLUTANTS)])
if min_len != n:
    logger.info("Truncating all arrays to min length %d to ensure row alignment.", min_len)
    times = times[:min_len]

# ---- 2) Transform Data: build rows and create PySpark DataFrame ----
t2 = time.perf_counter()
logger.info("Starting transformation/parsing at %s", now_ts())

rows = []
for i in range(min_len):
    row = {"time": times[i]}
    for p in POLLUTANTS:
        # Some values may be null (None in JSON)
        val = hourly[p][i]
        # Keep None as-is; pandas will hold NaN for float values
        row[p] = val
    rows.append(row)

# Create a pandas DataFrame first (small to medium-sized payload)
pdf = pd.DataFrame(rows)
logger.info("Created pandas DataFrame with shape %s", pdf.shape)

# Convert empty strings to NaN if any (defensive)
pdf.replace("", pd.NA, inplace=True)

# Create Spark DataFrame: specify schema to get consistent dtypes
schema = StructType([
    StructField("time", StringType(), True),
    StructField("pm10", DoubleType(), True),
    StructField("pm2_5", DoubleType(), True),
    StructField("carbon_monoxide", DoubleType(), True),
    StructField("carbon_dioxide", DoubleType(), True),
    StructField("nitrogen_dioxide", DoubleType(), True),
    StructField("sulphur_dioxide", DoubleType(), True),
    StructField("ozone", DoubleType(), True),
])

sdf = spark.createDataFrame(pdf, schema=schema)

# Convert time string to timestamp type (assuming ISO format)
sdf = sdf.withColumn("timestamp", to_timestamp(col("time")))
# Extract year and month
sdf = sdf.withColumn("year", year(col("timestamp"))).withColumn("month", month(col("timestamp")))
# Add ingestion_date column
sdf = sdf.withColumn("ingestion_date", current_timestamp())

# Reorder / select useful columns
select_cols = ["timestamp", "year", "month", "ingestion_date"] + POLLUTANTS
sdf = sdf.select(*select_cols)

t3 = time.perf_counter()
logger.info("Transformation completed in %.2f seconds", t3 - t2)
logger.info("Spark DataFrame schema:")
sdf.printSchema()
logger.info("Total rows in Spark DataFrame: %d", sdf.count())

# ---- 3) Monthly Aggregation ----
t4 = time.perf_counter()
logger.info("Starting monthly aggregation at %s", now_ts())

# Build aggregation expression - average for each pollutant.
# Spark's avg ignores nulls by default, so nulls are handled automatically.
agg_exprs = [avg(col(p)).alias(f"avg_{p}") for p in POLLUTANTS]

monthly = sdf.groupBy("year", "month").agg(*agg_exprs).orderBy("year", "month")

t5 = time.perf_counter()
logger.info("Aggregation completed in %.2f seconds", t5 - t4)
logger.info("Number of monthly groups: %d", monthly.count())

# Add metadata columns (ingestion_date = now) to the monthly summary
monthly = monthly.withColumn("ingestion_date", current_timestamp())

# ---- 4) Save Results to Delta table ----
t6 = time.perf_counter()
logger.info("Saving monthly aggregated results to Delta table '%s' (append mode).", DELTA_TABLE_NAME)

# Optionally enable autoMerge (if schema evolution required) - uncomment if needed
# spark.conf.set("spark.databricks.delta.schema.autoMerge.enabled", "true")

try:
    monthly.write.format("delta").mode("append").saveAsTable(DELTA_TABLE_NAME)
    logger.info("Write to Delta table completed.")
except Exception as e:
    logger.exception("Failed to write to Delta table: %s", e)
    raise

t7 = time.perf_counter()
logger.info("Save completed in %.2f seconds", t7 - t6)

# ---- Display sample results and summary stats ----
logger.info("Displaying sample monthly aggregated rows:")
display(monthly.limit(10))   # In Databricks, display() renders DataFrame nicely. If not available, use show().
# Also show descriptive statistics for the numeric average columns
logger.info("Summary statistics for monthly averages (describe):")
display(monthly.select([f"avg_{p}" for p in POLLUTANTS]).summary())

t8 = time.perf_counter()
total_elapsed = t8 - t0
logger.info("Total pipeline execution time: %.2f seconds", total_elapsed)

# Print final counts and times
print("==== PIPELINE COMPLETE ====")
print(f"Records (hourly) processed: {sdf.count()}")
print(f"Monthly groups produced: {monthly.count()}")
print(f"Total time: {total_elapsed:.2f} seconds")
print(f"Delta table appended: {DELTA_TABLE_NAME}")
print("Sample monthly results (first 10 rows):")
monthly.show(10, truncate=False)

# If you would like to query the table afterwards, example:
print("Example SELECT from Delta table:")
spark.sql(f"SELECT * FROM {DELTA_TABLE_NAME} ORDER BY year, month LIMIT 10").show(truncate=False)
