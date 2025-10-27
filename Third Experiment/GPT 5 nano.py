# Databricks Python notebook script: Open-Meteo Air Quality + Weather ETL (Bronze/Silver, Serverless-ready)
# Imports
import requests
from pyspark.sql import functions as F, types as T

# 1) Configuration

AIR_QUALITY_URL = (
    "https://air-quality-api.open-meteo.com/v1/air-quality"
    "?latitude=40.3548&longitude=18.1724"
    "&hourly=pm10,pm2_5,carbon_monoxide,carbon_dioxide,nitrogen_dioxide,sulphur_dioxide,ozone"
    "&past_days=31&forecast_days=1"
)

WEATHER_URL = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude=40.3548&longitude=18.1724"
    "&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,apparent_temperature,precipitation_probability,rain,wind_speed_10m"
    "&past_days=31&forecast_days=1"
)

# Target table names (saved in the current catalog/schema context)
AIR_QUALITY_BRONZE_TBL = "air_quality_bronze"
WEATHER_BRONZE_TBL = "weather_bronze"
SILVER_TBL = "air_quality_and_weather_silver"

# 2) Helper functions
def fetch_json(url: str, timeout: int = 60) -> dict:
    """Fetch JSON from an HTTP endpoint with basic error handling."""
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()

def hourly_dict_to_rows(hourly: dict) -> list:
    """
    Convert an Open-Meteo 'hourly' object (arrays) into a list of row dicts.
    Safely aligns all arrays by the shortest length if mismatched.
    """
    if not hourly or "time" not in hourly:
        raise ValueError("Invalid hourly payload: missing 'time'.")

    # Compute lengths for all list-like fields
    lengths = {k: len(v) for k, v in hourly.items() if isinstance(v, list)}
    if not lengths:
        raise ValueError("Invalid hourly payload: no list-like fields found.")

    # Truncate to shortest length if any mismatch occurs
    min_len = min(lengths.values())
    if len(set(lengths.values())) != 1:
        print(f"Warning: hourly arrays have different lengths {lengths}; truncating to {min_len} rows.")

    time_list = hourly["time"][:min_len]
    variable_keys = [k for k in hourly.keys() if k != "time" and isinstance(hourly[k], list)]

    rows = []
    for i in range(min_len):
        row = {"time": time_list[i]}
        for k in variable_keys:
            row[k] = hourly[k][i] if i < len(hourly[k]) else None
        rows.append(row)
    return rows

def hourly_to_spark_df(hourly: dict):
    """
    Convert an Open-Meteo 'hourly' dict to a Spark DataFrame:
    - Flattens arrays into rows
    - Casts 'time' to timestamp
    - Casts numeric fields to DoubleType
    - Adds ingestion_date (DateType)
    """
    rows = hourly_dict_to_rows(hourly)
    df = spark.createDataFrame(rows)

    # Cast 'time' to timestamp
    df = df.withColumn("time", F.to_timestamp("time"))

    # Cast all non-time fields to Double (safe for ints/floats)
    numeric_cols = [c for c in df.columns if c != "time"]
    for c in numeric_cols:
        df = df.withColumn(c, F.col(c).cast(T.DoubleType()))

    # Add ingestion_date (serverless-ready; evaluated per row)
    df = df.withColumn("ingestion_date", F.current_date())

    return df

def print_null_summary(df, exclude_cols=None, title="Null Check Summary"):
    """Print null counts for all columns except those excluded."""
    exclude_cols = set(exclude_cols or [])
    cols = [c for c in df.columns if c not in exclude_cols]
    if not cols:
        print(f"{title}: No columns to evaluate.")
        return
    agg_exprs = [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in cols]
    row = df.select(agg_exprs).collect()[0]
    total_nulls = 0
    print(title)
    for c in cols:
        count_c = row[c] if row[c] is not None else 0
        print(f" {c}: {count_c}")
        total_nulls += count_c
    print(f"Total nulls across evaluated columns: {total_nulls}")

# 3) Data Extraction
print("Starting extraction from Open-Meteo APIs...")
air_quality_json = fetch_json(AIR_QUALITY_URL)
weather_json = fetch_json(WEATHER_URL)
print("Extraction completed.")

# 4) Data Processing (Flatten hourly arrays into structured rows)
print("Parsing and flattening hourly payloads into Spark DataFrames...")
air_quality_df = hourly_to_spark_df(air_quality_json.get("hourly", {}))
weather_df = hourly_to_spark_df(weather_json.get("hourly", {}))

# Basic sanity checks
print(f"Air Quality rows: {air_quality_df.count()}, columns: {len(air_quality_df.columns)}")
print(f"Weather rows: {weather_df.count()}, columns: {len(weather_df.columns)}")

# 5) Bronze Layer: Save raw, unmerged DataFrames to Delta (partitioned by ingestion_date)
print("Writing Bronze tables (Delta, partitioned by ingestion_date)...")
(
    air_quality_df.write
    .format("delta")
    .mode("append")
    .partitionBy("ingestion_date")
    .saveAsTable(AIR_QUALITY_BRONZE_TBL)
)
(
    weather_df.write
    .format("delta")
    .mode("append")
    .partitionBy("ingestion_date")
    .saveAsTable(WEATHER_BRONZE_TBL)
)
print(f"Bronze writes complete: {AIR_QUALITY_BRONZE_TBL}, {WEATHER_BRONZE_TBL}")

# 6) Merge DataFrames on 'time' (inner join)
print("Merging Air Quality and Weather data on 'time' (inner join)...")
weather_df_renamed = weather_df.withColumnRenamed("ingestion_date", "ingestion_date_weather")

merged_df = (
    air_quality_df
    .join(weather_df_renamed, on="time", how="inner")
    # Keep the primary ingestion_date from air quality and drop the secondary
    .drop("ingestion_date_weather")
)

print(f"Merged rows: {merged_df.count()}, columns: {len(merged_df.columns)}")

# 7) Data Quality Checks on merged DataFrame
print("Running Data Quality Checks...")

# Null Check (for pollutants and weather columns; exclude non-measurement fields)
exclude_for_nulls = {"time", "ingestion_date"}
print_null_summary(merged_df, exclude_cols=exclude_for_nulls, title="Null Check: Merged DataFrame")

# Duplicate Check on 'time'
merged_count_before = merged_df.count()
deduped_df = merged_df.dropDuplicates(["time"])
merged_count_after = deduped_df.count()
duplicate_count = merged_count_before - merged_count_after

print("Duplicate Check: Merged DataFrame")
print(f" Rows before dedup: {merged_count_before}")
print(f" Rows after dedup: {merged_count_after}")
print(f" Duplicate rows removed (based on time): {duplicate_count}")

# 8) Silver Layer: Save the clean, merged DataFrame to Delta (append)
print("Writing Silver table (clean, deduplicated merged data) to Delta...")
(
    deduped_df.write
    .format("delta")
    .mode("append")
    .saveAsTable(SILVER_TBL)
)
print(f"Silver write complete: {SILVER_TBL}")

print("ETL pipeline finished successfully.")
