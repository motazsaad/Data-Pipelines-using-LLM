# =============================================
# Databricks Notebook: Open-Meteo ETL Pipeline
# Environment: Databricks Serverless Compute
# Features: API Extraction, Data Quality Checks, Delta Lake (Bronze & Silver)
# =============================================

# ----------------------------
# 1. IMPORTS (Avoid wildcard to prevent name conflicts)
# ----------------------------
import requests
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, to_timestamp
from pyspark.sql.types import *
import pandas as pd
from datetime import datetime
import builtins  # To safely access Python's built-in functions

# Safely preserve Python built-ins before any potential overrides
py_sum = builtins.sum

# Get or create Spark session
spark = SparkSession.builder.getOrCreate()

# Set ingestion date
ingestion_date = datetime.now().date()
print(f"🚀 Starting ETL pipeline | Ingestion Date: {ingestion_date}")

# ----------------------------
# 2. DATA EXTRACTION FROM OPEN-METO APIs
# ----------------------------

air_quality_url = (
    "https://air-quality-api.open-meteo.com/v1/air-quality?"
    "latitude=40.3548&longitude=18.1724"
    "&hourly=pm10,pm2_5,carbon_monoxide,carbon_dioxide,nitrogen_dioxide,sulphur_dioxide,ozone"
    "&past_days=31&forecast_days=1"
)

weather_url = (
    "https://api.open-meteo.com/v1/forecast?"
    "latitude=40.3548&longitude=18.1724"
    "&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,"
    "apparent_temperature,precipitation_probability,rain,wind_speed_10m"
    "&past_days=31&forecast_days=1"
)

print("📡 Fetching data from Open-Meteo APIs...")

try:
    air_quality_response = requests.get(air_quality_url)
    air_quality_response.raise_for_status()
    air_quality_json = air_quality_response.json()
    print("✅ Air quality data fetched successfully.")
except Exception as e:
    print(f"❌ Failed to fetch air quality data: {e}")
    raise

try:
    weather_response = requests.get(weather_url)
    weather_response.raise_for_status()
    weather_json = weather_response.json()
    print("✅ Weather data fetched successfully.")
except Exception as e:
    print(f"❌ Failed to fetch weather data: {e}")
    raise

# ----------------------------
# 3. FLATTEN JSON INTO PySpark DATAFRAMES
# ----------------------------

def flatten_hourly_data(api_response):
    """
    Converts 'hourly' object into a Pandas DataFrame.
    Each timestamp becomes a row.
    """
    hourly = api_response.get("hourly", {})
    time_stamps = hourly.get("time", [])
    
    rows = []
    for idx, t in enumerate(time_stamps):
        row = {"time": t}
        for key, values in hourly.items():
            if key != "time":
                row[key] = values[idx] if idx < len(values) else None
        rows.append(row)
    return pd.DataFrame(rows)

print("🔁 Flattening API responses into structured DataFrames...")

# Convert to Spark DataFrames
air_quality_pdf = flatten_hourly_data(air_quality_json)
weather_pdf = flatten_hourly_data(weather_json)

air_quality_df = spark.createDataFrame(air_quality_pdf)
weather_df = spark.createDataFrame(weather_pdf)

# Add ingestion_date and fix time type
air_quality_df = air_quality_df.withColumn("ingestion_date", lit(ingestion_date).cast(DateType())) \
                               .withColumn("time", to_timestamp(col("time")))

weather_df = weather_df.withColumn("ingestion_date", lit(ingestion_date).cast(DateType())) \
                       .withColumn("time", to_timestamp(col("time")))

# Validate counts without caching
try:
    air_count = air_quality_df.count()
    weather_count = weather_df.count()
    print(f"📊 Air Quality Rows: {air_count}")
    print(f"📊 Weather Rows: {weather_count}")
except Exception as e:
    print(f"❌ Error counting rows: {e}")
    raise

# ----------------------------
# 4. BRONZE LAYER: SAVE RAW DATA TO DELTA TABLES
# ----------------------------

bronze_air_table = "air_quality_bronze"
bronze_weather_table = "weather_bronze"
silver_table = "air_quality_and_weather_silver"

print("📦 Writing raw data to Bronze Delta tables...")

# Write Air Quality Bronze
air_quality_df.write \
    .mode("append") \
    .format("delta") \
    .partitionBy("ingestion_date") \
    .option("mergeSchema", "true") \
    .saveAsTable(bronze_air_table)

# Write Weather Bronze
weather_df.write \
    .mode("append") \
    .format("delta") \
    .partitionBy("ingestion_date") \
    .option("mergeSchema", "true") \
    .saveAsTable(bronze_weather_table)

print(f"✅ Raw data saved to '{bronze_air_table}' and '{bronze_weather_table}'.")

# ----------------------------
# 5. MERGE ON 'time' USING INNER JOIN
# ----------------------------

print("🔗 Merging air quality and weather data on 'time' column...")
merged_df = air_quality_df.alias("aq") \
    .join(
        weather_df.alias("w"),
        col("aq.time") == col("w.time"),
        "inner"
    ) \
    .select(
        col("aq.time"),
        # Air Quality Metrics
        col("aq.pm10"),
        col("aq.pm2_5"),
        col("aq.carbon_monoxide"),
        col("aq.carbon_dioxide"),
        col("aq.nitrogen_dioxide"),
        col("aq.sulphur_dioxide"),
        col("aq.ozone"),
        # Weather Metrics
        col("w.temperature_2m"),
        col("w.relative_humidity_2m"),
        col("w.dew_point_2m"),
        col("w.apparent_temperature"),
        col("w.precipitation_probability"),
        col("w.rain"),
        col("w.wind_speed_10m"),
        # Metadata
        col("aq.ingestion_date")
    )

merged_count = merged_df.count()
print(f"✅ Merged DataFrame created with {merged_count} records.")

# ----------------------------
# 6. DATA QUALITY CHECKS
# ----------------------------

print("\n🔍 RUNNING DATA QUALITY CHECKS...")

# Define columns
pollutant_cols = ["pm10", "pm2_5", "carbon_monoxide", "carbon_dioxide",
                  "nitrogen_dioxide", "sulphur_dioxide", "ozone"]
weather_cols = ["temperature_2m", "relative_humidity_2m", "dew_point_2m",
                "apparent_temperature", "precipitation_probability", "rain", "wind_speed_10m"]
all_metric_cols = pollutant_cols + weather_cols

# NULL CHECK
print("\n🧾 Null value count per metric column:")
null_counts = {}
for col_name in all_metric_cols:
    null_cnt = merged_df.filter(col(col_name).isNull()).count()
    null_counts[col_name] = null_cnt
    status = "⚠️" if null_cnt > 0 else "✅"
    print(f"  {status} {col_name}: {null_cnt}")

# Use Python-native sum (safe from Spark override)
total_nulls = py_sum(null_counts.values())
print(f"📊 Total null values across all metrics: {total_nulls}")

# DUPLICATE CHECK ON 'time'
print("\n📋 Duplicate check on 'time' column...")
total_rows = merged_df.count()
distinct_times = merged_df.select("time").distinct().count()
duplicate_count = total_rows - distinct_times

print(f"Total rows          : {total_rows}")
print(f"Distinct timestamps : {distinct_times}")
print(f"Detected duplicates : {duplicate_count}")

# Deduplicate: Keep first occurrence
cleaned_df = merged_df.dropDuplicates(["time"])
final_row_count = cleaned_df.count()
print(f"✅ After deduplication: {final_row_count} valid records retained.")

# ----------------------------
# 7. SILVER LAYER: SAVE CLEANED DATA
# ----------------------------

print(f"\n💾 Saving cleaned data to Silver Delta table: `{silver_table}`...")

# Finalize schema and order
cleaned_df = cleaned_df.select(
    "time",
    *pollutant_cols,
    *weather_cols,
    "ingestion_date"
).orderBy("time")

# Write to Silver
cleaned_df.write \
    .mode("append") \
    .format("delta") \
    .option("mergeSchema", "true") \
    .saveAsTable(silver_table)

print(f"✅ Cleaned dataset saved to `{silver_table}`.")

# ----------------------------
# 8. FINAL SUMMARY REPORT
# ----------------------------

print("\n" + "="*60)
print("              🎉 ETL PIPELINE COMPLETED SUCCESSFULLY")
print("="*60)
print(f"Ingestion Date       : {ingestion_date}")
print(f"Bronze Tables        : {bronze_air_table}, {bronze_weather_table}")
print(f"Silver Table         : {silver_table}")
print(f"Final Record Count   : {final_row_count}")
print(f"Duplicate Timestamps : {duplicate_count}")
print(f"Total Null Values    : {total_nulls}")
print("💡 Data is now ready for analysis in Delta Lake.")
