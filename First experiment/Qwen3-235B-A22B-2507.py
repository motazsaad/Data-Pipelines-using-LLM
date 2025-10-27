# Databricks notebook source
# MAGIC %md
# MAGIC # 🌫️ Air Quality ETL Pipeline (Bronze → Silver)
# MAGIC 
# MAGIC This notebook:
# MAGIC 1. Extracts air quality data from Open-Meteo API
# MAGIC 2. Loads raw data into **Bronze Delta Table**
# MAGIC 3. Applies **data quality checks** (nulls, duplicates)
# MAGIC 4. Writes clean data to **Silver Delta Table**
# MAGIC 
# MAGIC ✅ Designed for **Databricks Student Version** (serverless-compatible)
# MAGIC 🔍 Includes quality reporting

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Step 1: Install & Import Dependencies

# COMMAND ----------

import requests
import json
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window


# COMMAND ----------

# MAGIC %md
# MAGIC ## 📥 Step 2: Extract Data from API

# COMMAND ----------

# Define the API endpoint with coordinates and parameters
api_url = (
    "https://air-quality-api.open-meteo.com/v1/air-quality?"
    "latitude=40.3548&longitude=18.1724&"
    "hourly=pm10,pm2_5,carbon_monoxide,carbon_dioxide,"
    "nitrogen_dioxide,sulphur_dioxide,ozone&"
    "past_days=31&forecast_days=1"
)

# Make the request
response = requests.get(api_url)

# Check if request was successful
if response.status_code != 200:
    raise Exception(f"API request failed with status {response.status_code}: {response.text}")

# Parse JSON
data = response.json()

print("✅ Successfully fetched data from API")
print(f"Received {len(data.get('hourly', {}).get('time', []))} hourly records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🛠️ Step 3: Transform JSON to Structured PySpark DataFrame

# COMMAND ----------

# Extract hourly data
hourly_data = data['hourly']
timestamps = hourly_data['time']
pollutants = {
    'pm10': hourly_data['pm10'],
    'pm2_5': hourly_data['pm2_5'],
    'carbon_monoxide': hourly_data['carbon_monoxide'],
    'carbon_dioxide': hourly_data['carbon_dioxide'],
    'nitrogen_dioxide': hourly_data['nitrogen_dioxide'],
    'sulphur_dioxide': hourly_data['sulphur_dioxide'],
    'ozone': hourly_data['ozone']
}

# Combine into list of tuples
rows = []
for i in range(len(timestamps)):
    row = [timestamps[i]]
    row.extend([
        pollutants['pm10'][i],
        pollutants['pm2_5'][i],
        pollutants['carbon_monoxide'][i],
        pollutants['carbon_dioxide'][i],
        pollutants['nitrogen_dioxide'][i],
        pollutants['sulphur_dioxide'][i],
        pollutants['ozone'][i]
    ])
    rows.append(row)

# Define schema
schema = StructType([
    StructField("timestamp", StringType(), True),
    StructField("pm10", DoubleType(), True),
    StructField("pm2_5", DoubleType(), True),
    StructField("carbon_monoxide", DoubleType(), True),
    StructField("carbon_dioxide", DoubleType(), True),
    StructField("nitrogen_dioxide", DoubleType(), True),
    StructField("sulphur_dioxide", DoubleType(), True),
    StructField("ozone", DoubleType(), True)
])

# Create DataFrame
df_raw = spark.createDataFrame(rows, schema=schema)

# Add ingestion timestamp
ingestion_date = datetime.now().strftime("%Y-%m-%d")
df_with_ingestion = df_raw.withColumn("ingestion_date", lit(ingestion_date).cast(DateType()))

print(f"📊 Raw DataFrame created with {df_with_ingestion.count()} rows")

# Display sample
display(df_with_ingestion.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🗃️ Step 4: Save to Bronze Delta Table (Append Mode)

# COMMAND ----------

# Define table path/name
bronze_table_name = "air_quality_bronze"

# Write to Delta table (partitioned by ingestion_date, append mode)
(df_with_ingestion
 .write
 .mode("append")
 .partitionBy("ingestion_date")
 .format("delta")
 .saveAsTable(bronze_table_name)
)

print(f"✅ Raw data saved to Delta table: `{bronze_table_name}` (partitioned by ingestion_date)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔍 Step 5: Data Quality Checks on Bronze Layer

# COMMAND ----------

# Read from bronze table to apply checks
df_bronze = spark.table(bronze_table_name)

print("🔍 Running data quality checks...")

# -------------------------------
# CHECK 1: NULL Values in Pollutant Columns
# -------------------------------
pollutant_cols = [
    'pm10', 'pm2_5', 'carbon_monoxide',
    'carbon_dioxide', 'nitrogen_dioxide',
    'sulphur_dioxide', 'ozone'
]

null_report = {}
total_null_rows = 0

for col_name in pollutant_cols:
    null_count = df_bronze.filter(col(col_name).isNull()).count()
    if null_count > 0:
        null_report[col_name] = null_count
        total_null_rows += null_count

# Deduplicate rows based on timestamp
dup_check_df = df_bronze.withColumn("row_number", row_number().over(
    Window.partitionBy("timestamp").orderBy("ingestion_date")
))
duplicates_count = dup_check_df.filter(col("row_number") > 1).count()

# Keep only first occurrence (clean DataFrame)
df_no_dupes = dup_check_df.filter(col("row_number") == 1).drop("row_number")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📝 Step 6: Generate Data Quality Report

# COMMAND ----------

# Print quality summary
print("="*50)
print("📊 DATA QUALITY REPORT")
print("="*50)

print(f"Total records in bronze: {df_bronze.count()}")
print(f"Duplicate timestamps found: {duplicates_count}")

if null_report:
    print("❌ NULL VALUES DETECTED:")
    for col, count in null_report.items():
        print(f"  - {col}: {count} null(s)")
else:
    print("✅ No null values found in pollutant columns")

if duplicates_count > 0:
    print(f"⚠️  Removed {duplicates_count} duplicate timestamp entries")
else:
    print("✅ No duplicate timestamps found")

print("-"*50)

# Final clean count after deduplication
clean_row_count = df_no_dupes.count()
print(f"Final clean row count after removing duplicates: {clean_row_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✨ Step 7: Save Clean Data to Silver Delta Table

# COMMAND ----------

silver_table_name = "air_quality_silver"

# Only write clean data (after deduplication)
# Note: Nulls are allowed in scientific datasets unless specified otherwise.
# We keep them for transparency but could impute later.

(df_no_dupes
 .write
 .mode("append")
 .format("delta")
 .saveAsTable(silver_table_name)
)

print(f"✅ Clean data saved to `{silver_table_name}` (append mode)")

# Optional: Optimize table (small scale, optional)
spark.sql(f"OPTIMIZE {silver_table_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🏁 Final Summary

# COMMAND ----------

print("="*60)
print("🚀 AIR QUALITY PIPELINE COMPLETED")
print("="*60)
print(f"📍 Location: Latitude 40.3548, Longitude 18.1724")
print(f"📅 Data Range: Last 31 days + 1-day forecast")
print(f"📦 Pollutants: PM10, PM2.5, CO, CO₂, NO₂, SO₂, O₃")
print("")
print(f"💾 BRONZE TABLE: `{bronze_table_name}`")
print(f"   • Partitioned by: ingestion_date")
print(f"   • Mode: append")
print("")
print(f"✨ SILVER TABLE: `{silver_table_name}`")
print(f"   • Contains de-duplicated records")
print(f"   • Mode: append")
print("")
print(f"✅ Pipeline succeeded without critical failures.")
print("="*60)

# Optionally display sample of silver data
print("\n📋 Sample of clean silver data:")
display(spark.table(silver_table_name).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔧 Notes for Student Version Compatibility
# MAGIC 
# MAGIC - Uses **serverless-compatible** commands (`requests`, `pyspark`)
# MAGIC - Avoids cluster restarts or external libraries beyond standard Python/PySpark
# MAGIC - Delta tables stored in **default workspace storage**
# MAGIC - All operations use **append mode** as required
# MAGIC - No streaming or complex orchestration
# MAGIC 
# MAGIC 💡 You can schedule this notebook using Databricks Jobs (even in free tier).
