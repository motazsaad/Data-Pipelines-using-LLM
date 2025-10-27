# Databricks notebook source
# MAGIC %md
# MAGIC # Air Quality and Weather Data ETL Pipeline
# MAGIC This notebook extracts data from Open-Meteo APIs, processes it, and loads it into Delta tables using a Bronze/Silver architecture

# COMMAND ----------

# Import required libraries
import requests
import json
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lit, explode, arrays_zip, to_timestamp, 
    count, when, isnan, isnull, row_number, current_timestamp
)
from pyspark.sql.types import *
from pyspark.sql.window import Window

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Data Extraction

# COMMAND ----------

# Define API endpoints
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality?latitude=40.3548&longitude=18.1724&hourly=pm10,pm2_5,carbon_monoxide,carbon_dioxide,nitrogen_dioxide,sulphur_dioxide,ozone&past_days=31&forecast_days=1"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast?latitude=40.3548&longitude=18.1724&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,apparent_temperature,precipitation_probability,rain,wind_speed_10m&past_days=31&forecast_days=1"

# Function to fetch data from API
def fetch_api_data(url):
    """
    Fetch data from the given API URL
    Returns: JSON response
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from {url}: {e}")
        raise

# Fetch data from both APIs
print("Fetching air quality data...")
air_quality_data = fetch_api_data(AIR_QUALITY_URL)
print("Air quality data fetched successfully")

print("Fetching weather data...")
weather_data = fetch_api_data(WEATHER_URL)
print("Weather data fetched successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Data Processing and Transformation

# COMMAND ----------

# Function to flatten hourly data into structured rows
def flatten_hourly_data(data, data_type):
    """
    Flatten the nested hourly data structure into a list of dictionaries
    Each dictionary represents one hour of data
    """
    hourly_data = data.get('hourly', {})
    time_array = hourly_data.get('time', [])
    
    # Get all metric keys except 'time'
    metric_keys = [key for key in hourly_data.keys() if key != 'time']
    
    # Create list of dictionaries, one for each time point
    flattened_data = []
    for i, time_value in enumerate(time_array):
        row = {'time': time_value}
        
        # Add each metric value for this time point
        for metric in metric_keys:
            metric_values = hourly_data.get(metric, [])
            if i < len(metric_values):
                row[metric] = metric_values[i]
            else:
                row[metric] = None
                
        flattened_data.append(row)
    
    return flattened_data

# Flatten air quality data
air_quality_flattened = flatten_hourly_data(air_quality_data, 'air_quality')
print(f"Flattened {len(air_quality_flattened)} air quality records")

# Flatten weather data
weather_flattened = flatten_hourly_data(weather_data, 'weather')
print(f"Flattened {len(weather_flattened)} weather records")

# COMMAND ----------

# Create PySpark DataFrames from flattened data
# Define schema for air quality data
air_quality_schema = StructType([
    StructField("time", StringType(), True),
    StructField("pm10", DoubleType(), True),
    StructField("pm2_5", DoubleType(), True),
    StructField("carbon_monoxide", DoubleType(), True),
    StructField("carbon_dioxide", DoubleType(), True),
    StructField("nitrogen_dioxide", DoubleType(), True),
    StructField("sulphur_dioxide", DoubleType(), True),
    StructField("ozone", DoubleType(), True)
])

# Define schema for weather data
weather_schema = StructType([
    StructField("time", StringType(), True),
    StructField("temperature_2m", DoubleType(), True),
    StructField("relative_humidity_2m", DoubleType(), True),
    StructField("dew_point_2m", DoubleType(), True),
    StructField("apparent_temperature", DoubleType(), True),
    StructField("precipitation_probability", DoubleType(), True),
    StructField("rain", DoubleType(), True),
    StructField("wind_speed_10m", DoubleType(), True)
])

# Create DataFrames
air_quality_df = spark.createDataFrame(air_quality_flattened, schema=air_quality_schema)
weather_df = spark.createDataFrame(weather_flattened, schema=weather_schema)

# Convert time string to timestamp
air_quality_df = air_quality_df.withColumn("time", to_timestamp(col("time"), "yyyy-MM-dd'T'HH:mm"))
weather_df = weather_df.withColumn("time", to_timestamp(col("time"), "yyyy-MM-dd'T'HH:mm"))

# Add ingestion_date column
current_date = datetime.now().strftime("%Y-%m-%d")
air_quality_df = air_quality_df.withColumn("ingestion_date", lit(current_date))
weather_df = weather_df.withColumn("ingestion_date", lit(current_date))

# Add ingestion_timestamp for tracking
air_quality_df = air_quality_df.withColumn("ingestion_timestamp", current_timestamp())
weather_df = weather_df.withColumn("ingestion_timestamp", current_timestamp())

print("DataFrames created successfully")
print(f"Air quality DataFrame: {air_quality_df.count()} rows")
print(f"Weather DataFrame: {weather_df.count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Data Merging

# COMMAND ----------

# Merge the two DataFrames on the 'time' column using inner join
merged_df = air_quality_df.join(
    weather_df,
    on="time",
    how="inner"
).select(
    col("time"),
    # Air quality columns
    col("pm10"),
    col("pm2_5"),
    col("carbon_monoxide"),
    col("carbon_dioxide"),
    col("nitrogen_dioxide"),
    col("sulphur_dioxide"),
    col("ozone"),
    # Weather columns
    col("temperature_2m"),
    col("relative_humidity_2m"),
    col("dew_point_2m"),
    col("apparent_temperature"),
    col("precipitation_probability"),
    col("rain"),
    col("wind_speed_10m"),
    # Use the ingestion_date from air_quality_df (they should be the same)
    air_quality_df["ingestion_date"].alias("ingestion_date"),
    air_quality_df["ingestion_timestamp"].alias("ingestion_timestamp")
)

print(f"Merged DataFrame created with {merged_df.count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Data Quality Checks

# COMMAND ----------

# Define columns to check for nulls
pollutant_columns = ["pm10", "pm2_5", "carbon_monoxide", "carbon_dioxide", 
                    "nitrogen_dioxide", "sulphur_dioxide", "ozone"]
weather_columns = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", 
                  "apparent_temperature", "precipitation_probability", "rain", "wind_speed_10m"]
all_metric_columns = pollutant_columns + weather_columns

# Null Check - Count missing values for each column
print("=" * 80)
print("DATA QUALITY CHECK REPORT")
print("=" * 80)
print("\n1. NULL VALUE CHECK:")
print("-" * 40)

null_counts = {}
for column in all_metric_columns:
    null_count = merged_df.filter(col(column).isNull() | isnan(col(column))).count()
    null_counts[column] = null_count
    if null_count > 0:
        print(f"   {column}: {null_count} null values")

total_nulls = sum(null_counts.values())
print(f"\n   Total null values across all columns: {total_nulls}")

# Duplicate Check - Check for duplicate timestamps
print("\n2. DUPLICATE CHECK:")
print("-" * 40)

# Count total rows before deduplication
total_rows_before = merged_df.count()

# Count duplicates based on time column
duplicate_count = merged_df.groupBy("time").count().filter(col("count") > 1).count()
print(f"   Number of duplicate timestamps found: {duplicate_count}")

# If duplicates exist, show how many rows are affected
if duplicate_count > 0:
    duplicate_rows = merged_df.groupBy("time").count().filter(col("count") > 1).select("count").agg({"count": "sum"}).collect()[0][0]
    print(f"   Total rows affected by duplicates: {duplicate_rows}")

# Remove duplicates - keep only the first occurrence for each timestamp
window_spec = Window.partitionBy("time").orderBy("ingestion_timestamp")
deduplicated_df = merged_df.withColumn("row_num", row_number().over(window_spec)) \
                          .filter(col("row_num") == 1) \
                          .drop("row_num")

total_rows_after = deduplicated_df.count()
rows_removed = total_rows_before - total_rows_after

print(f"\n   Rows before deduplication: {total_rows_before}")
print(f"   Rows after deduplication: {total_rows_after}")
print(f"   Duplicate rows removed: {rows_removed}")

# Data completeness check
print("\n3. DATA COMPLETENESS:")
print("-" * 40)
print(f"   Total records in merged dataset: {total_rows_after}")
print(f"   Date range: {deduplicated_df.agg({'time': 'min'}).collect()[0][0]} to {deduplicated_df.agg({'time': 'max'}).collect()[0][0]}")

# Final clean DataFrame
clean_merged_df = deduplicated_df

print("\n" + "=" * 80)
print("Data quality checks completed successfully!")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Bronze Layer - Save Raw Data

# COMMAND ----------

# Define Bronze layer table names
AIR_QUALITY_BRONZE_TABLE = "air_quality_bronze"
WEATHER_BRONZE_TABLE = "weather_bronze"

# Save air quality data to Bronze layer
print("Saving air quality data to Bronze layer...")
air_quality_df.write \
    .mode("append") \
    .partitionBy("ingestion_date") \
    .format("delta") \
    .saveAsTable(AIR_QUALITY_BRONZE_TABLE)
print(f"Air quality data saved to table: {AIR_QUALITY_BRONZE_TABLE}")

# Save weather data to Bronze layer
print("Saving weather data to Bronze layer...")
weather_df.write \
    .mode("append") \
    .partitionBy("ingestion_date") \
    .format("delta") \
    .saveAsTable(WEATHER_BRONZE_TABLE)
print(f"Weather data saved to table: {WEATHER_BRONZE_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Silver Layer - Save Clean Merged Data

# COMMAND ----------

# Define Silver layer table name
SILVER_TABLE = "air_quality_and_weather_silver"

# Save clean merged data to Silver layer
print("Saving clean merged data to Silver layer...")
clean_merged_df.write \
    .mode("append") \
    .format("delta") \
    .saveAsTable(SILVER_TABLE)
print(f"Clean merged data saved to table: {SILVER_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Verification and Summary

# COMMAND ----------

# Verify the data was saved correctly
print("VERIFICATION SUMMARY")
print("=" * 80)

# Check Bronze tables
air_quality_bronze_count = spark.table(AIR_QUALITY_BRONZE_TABLE).count()
weather_bronze_count = spark.table(WEATHER_BRONZE_TABLE).count()
silver_count = spark.table(SILVER_TABLE).count()

print(f"\nBronze Layer:")
print(f"  - {AIR_QUALITY_BRONZE_TABLE}: {air_quality_bronze_count} records")
print(f"  - {WEATHER_BRONZE_TABLE}: {weather_bronze_count} records")

print(f"\nSilver Layer:")
print(f"  - {SILVER_TABLE}: {silver_count} records")

# Show sample of Silver table
print("\nSample data from Silver table:")
spark.table(SILVER_TABLE).select(
    "time", "pm10", "pm2_5", "temperature_2m", "relative_humidity_2m", "wind_speed_10m"
).orderBy("time", ascending=False).show(5, truncate=False)

print("\nETL Pipeline completed successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pipeline Summary
# MAGIC 
# MAGIC This ETL pipeline successfully:
# MAGIC 1. **Extracted** data from two Open-Meteo APIs (air quality and weather)
# MAGIC 2. **Transformed** the nested JSON structure into flattened PySpark DataFrames
# MAGIC 3. **Merged** the datasets on the time column using an inner join
# MAGIC 4. **Performed quality checks**:
# MAGIC    - Identified and reported null values
# MAGIC    - Detected and removed duplicate timestamps
# MAGIC 5. **Implemented Bronze/Silver architecture**:
# MAGIC    - Bronze: Raw data stored in separate tables, partitioned by ingestion_date
# MAGIC    - Silver: Clean, merged data stored in a single table
# MAGIC 
# MAGIC The pipeline is designed to run on Databricks serverless compute and can be scheduled for regular updates.
