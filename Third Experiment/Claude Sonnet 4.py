# Databricks notebook source
# MAGIC %md
# MAGIC # Air Quality and Weather Data ETL Pipeline
# MAGIC This notebook extracts data from Open-Meteo APIs, processes it, and loads it into Delta tables using a Bronze/Silver architecture

# COMMAND ----------

# Import required libraries
import requests
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lit, current_timestamp, explode, arrays_zip, 
    count, when, isnan, isnull, row_number, to_timestamp
)
from pyspark.sql.types import *
from pyspark.sql.window import Window
from datetime import datetime

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Data Extraction

# COMMAND ----------

# Define API endpoints
air_quality_url = "https://air-quality-api.open-meteo.com/v1/air-quality?latitude=40.3548&longitude=18.1724&hourly=pm10,pm2_5,carbon_monoxide,carbon_dioxide,nitrogen_dioxide,sulphur_dioxide,ozone&past_days=31&forecast_days=1"
weather_url = "https://api.open-meteo.com/v1/forecast?latitude=40.3548&longitude=18.1724&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,apparent_temperature,precipitation_probability,rain,wind_speed_10m&past_days=31&forecast_days=1"

# Fetch data from APIs
print("Fetching air quality data...")
air_quality_response = requests.get(air_quality_url)
air_quality_data = air_quality_response.json()

print("Fetching weather data...")
weather_response = requests.get(weather_url)
weather_data = weather_response.json()

print("Data extraction completed successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Data Processing and Transformation

# COMMAND ----------

# Function to flatten hourly data into structured rows
def flatten_hourly_data(data, data_type):
    """
    Flattens the nested hourly data structure into a list of dictionaries
    """
    hourly_data = data['hourly']
    time_array = hourly_data['time']
    
    # Create list to store flattened records
    flattened_records = []
    
    # Get all keys except 'time'
    metric_keys = [key for key in hourly_data.keys() if key != 'time']
    
    # Iterate through each timestamp
    for i, timestamp in enumerate(time_array):
        record = {'time': timestamp}
        
        # Add all metrics for this timestamp
        for metric in metric_keys:
            if metric in hourly_data and i < len(hourly_data[metric]):
                record[metric] = hourly_data[metric][i]
            else:
                record[metric] = None
                
        flattened_records.append(record)
    
    return flattened_records

# COMMAND ----------

# Flatten air quality data
air_quality_flattened = flatten_hourly_data(air_quality_data, 'air_quality')
print(f"Flattened {len(air_quality_flattened)} air quality records")

# Flatten weather data
weather_flattened = flatten_hourly_data(weather_data, 'weather')
print(f"Flattened {len(weather_flattened)} weather records")

# COMMAND ----------

# Create PySpark DataFrames from flattened data
# Air quality DataFrame
air_quality_df = spark.createDataFrame(air_quality_flattened)

# Add ingestion_date column
air_quality_df = air_quality_df.withColumn("ingestion_date", current_timestamp().cast("date"))

# Convert time column to timestamp
air_quality_df = air_quality_df.withColumn("time", to_timestamp(col("time"), "yyyy-MM-dd'T'HH:mm"))

print("Air Quality DataFrame Schema:")
air_quality_df.printSchema()

# COMMAND ----------

# Weather DataFrame
weather_df = spark.createDataFrame(weather_flattened)

# Add ingestion_date column
weather_df = weather_df.withColumn("ingestion_date", current_timestamp().cast("date"))

# Convert time column to timestamp
weather_df = weather_df.withColumn("time", to_timestamp(col("time"), "yyyy-MM-dd'T'HH:mm"))

print("Weather DataFrame Schema:")
weather_df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Save Bronze Layer (Raw Data)

# COMMAND ----------

# Save air quality data to Bronze layer
print("Saving air quality data to Bronze layer...")
air_quality_df.write \
    .mode("append") \
    .partitionBy("ingestion_date") \
    .format("delta") \
    .saveAsTable("air_quality_bronze")

print("Air quality Bronze layer saved successfully!")

# COMMAND ----------

# Save weather data to Bronze layer
print("Saving weather data to Bronze layer...")
weather_df.write \
    .mode("append") \
    .partitionBy("ingestion_date") \
    .format("delta") \
    .saveAsTable("weather_bronze")

print("Weather Bronze layer saved successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Data Merging

# COMMAND ----------

# Merge air quality and weather data on time column using inner join
print("Merging air quality and weather data...")
merged_df = air_quality_df.join(
    weather_df,
    air_quality_df.time == weather_df.time,
    "inner"
).select(
    air_quality_df.time,
    air_quality_df.pm10,
    air_quality_df.pm2_5,
    air_quality_df.carbon_monoxide,
    air_quality_df.carbon_dioxide,
    air_quality_df.nitrogen_dioxide,
    air_quality_df.sulphur_dioxide,
    air_quality_df.ozone,
    weather_df.temperature_2m,
    weather_df.relative_humidity_2m,
    weather_df.dew_point_2m,
    weather_df.apparent_temperature,
    weather_df.precipitation_probability,
    weather_df.rain,
    weather_df.wind_speed_10m,
    air_quality_df.ingestion_date
)

print(f"Merged DataFrame contains {merged_df.count()} records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Data Quality Checks

# COMMAND ----------

# Define columns to check for nulls
pollutant_columns = ['pm10', 'pm2_5', 'carbon_monoxide', 'carbon_dioxide', 
                     'nitrogen_dioxide', 'sulphur_dioxide', 'ozone']
weather_columns = ['temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 
                   'apparent_temperature', 'precipitation_probability', 'rain', 'wind_speed_10m']
all_check_columns = pollutant_columns + weather_columns

# COMMAND ----------

# Null Check - Count missing values for each column
print("=" * 80)
print("DATA QUALITY CHECK REPORT")
print("=" * 80)
print("\n1. NULL VALUE CHECK:")
print("-" * 40)

null_counts = {}
for column in all_check_columns:
    null_count = merged_df.filter(col(column).isNull() | isnan(col(column))).count()
    null_counts[column] = null_count
    if null_count > 0:
        print(f"   {column}: {null_count} null values")

total_nulls = sum(null_counts.values())
print(f"\n   Total null values across all columns: {total_nulls}")

# COMMAND ----------

# Duplicate Check - Check for duplicate timestamps
print("\n2. DUPLICATE CHECK:")
print("-" * 40)

# Count total records before deduplication
total_records_before = merged_df.count()

# Check for duplicates based on time column
duplicate_count = merged_df.groupBy("time").count().filter(col("count") > 1).count()
print(f"   Number of duplicate timestamps found: {duplicate_count}")

# Get actual number of duplicate records (total duplicates - unique timestamps)
if duplicate_count > 0:
    duplicate_records = merged_df.groupBy("time").count().filter(col("count") > 1)
    total_duplicate_records = duplicate_records.agg({"count": "sum"}).collect()[0][0] - duplicate_count
    print(f"   Total duplicate records to be removed: {total_duplicate_records}")

# COMMAND ----------

# Remove duplicates - keep only first occurrence for each timestamp
print("\n3. DATA CLEANING:")
print("-" * 40)

# Create window specification for deduplication
window_spec = Window.partitionBy("time").orderBy("time")

# Add row number and keep only first occurrence
cleaned_df = merged_df.withColumn("row_num", row_number().over(window_spec)) \
    .filter(col("row_num") == 1) \
    .drop("row_num")

total_records_after = cleaned_df.count()
records_removed = total_records_before - total_records_after

print(f"   Records before cleaning: {total_records_before}")
print(f"   Records after cleaning: {total_records_after}")
print(f"   Records removed: {records_removed}")

# COMMAND ----------

# Additional quality metrics
print("\n4. DATA COMPLETENESS METRICS:")
print("-" * 40)

total_cells = total_records_after * len(all_check_columns)
total_non_null_cells = total_cells - sum(null_counts.values())
completeness_percentage = (total_non_null_cells / total_cells) * 100

print(f"   Total data cells: {total_cells}")
print(f"   Non-null data cells: {total_non_null_cells}")
print(f"   Data completeness: {completeness_percentage:.2f}%")

print("\n" + "=" * 80)
print("DATA QUALITY CHECK COMPLETED")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Save Silver Layer (Clean, Merged Data)

# COMMAND ----------

# Save cleaned and merged data to Silver layer
print("Saving cleaned and merged data to Silver layer...")

cleaned_df.write \
    .mode("append") \
    .format("delta") \
    .saveAsTable("air_quality_and_weather_silver")

print("Silver layer saved successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Verify Final Results

# COMMAND ----------

# Display sample of final Silver layer data
print("Sample of final Silver layer data:")
display(spark.table("air_quality_and_weather_silver").limit(10))

# COMMAND ----------

# Show record counts for all tables
print("Record counts for all tables:")
print(f"Air Quality Bronze: {spark.table('air_quality_bronze').count()} records")
print(f"Weather Bronze: {spark.table('weather_bronze').count()} records")
print(f"Silver Layer: {spark.table('air_quality_and_weather_silver').count()} records")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pipeline Execution Summary
# MAGIC 
# MAGIC The ETL pipeline has been successfully executed with the following steps:
# MAGIC 
# MAGIC 1. **Data Extraction**: Retrieved data from Open-Meteo Air Quality and Weather APIs
# MAGIC 2. **Data Processing**: Flattened nested JSON structures into structured DataFrames
# MAGIC 3. **Bronze Layer**: Saved raw data to partitioned Delta tables
# MAGIC 4. **Data Merging**: Performed inner join on time column
# MAGIC 5. **Quality Checks**: Identified nulls and duplicates, cleaned data
# MAGIC 6. **Silver Layer**: Saved clean, merged dataset to Delta table
# MAGIC 
# MAGIC The pipeline is designed to run on Databricks serverless compute and follows medallion architecture best practices.

