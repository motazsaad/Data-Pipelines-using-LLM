# Databricks Notebook: Air Quality Data Processing Pipeline

import requests
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, year, month, to_timestamp, current_timestamp, lit
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType
import json
import time
from datetime import datetime

# Initialize Spark Session (Databricks handles this automatically, but good for clarity)
spark = SparkSession.builder.appName("AirQualityMonthlyAggregation").getOrCreate()

# ----------------------------
# 1. Extract Data from API
# ----------------------------

print("🚀 Starting air quality data extraction...")

# Define API endpoint
api_url = (
    "https://air-quality-api.open-meteo.com/v1/air-quality"
    "?latitude=40.3548"
    "&longitude=18.1724"
    "&hourly=pm10,pm2_5,carbon_monoxide,carbon_dioxide,nitrogen_dioxide,sulphur_dioxide,ozone"
    "&start_date=2025-03-01"
    "&end_date=2025-08-31"
)

start_time = time.time()

try:
    response = requests.get(api_url)
    response.raise_for_status()  # Raise exception for HTTP errors
    data = response.json()
    print(f"✅ Successfully fetched data from {api_url}")
except requests.exceptions.RequestException as e:
    print(f"❌ Error fetching data from API: {e}")
    raise

# Extract hourly data
hourly_data = data.get("hourly", {})
timestamps = hourly_data.get("time", [])
pollutants = {
    key: hourly_data.get(key, [])
    for key in [
        "pm10", "pm2_5", "carbon_monoxide", 
        "carbon_dioxide", "nitrogen_dioxide", 
        "sulphur_dioxide", "ozone"
    ]
}

print(f"📊 Retrieved {len(timestamps)} hourly records.")

# ----------------------------
# 2. Transform Data into PySpark DataFrame
# ----------------------------

print("🔄 Transforming data into structured format...")

# Prepare list of rows
rows = []
ingestion_datetime = datetime.utcnow()  # Use UTC to avoid timezone issues

timestamps = hourly_data.get("time", [])
pollutants = {
    key: hourly_data.get(key, [])
    for key in [
        "pm10", "pm2_5", "carbon_monoxide", 
        "carbon_dioxide", "nitrogen_dioxide", 
        "sulphur_dioxide", "ozone"
    ]
}

for i in range(len(timestamps)):
    row = {
        "timestamp": timestamps[i],
        "pm10": pollutants["pm10"][i] if i < len(pollutants["pm10"]) else None,
        "pm2_5": pollutants["pm2_5"][i] if i < len(pollutants["pm2_5"]) else None,
        "carbon_monoxide": pollutants["carbon_monoxide"][i] if i < len(pollutants["carbon_monoxide"]) else None,
        "carbon_dioxide": pollutants["carbon_dioxide"][i] if i < len(pollutants["carbon_dioxide"]) else None,
        "nitrogen_dioxide": pollutants["nitrogen_dioxide"][i] if i < len(pollutants["nitrogen_dioxide"]) else None,
        "sulphur_dioxide": pollutants["sulphur_dioxide"][i] if i < len(pollutants["sulphur_dioxide"]) else None,
        "ozone": pollutants["ozone"][i] if i < len(pollutants["ozone"]) else None,
        "ingestion_date": ingestion_datetime  # ✅ Fixed
    }
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
    StructField("ozone", DoubleType(), True),
    StructField("ingestion_date", TimestampType(), True)
])

# Create DataFrame
df_raw = spark.createDataFrame(rows, schema=schema)

# Convert timestamp to proper timestamp type and extract year/month
df_transformed = df_raw \
    .withColumn("timestamp", to_timestamp(col("timestamp"))) \
    .withColumn("year", year(col("timestamp"))) \
    .withColumn("month", month(col("timestamp")))

print("✅ Data transformation completed.")

# Show sample data
print("🔍 Sample of transformed data:")
df_transformed.show(10)

# ----------------------------
# 3. Monthly Aggregation
# ----------------------------

print("📈 Performing monthly aggregation...")

# Group by year and month, calculate average for each pollutant
# Null values are automatically ignored in avg() in Spark
df_monthly_avg = df_transformed.groupBy("year", "month") \
    .agg(
        avg("pm10").alias("avg_pm10"),
        avg("pm2_5").alias("avg_pm2_5"),
        avg("carbon_monoxide").alias("avg_carbon_monoxide"),
        avg("carbon_dioxide").alias("avg_carbon_dioxide"),
        avg("nitrogen_dioxide").alias("avg_nitrogen_dioxide"),
        avg("sulphur_dioxide").alias("avg_sulphur_dioxide"),
        avg("ozone").alias("avg_ozone")
    ) \
    .withColumn("ingestion_date", current_timestamp()) \
    .orderBy("year", "month")

print("✅ Monthly aggregation completed.")

# Show results
print("📊 Monthly Average Air Quality Summary:")
df_monthly_avg.show(truncate=False)

# Display summary statistics for numeric columns
print("📉 Summary Statistics for Pollutants:")
df_monthly_avg.select(
    "avg_pm10", "avg_pm2_5", "avg_carbon_monoxide",
    "avg_carbon_dioxide", "avg_nitrogen_dioxide",
    "avg_sulphur_dioxide", "avg_ozone"
).describe().show()

# ----------------------------
# 4. Save Results to Delta Table
# ----------------------------

table_name = "air_quality_monthly_avg"

print(f"💾 Saving results to Delta table: {table_name} (append mode)...")

try:
    df_monthly_avg.write.format("delta") \
        .mode("append") \
        .saveAsTable(table_name)
    print(f"✅ Data successfully appended to Delta table '{table_name}'.")
except Exception as e:
    print(f"❌ Error saving to Delta table: {e}")
    # Suggest creating the table first if it doesn't exist
    print("💡 Hint: You may need to create the table first using overwrite mode initially.")
    raise

# ----------------------------
# Final Execution Summary
# ----------------------------

end_time = time.time()
execution_time = end_time - start_time

print("✅✅✅ Pipeline Completed Successfully ✅✅✅")
print(f"📦 Total Records Processed: {df_transformed.count()}")
print(f"📅 Distinct Months Processed: {df_monthly_avg.count()}")
print(f"⏱  Execution Time: {execution_time:.2f} seconds")
print(f"📌 Output Table: {table_name}")

# Optional: Display data using display() in Databricks (for visualization)
try:
    display(df_monthly_avg)  # Databricks native visualization
except:
    print("ℹ️ Note: 'display()' is only available in Databricks Runtime.")
