# Databricks notebook source
# MAGIC %md
# MAGIC # Air Quality Data ETL Pipeline
# MAGIC This notebook extracts air quality data from Open-Meteo API, performs quality checks, and saves to Delta tables

# COMMAND ----------

# Import required libraries
import requests
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from datetime import datetime, timedelta
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Extract Data from API

# COMMAND ----------

# Define API endpoint and parameters
api_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
params = {
    "latitude": 40.3548,
    "longitude": 18.1724,
    "hourly": "pm10,pm2_5,carbon_monoxide,carbon_dioxide,nitrogen_dioxide,sulphur_dioxide,ozone",
    "past_days": 31,
    "forecast_days": 1
}

# Fetch data from API
try:
    response = requests.get(api_url, params=params)
    response.raise_for_status()
    data = response.json()
    print("✅ Successfully fetched data from API")
    print(f"API Response Status: {response.status_code}")
except requests.exceptions.RequestException as e:
    print(f"❌ Error fetching data: {e}")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Transform and Load to Bronze Layer

# COMMAND ----------

# Parse JSON data and create structured DataFrame
def parse_air_quality_data(json_data):
    """Parse JSON response and create structured rows"""
    hourly_data = json_data.get('hourly', {})
    
    # Extract time and pollutant data
    times = hourly_data.get('time', [])
    pm10 = hourly_data.get('pm10', [])
    pm2_5 = hourly_data.get('pm2_5', [])
    carbon_monoxide = hourly_data.get('carbon_monoxide', [])
    carbon_dioxide = hourly_data.get('carbon_dioxide', [])
    nitrogen_dioxide = hourly_data.get('nitrogen_dioxide', [])
    sulphur_dioxide = hourly_data.get('sulphur_dioxide', [])
    ozone = hourly_data.get('ozone', [])
    
    # Create list of rows
    rows = []
    for i in range(len(times)):
        row = {
            'timestamp': times[i] if i < len(times) else None,
            'pm10': pm10[i] if i < len(pm10) else None,
            'pm2_5': pm2_5[i] if i < len(pm2_5) else None,
            'carbon_monoxide': carbon_monoxide[i] if i < len(carbon_monoxide) else None,
            'carbon_dioxide': carbon_dioxide[i] if i < len(carbon_dioxide) else None,
            'nitrogen_dioxide': nitrogen_dioxide[i] if i < len(nitrogen_dioxide) else None,
            'sulphur_dioxide': sulphur_dioxide[i] if i < len(sulphur_dioxide) else None,
            'ozone': ozone[i] if i < len(ozone) else None,
            'latitude': json_data.get('latitude'),
            'longitude': json_data.get('longitude')
        }
        rows.append(row)
    
    return rows

# Parse the data
parsed_rows = parse_air_quality_data(data)
print(f"✅ Parsed {len(parsed_rows)} rows of air quality data")

# COMMAND ----------

# Create PySpark DataFrame
schema = StructType([
    StructField("timestamp", StringType(), True),
    StructField("pm10", DoubleType(), True),
    StructField("pm2_5", DoubleType(), True),
    StructField("carbon_monoxide", DoubleType(), True),
    StructField("carbon_dioxide", DoubleType(), True),
    StructField("nitrogen_dioxide", DoubleType(), True),
    StructField("sulphur_dioxide", DoubleType(), True),
    StructField("ozone", DoubleType(), True),
    StructField("latitude", DoubleType(), True),
    StructField("longitude", DoubleType(), True)
])

# Create DataFrame
df_bronze = spark.createDataFrame(parsed_rows, schema)

# Add ingestion_date column
df_bronze = df_bronze.withColumn("ingestion_date", current_date())

# Convert timestamp to proper datetime format
df_bronze = df_bronze.withColumn("timestamp", to_timestamp(col("timestamp")))

# Show sample data
print("Sample of Bronze data:")
df_bronze.show(5, truncate=False)
print(f"Total rows in Bronze: {df_bronze.count()}")

# COMMAND ----------

# Save to Bronze Delta table using DBFS path
# For Databricks Community Edition, we'll use the default database
bronze_table_name = "default.air_quality_bronze"

try:
    # Write to Delta table with partitioning
    df_bronze.write \
        .mode("append") \
        .partitionBy("ingestion_date") \
        .format("delta") \
        .saveAsTable(bronze_table_name)
    
    print(f"✅ Successfully saved {df_bronze.count()} rows to Bronze layer")
    
except Exception as e:
    print(f"❌ Error saving to Bronze layer: {e}")
    # If table already exists, try to append
    try:
        df_bronze.write \
            .mode("append") \
            .format("delta") \
            .insertInto(bronze_table_name)
        print(f"✅ Successfully appended {df_bronze.count()} rows to existing Bronze table")
    except:
        raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Data Quality Checks

# COMMAND ----------

# Define pollutant columns for quality checks
pollutant_columns = [
    "pm10", "pm2_5", "carbon_monoxide", "carbon_dioxide", 
    "nitrogen_dioxide", "sulphur_dioxide", "ozone"
]

# Initialize quality report
quality_report = {
    "total_rows": df_bronze.count(),
    "null_checks": {},
    "duplicate_checks": {},
    "issues_found": False
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1 Null Check

# COMMAND ----------

# Perform null checks
print("=" * 50)
print("NULL VALUE CHECK")
print("=" * 50)

null_summary = []
for col_name in pollutant_columns:
    null_count = df_bronze.filter(col(col_name).isNull()).count()
    null_percentage = (null_count / quality_report["total_rows"]) * 100
    
    quality_report["null_checks"][col_name] = {
        "null_count": null_count,
        "null_percentage": null_percentage
    }
    
    if null_count > 0:
        quality_report["issues_found"] = True
        print(f"⚠️  {col_name}: {null_count} null values ({null_percentage:.2f}%)")
        null_summary.append(f"{col_name}: {null_count} nulls")
    else:
        print(f"✅ {col_name}: No null values")

# Show sample rows with nulls
if quality_report["issues_found"]:
    print("\nSample rows with null values:")
    null_condition = " OR ".join([f"{col} IS NULL" for col in pollutant_columns])
    df_bronze.filter(null_condition).select("timestamp", *pollutant_columns).show(5, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2 Duplicate Check

# COMMAND ----------

# Check for duplicate timestamps
print("\n" + "=" * 50)
print("DUPLICATE TIMESTAMP CHECK")
print("=" * 50)

# Count duplicates
duplicate_count = df_bronze.groupBy("timestamp") \
    .count() \
    .filter(col("count") > 1) \
    .count()

quality_report["duplicate_checks"]["timestamp_duplicates"] = duplicate_count

if duplicate_count > 0:
    quality_report["issues_found"] = True
    print(f"⚠️  Found {duplicate_count} duplicate timestamps")
    
    # Show duplicate timestamps
    print("\nDuplicate timestamps:")
    df_bronze.groupBy("timestamp") \
        .count() \
        .filter(col("count") > 1) \
        .orderBy(col("count").desc()) \
        .show(10, truncate=False)
else:
    print("✅ No duplicate timestamps found")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3 Remove Duplicates and Create Clean Dataset

# COMMAND ----------

# Remove duplicates - keep first occurrence
df_clean = df_bronze.dropDuplicates(["timestamp"])

# Create a flag for rows with any null values in pollutant columns
null_condition = " OR ".join([f"{col} IS NULL" for col in pollutant_columns])
df_with_quality_flag = df_clean.withColumn(
    "has_null_values",
    when(expr(null_condition), True).otherwise(False)
)

# Filter to get only clean rows (no nulls in any pollutant column)
df_silver = df_with_quality_flag.filter(col("has_null_values") == False).drop("has_null_values")

print(f"✅ Clean data prepared:")
print(f"   - Original rows: {quality_report['total_rows']}")
print(f"   - After removing duplicates: {df_clean.count()}")
print(f"   - After removing nulls: {df_silver.count()}")
print(f"   - Rows removed: {quality_report['total_rows'] - df_silver.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Generate Quality Report and Save to Silver Layer

# COMMAND ----------

# Generate comprehensive quality report
print("\n" + "=" * 70)
print("DATA QUALITY REPORT SUMMARY")
print("=" * 70)
print(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Data Source: Open-Meteo Air Quality API")
print(f"Location: Latitude {params['latitude']}, Longitude {params['longitude']}")
print(f"Date Range: Past {params['past_days']} days + {params['forecast_days']} forecast days")
print("-" * 70)

print(f"\n📊 DATA VOLUME:")
print(f"   Total rows fetched: {quality_report['total_rows']}")
print(f"   Clean rows for Silver: {df_silver.count()}")
print(f"   Data quality score: {(df_silver.count() / quality_report['total_rows'] * 100):.2f}%")

print(f"\n🔍 QUALITY ISSUES FOUND:")
if not quality_report["issues_found"]:
    print("   ✅ No quality issues detected!")
else:
    print("   ⚠️  Issues detected:")
    
    # Null value summary
    null_issues = [f"{k}: {v['null_count']} nulls ({v['null_percentage']:.2f}%)" 
                   for k, v in quality_report["null_checks"].items() 
                   if v['null_count'] > 0]
    if null_issues:
        print(f"\n   NULL VALUES:")
        for issue in null_issues:
            print(f"      - {issue}")
    
    # Duplicate summary
    if quality_report["duplicate_checks"]["timestamp_duplicates"] > 0:
        print(f"\n   DUPLICATES:")
        print(f"      - {quality_report['duplicate_checks']['timestamp_duplicates']} duplicate timestamps found")

print("\n" + "=" * 70)

# COMMAND ----------

# Save clean data to Silver Delta table
silver_table_name = "default.air_quality_silver"

try:
    # Write to Silver Delta table
    df_silver.write \
        .mode("append") \
        .format("delta") \
        .saveAsTable(silver_table_name)
    
    print(f"\n✅ Successfully saved {df_silver.count()} clean rows to Silver layer")
    
    # Show sample of silver data
    print("\nSample of Silver data:")
    df_silver.select("timestamp", "pm10", "pm2_5", "ozone", "ingestion_date") \
        .orderBy(col("timestamp").desc()) \
        .show(5, truncate=False)
    
except Exception as e:
    print(f"❌ Error saving to Silver layer: {e}")
    # If table already exists, try to append
    try:
        df_silver.write \
            .mode("append") \
            .format("delta") \
            .insertInto(silver_table_name)
        print(f"✅ Successfully appended {df_silver.count()} rows to existing Silver table")
    except:
        raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Verify Delta Tables

# COMMAND ----------

# Verify Bronze table
print("BRONZE TABLE VERIFICATION:")
bronze_count = spark.table(bronze_table_name).count()
print(f"✅ Bronze table rows: {bronze_count}")

# Verify Silver table
print("\nSILVER TABLE VERIFICATION:")
silver_count = spark.table(silver_table_name).count()
print(f"✅ Silver table rows: {silver_count}")

# Show table history
print("\nDELTA TABLE HISTORY:")
spark.sql(f"DESCRIBE HISTORY {silver_table_name} LIMIT 5").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optional: Query the Tables

# COMMAND ----------

# Query Bronze table
print("Recent Bronze data:")
spark.sql(f"""
    SELECT timestamp, pm10, pm2_5, ozone, ingestion_date
    FROM {bronze_table_name}
    ORDER BY timestamp DESC
    LIMIT 10
""").show(truncate=False)

# Query Silver table
print("\nRecent Silver data:")
spark.sql(f"""
    SELECT timestamp, pm10, pm2_5, ozone, ingestion_date
    FROM {silver_table_name}
    ORDER BY timestamp DESC
    LIMIT 10
""").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pipeline Complete! 🎉
# MAGIC 
# MAGIC The ETL pipeline has successfully:
# MAGIC 1. ✅ Extracted air quality data from the API
# MAGIC 2. ✅ Transformed and loaded data to Bronze layer (with partitioning)
# MAGIC 3. ✅ Performed comprehensive data quality checks
# MAGIC 4. ✅ Generated detailed quality report
# MAGIC 5. ✅ Saved clean data to Silver layer
# MAGIC 
# MAGIC **Tables Created:**
# MAGIC - `default.air_quality_bronze` - Raw data with all records
# MAGIC - `default.air_quality_silver` - Clean data with quality checks passed
# MAGIC 
# MAGIC **Next Steps:**
# MAGIC - Schedule this notebook to run periodically
# MAGIC - Add additional quality checks as needed
# MAGIC - Create Gold layer aggregations for analytics
