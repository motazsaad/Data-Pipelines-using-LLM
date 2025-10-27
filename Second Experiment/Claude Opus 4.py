# Databricks notebook source
# MAGIC %md
# MAGIC # Air Quality Data Monthly Aggregation Pipeline
# MAGIC 
# MAGIC This notebook extracts air quality data from Open-Meteo API, transforms it into a structured format,
# MAGIC calculates monthly averages for all pollutants, and saves the results to a Delta table.

# COMMAND ----------

# Import required libraries
import requests
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, explode, arrays_zip, to_timestamp, 
    year, month, avg, count, when, isnan, 
    current_timestamp, lit, round as spark_round
)
from pyspark.sql.types import *
from datetime import datetime
import time

# Initialize Spark session
spark = SparkSession.builder.appName("AirQualityMonthlyAggregation").getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Extract Data from API

# COMMAND ----------

def fetch_air_quality_data(api_url):
    """
    Fetch air quality data from the Open-Meteo API
    
    Args:
        api_url (str): The API endpoint URL
        
    Returns:
        dict: JSON response from the API
    """
    start_time = time.time()
    print(f"Fetching data from API at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        response = requests.get(api_url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        elapsed_time = time.time() - start_time
        print(f"✓ Data fetched successfully in {elapsed_time:.2f} seconds")
        print(f"  - Location: Latitude {data['latitude']}, Longitude {data['longitude']}")
        print(f"  - Time range: {len(data['hourly']['time'])} hourly records")
        
        return data
    
    except requests.exceptions.RequestException as e:
        print(f"✗ Error fetching data: {str(e)}")
        raise

# API URL with parameters
api_url = "https://air-quality-api.open-meteo.com/v1/air-quality?latitude=40.3548&longitude=18.1724&hourly=pm10,pm2_5,carbon_monoxide,carbon_dioxide,nitrogen_dioxide,sulphur_dioxide,ozone&start_date=2025-03-01&end_date=2025-08-31"

# Fetch the data
air_quality_data = fetch_air_quality_data(api_url)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Transform Data into PySpark DataFrame

# COMMAND ----------

def create_air_quality_dataframe(data):
    """
    Transform JSON data into a structured PySpark DataFrame
    
    Args:
        data (dict): JSON response from API
        
    Returns:
        DataFrame: Structured PySpark DataFrame with air quality data
    """
    start_time = time.time()
    print(f"\nTransforming data to DataFrame at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Extract hourly data
    hourly_data = data['hourly']
    
    # Create lists for DataFrame creation
    rows = []
    for i in range(len(hourly_data['time'])):
        row = {
            'timestamp': hourly_data['time'][i],
            'pm10': hourly_data['pm10'][i],
            'pm2_5': hourly_data['pm2_5'][i],
            'carbon_monoxide': hourly_data['carbon_monoxide'][i],
            'carbon_dioxide': hourly_data['carbon_dioxide'][i],
            'nitrogen_dioxide': hourly_data['nitrogen_dioxide'][i],
            'sulphur_dioxide': hourly_data['sulphur_dioxide'][i],
            'ozone': hourly_data['ozone'][i]
        }
        rows.append(row)
    
    # Create DataFrame
    df = spark.createDataFrame(rows)
    
    # Convert timestamp string to timestamp type and add date components
    df = df.withColumn("timestamp", to_timestamp(col("timestamp"), "yyyy-MM-dd'T'HH:mm")) \
           .withColumn("year", year(col("timestamp"))) \
           .withColumn("month", month(col("timestamp"))) \
           .withColumn("ingestion_date", current_timestamp())
    
    elapsed_time = time.time() - start_time
    print(f"✓ DataFrame created successfully in {elapsed_time:.2f} seconds")
    print(f"  - Total records: {df.count()}")
    print(f"  - Columns: {', '.join(df.columns)}")
    
    return df

# Create the DataFrame
df_air_quality = create_air_quality_dataframe(air_quality_data)

# Display sample data
print("\nSample data (first 10 rows):")
df_air_quality.show(10, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Data Quality Check and Null Value Analysis

# COMMAND ----------

def analyze_data_quality(df):
    """
    Analyze data quality and null values in the DataFrame
    
    Args:
        df (DataFrame): Input DataFrame
        
    Returns:
        DataFrame: Summary statistics
    """
    print(f"\nAnalyzing data quality at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get pollutant columns
    pollutant_cols = ['pm10', 'pm2_5', 'carbon_monoxide', 'carbon_dioxide', 
                      'nitrogen_dioxide', 'sulphur_dioxide', 'ozone']
    
    # Calculate null counts for each pollutant
    null_counts = []
    total_count = df.count()
    
    for col_name in pollutant_cols:
        null_count = df.filter(col(col_name).isNull() | isnan(col(col_name))).count()
        null_percentage = (null_count / total_count) * 100
        null_counts.append({
            'pollutant': col_name,
            'null_count': null_count,
            'null_percentage': round(null_percentage, 2),
            'valid_count': total_count - null_count
        })
    
    # Create summary DataFrame
    null_summary_df = spark.createDataFrame(null_counts)
    
    print("\nData Quality Summary:")
    null_summary_df.show(truncate=False)
    
    return null_summary_df

# Analyze data quality
data_quality_summary = analyze_data_quality(df_air_quality)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Monthly Aggregation

# COMMAND ----------

def calculate_monthly_averages(df):
    """
    Calculate monthly average values for all pollutants
    
    Args:
        df (DataFrame): Input DataFrame with hourly data
        
    Returns:
        DataFrame: Monthly aggregated data
    """
    start_time = time.time()
    print(f"\nCalculating monthly averages at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define pollutant columns
    pollutant_cols = ['pm10', 'pm2_5', 'carbon_monoxide', 'carbon_dioxide', 
                      'nitrogen_dioxide', 'sulphur_dioxide', 'ozone']
    
    # Create aggregation expressions
    agg_exprs = []
    for col_name in pollutant_cols:
        # Calculate average ignoring null values
        agg_exprs.append(
            spark_round(avg(when(col(col_name).isNotNull() & ~isnan(col(col_name)), col(col_name))), 2)
            .alias(f"avg_{col_name}")
        )
        # Count non-null values
        agg_exprs.append(
            count(when(col(col_name).isNotNull() & ~isnan(col(col_name)), col(col_name)))
            .alias(f"count_{col_name}")
        )
    
    # Add total record count
    agg_exprs.append(count("*").alias("total_records"))
    
    # Perform monthly aggregation
    monthly_avg_df = df.groupBy("year", "month") \
                       .agg(*agg_exprs) \
                       .orderBy("year", "month")
    
    # Add metadata columns
    monthly_avg_df = monthly_avg_df.withColumn("aggregation_date", current_timestamp()) \
                                   .withColumn("latitude", lit(40.3548)) \
                                   .withColumn("longitude", lit(18.1724))
    
    elapsed_time = time.time() - start_time
    print(f"✓ Monthly aggregation completed in {elapsed_time:.2f} seconds")
    print(f"  - Months aggregated: {monthly_avg_df.count()}")
    
    return monthly_avg_df

# Calculate monthly averages
df_monthly_avg = calculate_monthly_averages(df_air_quality)

# Display results
print("\nMonthly Average Air Quality Data:")
df_monthly_avg.select("year", "month", "avg_pm10", "avg_pm2_5", "avg_carbon_monoxide", 
                      "avg_nitrogen_dioxide", "avg_ozone", "total_records").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Create Summary Statistics

# COMMAND ----------

def create_summary_statistics(df):
    """
    Create summary statistics for the monthly aggregated data
    
    Args:
        df (DataFrame): Monthly aggregated DataFrame
        
    Returns:
        DataFrame: Summary statistics
    """
    print(f"\nGenerating summary statistics at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Calculate overall statistics for each pollutant
    pollutant_cols = ['avg_pm10', 'avg_pm2_5', 'avg_carbon_monoxide', 'avg_carbon_dioxide', 
                      'avg_nitrogen_dioxide', 'avg_sulphur_dioxide', 'avg_ozone']
    
    summary_stats = []
    for col_name in pollutant_cols:
        stats = df.select(col_name).summary("min", "max", "mean", "stddev").collect()
        
        pollutant_name = col_name.replace("avg_", "")
        summary_stats.append({
            'pollutant': pollutant_name,
            'min_monthly_avg': float(stats[0][1]) if stats[0][1] else None,
            'max_monthly_avg': float(stats[1][1]) if stats[1][1] else None,
            'overall_mean': round(float(stats[2][1]), 2) if stats[2][1] else None,
            'std_deviation': round(float(stats[3][1]), 2) if stats[3][1] else None
        })
    
    summary_df = spark.createDataFrame(summary_stats)
    
    print("\nOverall Summary Statistics:")
    summary_df.show(truncate=False)
    
    return summary_df

# Generate summary statistics
summary_stats_df = create_summary_statistics(df_monthly_avg)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Save Results to Delta Table

# COMMAND ----------

def save_to_delta_table(df, table_name, mode="append"):
    """
    Save DataFrame to Delta table
    
    Args:
        df (DataFrame): DataFrame to save
        table_name (str): Name of the Delta table
        mode (str): Save mode (append, overwrite, etc.)
    """
    start_time = time.time()
    print(f"\nSaving to Delta table '{table_name}' at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Save to Delta table
        df.write \
          .mode(mode) \
          .option("mergeSchema", "true") \
          .saveAsTable(table_name)
        
        elapsed_time = time.time() - start_time
        print(f"✓ Data saved successfully to '{table_name}' in {elapsed_time:.2f} seconds")
        
        # Verify the save
        record_count = spark.table(table_name).count()
        print(f"  - Total records in table: {record_count}")
        
    except Exception as e:
        print(f"✗ Error saving to Delta table: {str(e)}")
        raise

# Save monthly aggregated data to Delta table
save_to_delta_table(df_monthly_avg, "air_quality_monthly_avg", mode="append")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Visualization and Final Summary

# COMMAND ----------

# Create a visualization-friendly DataFrame
viz_df = df_monthly_avg.select(
    "year", 
    "month",
    "avg_pm10",
    "avg_pm2_5",
    "avg_ozone",
    "avg_nitrogen_dioxide"
).orderBy("year", "month")

print("\nMonthly Trends Summary:")
viz_df.show(truncate=False)

# Display execution summary
print("\n" + "="*80)
print("EXECUTION SUMMARY")
print("="*80)
print(f"✓ Data extraction: Successfully fetched {df_air_quality.count()} hourly records")
print(f"✓ Data transformation: Created structured DataFrame with {len(df_air_quality.columns)} columns")
print(f"✓ Monthly aggregation: Calculated averages for {df_monthly_avg.count()} months")
print(f"✓ Data persistence: Saved results to 'air_quality_monthly_avg' Delta table")
print(f"✓ Execution completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Query Verification
# MAGIC 
# MAGIC Verify the saved data by querying the Delta table

# COMMAND ----------

# Query the Delta table to verify the save
print("Verifying saved data in Delta table:")
spark.sql("""
    SELECT 
        year,
        month,
        ROUND(avg_pm10, 2) as pm10,
        ROUND(avg_pm2_5, 2) as pm2_5,
        ROUND(avg_ozone, 2) as ozone,
        ROUND(avg_nitrogen_dioxide, 2) as no2,
        total_records,
        aggregation_date
    FROM air_quality_monthly_avg
    ORDER BY year DESC, month DESC
    LIMIT 10
""").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pipeline Complete! 🎉
# MAGIC 
# MAGIC The air quality data has been successfully:
# MAGIC 1. Extracted from the Open-Meteo API
# MAGIC 2. Transformed into a structured format
# MAGIC 3. Aggregated by month with proper null handling
# MAGIC 4. Saved to the `air_quality_monthly_avg` Delta table
# MAGIC 
# MAGIC The data is now ready for further analysis, reporting, or visualization.
