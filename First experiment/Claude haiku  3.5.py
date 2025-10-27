# Databricks Notebook: Air Quality ETL Pipeline

# Import required libraries
import requests
import json
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType
from datetime import datetime, timezone, timedelta
import pytz

# 1. Extract Data Function
def fetch_air_quality_data():
    """
    Fetch air quality data from Open Meteo API
    """
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": 40.3548,
        "longitude": 18.1724,
        "hourly": "pm10,pm2_5,carbon_monoxide,carbon_dioxide,nitrogen_dioxide,sulphur_dioxide,ozone",
        "past_days": 31,
        "forecast_days": 1
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an exception for bad responses
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

# 2. Transform Data Function
def transform_air_quality_data(air_quality_json):
    """
    Transform JSON data into a PySpark DataFrame
    """
    # Define schema for the DataFrame
    schema = StructType([
        StructField("timestamp", TimestampType(), True),
        StructField("pm10", DoubleType(), True),
        StructField("pm2_5", DoubleType(), True),
        StructField("carbon_monoxide", DoubleType(), True),
        StructField("carbon_dioxide", DoubleType(), True),
        StructField("nitrogen_dioxide", DoubleType(), True),
        StructField("sulphur_dioxide", DoubleType(), True),
        StructField("ozone", DoubleType(), True),
        StructField("ingestion_date", TimestampType(), True)
    ])
    
    # Prepare data for DataFrame
    ingestion_date = datetime.now(timezone.utc)
    rows = []
    
    # Extract hourly data
    timestamps = air_quality_json['hourly']['time']
    pollutants = {
        'pm10': air_quality_json['hourly']['pm10'],
        'pm2_5': air_quality_json['hourly']['pm2_5'],
        'carbon_monoxide': air_quality_json['hourly']['carbon_monoxide'],
        'carbon_dioxide': air_quality_json['hourly']['carbon_dioxide'],
        'nitrogen_dioxide': air_quality_json['hourly']['nitrogen_dioxide'],
        'sulphur_dioxide': air_quality_json['hourly']['sulphur_dioxide'],
        'ozone': air_quality_json['hourly']['ozone']
    }
    
    # Create rows with proper timestamp conversion
    for i in range(len(timestamps)):
        try:
            # Convert timestamp to UTC datetime
            ts = datetime.fromisoformat(timestamps[i]).replace(tzinfo=timezone.utc)
            
            row = {
                'timestamp': ts,
                'pm10': float(pollutants['pm10'][i]) if pollutants['pm10'][i] is not None else None,
                'pm2_5': float(pollutants['pm2_5'][i]) if pollutants['pm2_5'][i] is not None else None,
                'carbon_monoxide': float(pollutants['carbon_monoxide'][i]) if pollutants['carbon_monoxide'][i] is not None else None,
                'carbon_dioxide': float(pollutants['carbon_dioxide'][i]) if pollutants['carbon_dioxide'][i] is not None else None,
                'nitrogen_dioxide': float(pollutants['nitrogen_dioxide'][i]) if pollutants['nitrogen_dioxide'][i] is not None else None,
                'sulphur_dioxide': float(pollutants['sulphur_dioxide'][i]) if pollutants['sulphur_dioxide'][i] is not None else None,
                'ozone': float(pollutants['ozone'][i]) if pollutants['ozone'][i] is not None else None,
                'ingestion_date': ingestion_date
            }
            rows.append(row)
        except Exception as e:
            print(f"Error processing timestamp {timestamps[i]}: {e}")
    
    # Create DataFrame
    df = spark.createDataFrame(rows, schema=schema)
    return df

# 3. Data Quality Checks Function
def perform_data_quality_checks(df):
    """
    Perform data quality checks and generate report
    """
    # Null Check
    null_counts = {}
    pollutant_columns = [
        'pm10', 'pm2_5', 'carbon_monoxide', 'carbon_dioxide', 
        'nitrogen_dioxide', 'sulphur_dioxide', 'ozone'
    ]
    
    for col in pollutant_columns:
        null_count = df.filter(F.col(col).isNull()).count()
        if null_count > 0:
            null_counts[col] = null_count
    
    # Duplicate Check
    total_rows = df.count()
    distinct_rows = df.dropDuplicates(['timestamp']).count()
    duplicate_count = total_rows - distinct_rows
    
    # Generate Quality Report
    print("Data Quality Report:")
    print("-" * 20)
    
    if null_counts:
        print("Null Value Counts:")
        for col, count in null_counts.items():
            print(f"{col}: {count} null values")
    else:
        print("No null values found in pollutant columns.")
    
    print(f"\nDuplicate Timestamps: {duplicate_count}")
    
    # Remove duplicates and null rows
    clean_df = df.dropDuplicates(['timestamp']).na.drop(subset=pollutant_columns)
    
    return clean_df, null_counts, duplicate_count

# Main ETL Pipeline
def air_quality_etl():
    # 1. Extract Data
    air_quality_json = fetch_air_quality_data()
    
    if not air_quality_json:
        print("Failed to fetch air quality data.")
        return
    
    # 2. Transform Data
    try:
        bronze_df = transform_air_quality_data(air_quality_json)
        
        # Save Bronze Layer
        bronze_df.write \
            .format("delta") \
            .mode("append") \
            .partitionBy("ingestion_date") \
            .saveAsTable("air_quality_bronze")
        
        # 3. Data Quality Checks
        silver_df, null_issues, duplicate_count = perform_data_quality_checks(bronze_df)
        
        # 4. Save Silver Layer
        silver_df.write \
            .format("delta") \
            .mode("append") \
            .saveAsTable("air_quality_silver")
        
        print("\nETL Process Completed Successfully!")
    
    except Exception as e:
        print(f"Error in ETL process: {e}")
        import traceback
        traceback.print_exc()

# Execute the ETL Pipeline
try:
    air_quality_etl()
except Exception as e:
    print(f"Unexpected error: {e}")
    import traceback
    traceback.print_exc()
