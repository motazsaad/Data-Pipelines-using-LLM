# Databricks Notebook: Open-Meteo Data ETL Pipeline

# Import required libraries
import requests
import json
import traceback
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp, lit, expr
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType

# Initialize Spark Session
spark = SparkSession.builder.appName("OpenMeteoDataETL").getOrCreate()

# Define API Endpoints
AIR_QUALITY_API = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_API = "https://api.open-meteo.com/v1/forecast"

# Separate parameters for each API
AIR_QUALITY_PARAMS = {
    "latitude": 40.3548,
    "longitude": 18.1724,
    "hourly": "pm10,pm2_5,carbon_monoxide,carbon_dioxide,nitrogen_dioxide,sulphur_dioxide,ozone",
    "past_days": 31,
    "forecast_days": 1
}

WEATHER_PARAMS = {
    "latitude": 40.3548,
    "longitude": 18.1724,
    "hourly": "temperature_2m,relative_humidity_2m,dew_point_2m,apparent_temperature,precipitation_probability,rain,wind_speed_10m",
    "past_days": 31,
    "forecast_days": 1
}

# (Keep previous fetch_api_data and safe_fetch_data functions)

# 2. Data Processing and Merging
def create_dataframe_from_hourly_data(data, data_type):
    """
    Convert hourly API data to Spark DataFrame with enhanced error handling
    
    Args:
        data (dict): API response data
        data_type (str): Type of data (air_quality or weather)
    
    Returns:
        DataFrame: Spark DataFrame with hourly data
    """
    try:
        # Validate hourly data
        if 'hourly' not in data or 'time' not in data['hourly']:
            raise ValueError(f"Invalid hourly data structure for {data_type}")
        
        # Prepare schema dynamically based on hourly keys
        schema_fields = [
            StructField("time", TimestampType(), True),
            *[StructField(key, DoubleType(), True) for key in data['hourly'].keys() if key != 'time']
        ]
        schema = StructType(schema_fields)
        
        # Safely convert timestamps
        def safe_timestamp_convert(timestamp_str):
            try:
                return spark.sql(f"select to_timestamp('{timestamp_str}') as time").first().time
            except Exception as e:
                print(f"Timestamp conversion error for {timestamp_str}: {e}")
                return None
        
        # Zip time with other columns, handling potential conversion errors
        zipped_data = []
        for i, timestamp in enumerate(data['hourly']['time']):
            converted_time = safe_timestamp_convert(timestamp)
            if converted_time is not None:
                row_data = [converted_time]
                row_data.extend([
                    data['hourly'][col][i] if col != 'time' else None 
                    for col in data['hourly'].keys() if col != 'time'
                ])
                zipped_data.append(row_data)
        
        # Create DataFrame
        df = spark.createDataFrame(zipped_data, schema=schema)
        
        # Add metadata columns
        df = (df
              .withColumn("data_source", lit(data_type))
              .withColumn("ingestion_timestamp", current_timestamp())
        )
        
        return df
    
    except Exception as e:
        print(f"Error creating DataFrame for {data_type}: {e}")
        traceback.print_exc()
        raise

# (Keep previous error handling and data fetching code)

# 3. Data Quality Checks
def perform_data_quality_checks(df):
    """
    Perform data quality checks on DataFrame
    
    Args:
        df (DataFrame): Input DataFrame
    
    Returns:
        DataFrame: Cleaned DataFrame
    """
    # Null Check
    null_counts = df.select([col(c).isNull().cast("int").alias(c) for c in df.columns])
    null_summary = null_counts.groupBy().sum().collect()[0]
    
    print("--- Data Quality Checks ---")
    print("Null Value Counts:")
    for col_name, count in zip(df.columns, null_summary):
        print(f"{col_name}: {count}")
    
    # Duplicate Check
    total_rows = df.count()
    distinct_rows = df.dropDuplicates(["time"]).count()
    duplicate_count = total_rows - distinct_rows
    
    print(f"\nDuplicate Entries (based on time): {duplicate_count}")
    
    # Remove duplicates
    df = df.dropDuplicates(["time"])
    
    return df

# Apply data quality checks
air_quality_df = perform_data_quality_checks(air_quality_df)
weather_df = perform_data_quality_checks(weather_df)

# 4. Merge DataFrames
# Rename columns to avoid conflicts
def prepare_dataframe_for_merge(df, prefix):
    """
    Prepare DataFrame for merging by renaming columns
    
    Args:
        df (DataFrame): Input DataFrame
        prefix (str): Prefix for columns
    
    Returns:
        DataFrame: Prepared DataFrame
    """
    # Select and rename columns, excluding metadata columns
    renamed_cols = [
        col("time")
    ]
    
    # Rename data columns
    for column in df.columns:
        if column not in ['time', 'data_source', 'ingestion_timestamp']:
            renamed_cols.append(col(column).alias(f"{prefix}_{column}"))
    
    return df.select(renamed_cols)

# Prepare DataFrames for merging
air_quality_df_merged = prepare_dataframe_for_merge(air_quality_df, "aq")
weather_df_merged = prepare_dataframe_for_merge(weather_df, "wx")

# Merge DataFrames
merged_df = air_quality_df_merged.join(weather_df_merged, "time", "inner")

# 5. Layered Architecture
# Bronze Layer
def save_bronze_table(df, table_name):
    """
    Save DataFrame to Bronze layer Delta table
    
    Args:
        df (DataFrame): Input DataFrame
        table_name (str): Name of the table to save
    """
    # Remove any existing columns that might cause conflicts
    columns_to_keep = [col for col in df.columns if col not in ['ingestion_timestamp']]
    
    df.select(columns_to_keep) \
      .write.format("delta") \
      .mode("overwrite") \
      .saveAsTable(table_name)

# Save Bronze tables
save_bronze_table(air_quality_df, "air_quality_bronze")
save_bronze_table(weather_df, "weather_bronze")

# Silver Layer
merged_df.write.format("delta").mode("overwrite").saveAsTable("air_quality_and_weather_silver")

print("ETL Process Completed Successfully!")
