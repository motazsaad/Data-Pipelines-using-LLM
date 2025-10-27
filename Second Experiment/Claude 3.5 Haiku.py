# Databricks Notebook: Air Quality Monthly Aggregation

# Import required libraries
import requests
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, year, month, current_date
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType
from datetime import datetime
import traceback

# Initialize Spark Session
spark = SparkSession.builder.appName("AirQualityMonthlyAggregation").getOrCreate()

def fetch_air_quality_data():
    """
    Fetch air quality data from the Open Meteo API
    """
    try:
        url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        params = {
            "latitude": 40.3548,
            "longitude": 18.1724,
            "hourly": "pm10,pm2_5,carbon_monoxide,carbon_dioxide,nitrogen_dioxide,sulphur_dioxide,ozone",
            "start_date": "2025-03-01",
            "end_date": "2025-08-31"
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an exception for bad responses
        
        # Print raw response for debugging
        print("Raw API Response:")
        print(json.dumps(response.json(), indent=2))
        
        return response.json()
    
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        print(f"Full traceback: {traceback.format_exc()}")
        return None

def transform_air_quality_data(air_quality_json):
    """
    Transform JSON data into a structured PySpark DataFrame
    """
    try:
        # Validate input JSON structure
        if not air_quality_json or 'hourly' not in air_quality_json:
            raise ValueError("Invalid JSON structure")
        
        # Check if all required keys exist
        required_keys = ['time', 'pm10', 'pm2_5', 'carbon_monoxide', 
                         'carbon_dioxide', 'nitrogen_dioxide', 
                         'sulphur_dioxide', 'ozone']
        
        for key in required_keys:
            if key not in air_quality_json['hourly']:
                raise ValueError(f"Missing key in hourly data: {key}")
        
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
        data = []
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
        
        # Print data lengths for debugging
        print("Data Lengths:")
        for key, value in pollutants.items():
            print(f"{key}: {len(value)}")
        print(f"Timestamps: {len(timestamps)}")
        
        ingestion_date = datetime.now()
        
        for i in range(len(timestamps)):
            row = [
                datetime.fromisoformat(timestamps[i]),
                pollutants['pm10'][i] if pollutants['pm10'][i] is not None else None,
                pollutants['pm2_5'][i] if pollutants['pm2_5'][i] is not None else None,
                pollutants['carbon_monoxide'][i] if pollutants['carbon_monoxide'][i] is not None else None,
                pollutants['carbon_dioxide'][i] if pollutants['carbon_dioxide'][i] is not None else None,
                pollutants['nitrogen_dioxide'][i] if pollutants['nitrogen_dioxide'][i] is not None else None,
                pollutants['sulphur_dioxide'][i] if pollutants['sulphur_dioxide'][i] is not None else None,
                pollutants['ozone'][i] if pollutants['ozone'][i] is not None else None,
                ingestion_date
            ]
            data.append(row)
        
        # Create DataFrame
        df = spark.createDataFrame(data, schema)
        
        # Print DataFrame info
        print("DataFrame Info:")
        df.printSchema()
        df.show(5)
        
        return df
    
    except Exception as e:
        print(f"Detailed Error in transform_air_quality_data: {e}")
        print(f"Full traceback: {traceback.format_exc()}")
        return None

def aggregate_monthly_data(df):
    """
    Group and aggregate data by month
    """
    try:
        monthly_avg = df.groupBy(
            year(col("timestamp")).alias("year"),
            month(col("timestamp")).alias("month")
        ).agg(
            avg("pm10").alias("avg_pm10"),
            avg("pm2_5").alias("avg_pm2_5"),
            avg("carbon_monoxide").alias("avg_carbon_monoxide"),
            avg("carbon_dioxide").alias("avg_carbon_dioxide"),
            avg("nitrogen_dioxide").alias("avg_nitrogen_dioxide"),
            avg("sulphur_dioxide").alias("avg_sulphur_dioxide"),
            avg("ozone").alias("avg_ozone")
        )
        
        return monthly_avg
    
    except Exception as e:
        print(f"Error aggregating data: {e}")
        print(f"Full traceback: {traceback.format_exc()}")
        return None

def main():
    """
    Main execution function
    """
    start_time = datetime.now()
    
    try:
        # Fetch air quality data
        air_quality_json = fetch_air_quality_data()
        if not air_quality_json:
            raise ValueError("Failed to fetch air quality data")
        
        # Transform data to DataFrame
        air_quality_df = transform_air_quality_data(air_quality_json)
        if air_quality_df is None:
            raise ValueError("Failed to transform air quality data")
        
        # Aggregate monthly data
        monthly_avg_df = aggregate_monthly_data(air_quality_df)
        if monthly_avg_df is None:
            raise ValueError("Failed to aggregate monthly data")
        
        # Save to Delta table
        monthly_avg_df.write.format("delta").mode("append").saveAsTable("air_quality_monthly_avg")
        
        # Display results
        monthly_avg_df.show()
        monthly_avg_df.printSchema()
        
        # Execution summary
        end_time = datetime.now()
        print(f"Execution completed in {end_time - start_time}")
    
    except Exception as e:
        print(f"Error in main execution: {e}")
        print(f"Full traceback: {traceback.format_exc()}")

# Run the main function
main()

