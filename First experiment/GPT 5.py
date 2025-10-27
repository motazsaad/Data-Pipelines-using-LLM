# Databricks Python ETL: Open-Meteo Air Quality -> Bronze/Silver Delta tables
# Serverless compute (Databricks — student/workspace)
# End-to-end: Extract from API -> Bronze (partitioned) -> Quality checks -> Silver (cleaned)

# Prerequisites:
# - requests library available in the cluster (Databricks runtime). If not, install via Libraries.
# - Delta Lake enabled. This script uses Delta tables via saveAsTable(..., format="delta").

import requests
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, to_timestamp, current_date

# 1) Extract: Fetch data from API using requests
def fetch_air_quality_json(url: str, timeout: int = 60) -> dict:
    """
    Fetch JSON payload from the given API URL.
    """
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()

# 2) Transform: Convert API JSON to a Spark DataFrame
def json_to_spark_df(spark: SparkSession, data: dict):
    """
    Translate the hourly air quality JSON into a Spark DataFrame with:
    - time (timestamp)
    - ingestion_date (to be populated later)
    - pollutant columns: pm10, pm2_5, carbon_monoxide, carbon_dioxide,
      nitrogen_dioxide, sulphur_dioxide, ozone
    """
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])

    pollutant_keys = [
        "pm10",
        "pm2_5",
        "carbon_monoxide",
        "carbon_dioxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",
    ]

    pollutant_arrays = {k: hourly.get(k, []) for k in pollutant_keys}

    rows = []
    for i, t in enumerate(times):
        row = {"time": t}
        for k in pollutant_keys:
            arr = pollutant_arrays.get(k, [])
            value = arr[i] if i < len(arr) else None
            row[k] = value
        rows.append(row)

    if not rows:
        # Return an empty DataFrame if payload is empty
        return spark.createDataFrame([], schema=None)

    df = spark.createDataFrame(rows)

    # Normalize time to Spark Timestamp
    df = df.withColumn("time", to_timestamp(col("time")))

    # Placeholder for ingestion_date; will be filled in main()
    df = df.withColumn("ingestion_date", F.lit(None).cast("date"))

    return df

def main():
    # Create Spark session (Databricks runtime provides this)
    spark = SparkSession.builder.getOrCreate()

    # API URL (provided)
    api_url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality?"
        "latitude=40.3548&longitude=18.1724&hourly=pm10,pm2_5,carbon_monoxide,"
        "carbon_dioxide,nitrogen_dioxide,sulphur_dioxide,ozone&past_days=31&forecast_days=1"
    )

    # 1) Extract
    try:
        api_data = fetch_air_quality_json(api_url, timeout=60)
        print("API data retrieved successfully.")
    except Exception as e:
        print(f"Error fetching API data: {e}")
        return

    # 2) Transform: to Spark DataFrame
    df_raw = json_to_spark_df(spark, api_data)
    # Emptiness check without RDDs: use DataFrame.count()
    if df_raw.count() == 0:
        print("No data transformed from API payload.")
        return

    pollutant_cols = [
        "pm10",
        "pm2_5",
        "carbon_monoxide",
        "carbon_dioxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",
    ]

    # Populate ingestion_date (current date) for Bronze partitioning
    df = df_raw.withColumn("ingestion_date", current_date())

    # Reorder columns for readability
    df = df.select(["time", "ingestion_date"] + pollutant_cols)

    # 2a) Bronze: Write raw data to Delta Bronze table (partitioned by ingestion_date)
    try:
        df.write.format("delta").mode("append").partitionBy("ingestion_date").saveAsTable("air_quality_bronze")
        print("Bronze table updated: air_quality_bronze (partitioned by ingestion_date).")
    except Exception as e:
        print(f"Error writing Bronze table: {e}")
        return

    # 3) Data Quality Checks
    total_rows = df.count()

    # Null checks per pollutant column
    null_counts = {}
    for c in pollutant_cols:
        null_counts[c] = df.filter(col(c).isNull()).count()

    # Rows with any null among pollutant columns
    any_null_expr = None
    for c in pollutant_cols:
        if any_null_expr is None:
            any_null_expr = col(c).isNull()
        else:
            any_null_expr = any_null_expr | col(c).isNull()
    rows_with_any_null = df.filter(any_null_expr).count()

    # Duplicate check on "time"
    dup_times = df.groupBy("time").count().filter(col("count") > 1).collect()
    duplicate_times = [row["time"] for row in dup_times]

    # Deduplicate by time (keep first occurrence)
    df_dedup = df.dropDuplicates(["time"])
    dedup_rows = df_dedup.count()

    # Filter to clean rows: non-null for all pollutants
    clean_df = df_dedup
    for c in pollutant_cols:
        clean_df = clean_df.filter(col(c).isNotNull())
    clean_rows = clean_df.count()

    # Quality report (printed)
    print("DATA QUALITY REPORT")
    print("===================")
    print(f"Total Bronze rows before quality filtering: {total_rows}")
    print("Null counts per pollutant column:")
    for c in pollutant_cols:
        print(f" - {c}: {null_counts[c]}")
    print(f"Rows with any null pollutant value: {rows_with_any_null}")
    print(f"Duplicate timestamps found: {duplicate_times if duplicate_times else []}")
    print(f"Rows after deduplication: {dedup_rows}")
    print(f"Rows passing all quality checks (non-null for all pollutants, deduplicated): {clean_rows}")

    # 4) Final Save: write only clean rows to Silver table (append)
    if clean_rows > 0:
        try:
            clean_df.write.format("delta").mode("append").saveAsTable("air_quality_silver")
            print("Silver table updated: air_quality_silver (append).")
        except Exception as e:
            print(f"Error writing Silver table: {e}")
            return
    else:
        print("No clean data to save to Silver.")

if __name__ == "__main__":
    main()
