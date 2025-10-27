# Databricks Python notebook / script
# ETL: Open-Meteo Air Quality API -> Bronze/Silver Delta tables with data quality checks
# - Bronze: air_quality_bronze partitioned by ingestion_date (append)
# - Silver: air_quality_silver partitioned by ingestion_date (append)
# - Serverless compute friendly (Databricks student/serverless environment)

import requests
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# 1) Extract Data
def fetch_air_quality_json(api_url, timeout=60):
    """Fetch JSON payload from API using requests."""
    resp = requests.get(api_url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()

def api_to_rows(api_json):
    """
    Transform the API's hourly section into a list of dict rows.
    API structure:
      hourly: {
        time: [...],
        pm10: [...],
        pm2_5: [...],
        carbon_monoxide: [...],
        carbon_dioxide: [...],
        nitrogen_dioxide: [...],
        sulphur_dioxide: [...],
        ozone: [...]
      }
    """
    hourly = api_json.get("hourly", {})
    times = hourly.get("time", [])
    pollutant_keys = [
        "pm10",
        "pm2_5",
        "carbon_monoxide",
        "carbon_dioxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone"
    ]

    pollutant_lists = {k: hourly.get(k, []) for k in pollutant_keys}
    n = len(times)

    # Normalize all pollutant lists to length n
    for k in pollutant_keys:
        lst = pollutant_lists.get(k, [])
        if len(lst) < n:
            lst = lst + [None] * (n - len(lst))
        else:
            lst = lst[:n]
        pollutant_lists[k] = lst

    # Build rows
    rows = []
    for i in range(n):
        row = {"time": times[i]}
        for k in pollutant_keys:
            row[k] = pollutant_lists[k][i]
        rows.append(row)
    return rows

# 2) Transform and Load
def main_etl():
    # API URL (as provided)
    api_url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
        "?latitude=40.3548&longitude=18.1724"
        "&hourly=pm10,pm2_5,carbon_monoxide,carbon_dioxide,nitrogen_dioxide,sulphur_dioxide,ozone"
        "&past_days=31&forecast_days=1"
    )

    # 1) Extract
    try:
        api_json = fetch_air_quality_json(api_url, timeout=60)
        rows = api_to_rows(api_json)
        if not rows:
            print("No data retrieved from API or empty payload.")
            return
    except Exception as e:
        print(f"ERROR during API fetch/parse: {e}")
        raise

    # 2) Create DataFrame and basic normalization
    df = spark.createDataFrame(rows)
    df = df.withColumn("time", F.to_timestamp(F.col("time")))
    df = df.withColumn("ingestion_date", F.current_date())  # partitioning key

    pollutant_cols = [
        "pm10",
        "pm2_5",
        "carbon_monoxide",
        "carbon_dioxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone"
    ]

    # Bronze: save all data (append), partitioned by ingestion_date
    try:
        df.write \
          .mode("append") \
          .partitionBy("ingestion_date") \
          .format("delta") \
          .saveAsTable("air_quality_bronze")
        print("Bronze load complete: air_quality_bronze")
    except Exception as e:
        print(f"ERROR writing Bronze table: {e}")
        raise

    # 3) Data Quality Checks (on Bronze data)
    total_rows = df.count()

    # Null counts per pollutant
    null_counts = {c: df.filter(F.col(c).isNull()).count() for c in pollutant_cols}

    # Rows affected by any null in pollutant columns
    any_null_condition = None
    for c in pollutant_cols:
        cond = F.col(c).isNull()
        any_null_condition = cond if any_null_condition is None else (any_null_condition | cond)
    affected_rows_with_nulls = df.filter(any_null_condition).count() if any_null_condition is not None else 0

    # Duplicate timestamps (time)
    duplicates_per_time = df.groupBy("time").count().filter(F.col("count") > 1).count()

    # Quality report (print statements)
    print("DATA QUALITY REPORT (Bronze)")
    print(f"Total rows loaded: {total_rows}")
    print("Null value counts per pollutant:")
    for c in pollutant_cols:
        print(f" - {c}: {null_counts[c]}")
    print(f"Rows affected by any null in pollutant columns: {affected_rows_with_nulls}")
    print(f"Duplicate timestamps detected: {duplicates_per_time}")

    # 4) Final Save (Silver) - only clean rows
    # Build non-null condition for all pollutant columns
    non_null_expr = None
    for c in pollutant_cols:
        expr = F.col(c).isNotNull()
        non_null_expr = expr if non_null_expr is None else (non_null_expr & expr)

    clean_df = df.filter(non_null_expr)

    # Remove duplicates by time (keep first occurrence)
    w = Window.partitionBy("time").orderBy(F.col("time").asc())
    df_with_rn = clean_df.withColumn("rn", F.row_number().over(w))
    silver_df = df_with_rn.filter(F.col("rn") == 1).drop("rn")

    # Silver: append, partitioned by ingestion_date
    try:
        silver_df.write \
            .mode("append") \
            .partitionBy("ingestion_date") \
            .format("delta") \
            .saveAsTable("air_quality_silver")
        print("Silver load complete: air_quality_silver")
        print(f"Silver rows written: {silver_df.count()}")
    except Exception as e:
        print(f"ERROR writing Silver table: {e}")
        raise

# Execute ETL (works well in Databricks notebooks)
main_etl()
