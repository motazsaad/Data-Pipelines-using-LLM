# Databricks Python Notebook
# End-to-end ETL for Open-Meteo Air Quality and Weather data with robust Delta schema handling.

import requests
from datetime import datetime
from pyspark.sql import functions as F
from pyspark.sql import types as T

spark.conf.set("spark.sql.shuffle.partitions", "64")

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality?latitude=40.3548&longitude=18.1724&hourly=pm10,pm2_5,carbon_monoxide,carbon_dioxide,nitrogen_dioxide,sulphur_dioxide,ozone&past_days=31&forecast_days=1"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast?latitude=40.3548&longitude=18.1724&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,apparent_temperature,precipitation_probability,rain,wind_speed_10m&past_days=31&forecast_days=1"

air_cols = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "carbon_dioxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
]
wx_cols = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "precipitation_probability",
    "rain",
    "wind_speed_10m",
]
quality_check_cols = air_cols + wx_cols

# Set these True ONCE to auto-repair schema by dropping/recreating tables if mismatches are found
REPAIR_BRONZE_SCHEMA = False
REPAIR_SILVER_SCHEMA = False

def fetch_json(url: str) -> dict:
    headers = {"User-Agent": "databricks-etl/1.0 (+https://databricks.com/)"}
    resp = requests.get(url, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()

def hourly_to_rows(hourly_obj: dict, keys: list[str]) -> list[dict]:
    times = hourly_obj.get("time", [])
    n = len(times)
    rows = []
    for i in range(n):
        row = {"time": times[i]}
        for k in keys:
            series = hourly_obj.get(k)
            row[k] = series[i] if series is not None and i < len(series) else None
        rows.append(row)
    return rows

# Explicit schemas for Bronze and Silver tables
aq_bronze_schema = T.StructType([
    T.StructField("time", T.TimestampType(), True),
    T.StructField("pm10", T.DoubleType(), True),
    T.StructField("pm2_5", T.DoubleType(), True),
    T.StructField("carbon_monoxide", T.DoubleType(), True),
    T.StructField("carbon_dioxide", T.DoubleType(), True),
    T.StructField("nitrogen_dioxide", T.DoubleType(), True),
    T.StructField("sulphur_dioxide", T.DoubleType(), True),
    T.StructField("ozone", T.DoubleType(), True),
    T.StructField("ingestion_date", T.DateType(), True),
])

wx_bronze_schema = T.StructType([
    T.StructField("time", T.TimestampType(), True),
    T.StructField("temperature_2m", T.DoubleType(), True),
    T.StructField("relative_humidity_2m", T.DoubleType(), True),
    T.StructField("dew_point_2m", T.DoubleType(), True),
    T.StructField("apparent_temperature", T.DoubleType(), True),
    T.StructField("precipitation_probability", T.DoubleType(), True),
    T.StructField("rain", T.DoubleType(), True),
    T.StructField("wind_speed_10m", T.DoubleType(), True),
    T.StructField("ingestion_date", T.DateType(), True),
])

silver_schema = T.StructType([
    T.StructField("time", T.TimestampType(), True),
    T.StructField("pm10", T.DoubleType(), True),
    T.StructField("pm2_5", T.DoubleType(), True),
    T.StructField("carbon_monoxide", T.DoubleType(), True),
    T.StructField("carbon_dioxide", T.DoubleType(), True),
    T.StructField("nitrogen_dioxide", T.DoubleType(), True),
    T.StructField("sulphur_dioxide", T.DoubleType(), True),
    T.StructField("ozone", T.DoubleType(), True),
    T.StructField("temperature_2m", T.DoubleType(), True),
    T.StructField("relative_humidity_2m", T.DoubleType(), True),
    T.StructField("dew_point_2m", T.DoubleType(), True),
    T.StructField("apparent_temperature", T.DoubleType(), True),
    T.StructField("precipitation_probability", T.DoubleType(), True),
    T.StructField("rain", T.DoubleType(), True),
    T.StructField("wind_speed_10m", T.DoubleType(), True),
    T.StructField("ingestion_date", T.DateType(), True),
])

def table_exists(table_name: str) -> bool:
    return spark.catalog.tableExists(table_name)

def schemas_compatible(actual: T.StructType, expected: T.StructType) -> bool:
    # Compare by column name and dataType; ignore nullability
    actual_map = {f.name: f.dataType.simpleString() for f in actual}
    expected_map = {f.name: f.dataType.simpleString() for f in expected}
    # All expected columns must exist and have identical types
    for k, v in expected_map.items():
        if k not in actual_map or actual_map[k] != v:
            return False
    return True

def ensure_table_with_schema(table_name: str, expected_schema: T.StructType, partition_cols: list[str], repair: bool):
    if table_exists(table_name):
        actual_schema = spark.table(table_name).schema
        if not schemas_compatible(actual_schema, expected_schema):
            msg = f"Schema mismatch detected for {table_name}. Actual: {actual_schema.json()}, Expected: {expected_schema.json()}"
            if repair:
                print(msg)
                print(f"Dropping and recreating {table_name} due to schema mismatch (repair=True).")
                spark.sql(f"DROP TABLE {table_name}")
            else:
                raise ValueError(msg + " Set REPAIR_*_SCHEMA=True to auto-recreate once.")
    # Create if not exists with explicit schema
    if not table_exists(table_name):
        # Build CREATE TABLE SQL from expected schema
        cols_sql = ",\n  ".join([f"`{f.name}` {f.dataType.simpleString().upper()}" for f in expected_schema])
        part_sql = f"PARTITIONED BY ({', '.join(partition_cols)})" if partition_cols else ""
        create_sql = f"""
        CREATE TABLE {table_name} (
          {cols_sql}
        )
        USING DELTA
        {part_sql}
        """
        spark.sql(create_sql)
        print(f"Created table {table_name} with expected schema.")

# 1) Extract
air_payload = fetch_json(AIR_QUALITY_URL)
weather_payload = fetch_json(WEATHER_URL)

# 2) Transform to flattened rows
aq_rows = hourly_to_rows(air_payload.get("hourly", {}), air_cols)
wx_rows = hourly_to_rows(weather_payload.get("hourly", {}), wx_cols)

aq_df = spark.createDataFrame(aq_rows)
wx_df = spark.createDataFrame(wx_rows)

# Cast to numeric and normalize time; add ingestion_date
def cast_to_expected_aq(df):
    return (
        df
        .withColumn("time", F.to_timestamp("time"))
        .select(
            F.col("time"),
            *[F.col(c).cast("double").alias(c) for c in air_cols],
        )
        .withColumn("ingestion_date", F.current_date().cast("date"))
    )

def cast_to_expected_wx(df):
    return (
        df
        .withColumn("time", F.to_timestamp("time"))
        .select(
            F.col("time"),
            *[F.col(c).cast("double").alias(c) for c in wx_cols],
        )
        .withColumn("ingestion_date", F.current_date().cast("date"))
    )

aq_df = cast_to_expected_aq(aq_df)
wx_df = cast_to_expected_wx(wx_df)

# 3) Ensure Bronze tables exist with correct schema (and repair if needed)
ensure_table_with_schema("air_quality_bronze", aq_bronze_schema, ["ingestion_date"], REPAIR_BRONZE_SCHEMA)
ensure_table_with_schema("weather_bronze", wx_bronze_schema, ["ingestion_date"], REPAIR_BRONZE_SCHEMA)

# 4) Bronze write (append)
(
    aq_df.select([f.name for f in aq_bronze_schema])  # enforce column order/schema
    .write.format("delta")
    .mode("append")
    .partitionBy("ingestion_date")
    .saveAsTable("air_quality_bronze")
)

(
    wx_df.select([f.name for f in wx_bronze_schema])
    .write.format("delta")
    .mode("append")
    .partitionBy("ingestion_date")
    .saveAsTable("weather_bronze")
)

print("Bronze write completed.")
print(f"air_quality_bronze rows this run: {aq_df.count()}")
print(f"weather_bronze rows this run: {wx_df.count()}")

# 5) Merge (Inner Join on time)
aq_df_renamed = aq_df.withColumnRenamed("ingestion_date", "ingestion_date_aq")
wx_df_renamed = wx_df.withColumnRenamed("ingestion_date", "ingestion_date_wx")

merged_df = aq_df_renamed.alias("aq").join(wx_df_renamed.alias("wx"), on="time", how="inner")
merged_count = merged_df.count()
print(f"Merged rows (inner join on time): {merged_count}")

# 6) Data Quality Checks
null_exprs = [F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c) for c in quality_check_cols]
null_counts_row = merged_df.select(*null_exprs).collect()[0].asDict()
total_nulls = int(sum(v for v in null_counts_row.values() if v is not None))

print("Null Check - per column:")
for c in quality_check_cols:
    print(f"  {c}: {int(null_counts_row.get(c, 0) or 0)}")
print(f"Total nulls across pollutant and weather columns: {total_nulls}")

duplicate_rows = merged_count - merged_df.dropDuplicates(["time"]).count()
print(f"Duplicate rows based on time: {duplicate_rows}")

merged_dedup = merged_df.dropDuplicates(["time"])

# 7) Prepare Silver DataFrame
silver_df = (
    merged_dedup
    .select(
        "time",
        *[F.col(c) for c in air_cols],
        *[F.col(c) for c in wx_cols],
    )
    .withColumn("ingestion_date", F.current_date().cast("date"))
)

print(f"Silver candidate rows (after dedup): {silver_df.count()}")
print(f"Silver schema: {silver_df.dtypes}")

# 8) Ensure Silver table exists with correct schema (and repair if needed)
ensure_table_with_schema("air_quality_and_weather_silver", silver_schema, [], REPAIR_SILVER_SCHEMA)

# 9) Silver write (append)
(
    silver_df.select([f.name for f in silver_schema])
    .write.format("delta")
    .mode("append")
    .saveAsTable("air_quality_and_weather_silver")
)

print("Silver write completed to table: air_quality_and_weather_silver")

# 10) Final Summary
print("=== ETL Summary Report ===")
print(f"Air Quality (bronze) rows this batch: {aq_df.count()}")
print(f"Weather (bronze) rows this batch: {wx_df.count()}")
print(f"Merged rows pre-dedup: {merged_count}")
print(f"Duplicate rows dropped: {duplicate_rows}")
print(f"Silver rows written this batch: {silver_df.count()}")
print("Null counts per column in merged (pre-dedup):")
for c in quality_check_cols:
    print(f"  {c}: {int(null_counts_row.get(c, 0) or 0)}")
print(f"Total nulls across columns (pre-dedup): {total_nulls}")
