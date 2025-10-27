# Databricks Python notebook cell
# Fetch air quality data, compute monthly averages, align schema to existing Delta table, and append.
#
# If 'requests' is not available, run: %pip install requests

import time
from datetime import datetime
import requests

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, TimestampType

spark = SparkSession.builder.getOrCreate()

API_URL = (
    "https://air-quality-api.open-meteo.com/v1/air-quality"
    "?latitude=40.3548&longitude=18.1724"
    "&hourly=pm10,pm2_5,carbon_monoxide,carbon_dioxide,nitrogen_dioxide,sulphur_dioxide,ozone"
    "&start_date=2025-03-01&end_date=2025-08-31"
)

TARGET_TABLE = "air_quality_monthly_avg"

def safe_get(lst, idx):
    try:
        return lst[idx]
    except Exception:
        return None

def fetch_api(url, timeout=30):
    print(f"[{datetime.now()}] Fetching data from API...")
    t0 = time.time()
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        elapsed = time.time() - t0
        print(f"[{datetime.now()}] Fetched API in {elapsed:.2f}s. Top-level keys: {list(data.keys())}")
        return data, elapsed
    except Exception as e:
        elapsed = time.time() - t0
        print(f"[{datetime.now()}] ERROR fetching API ({elapsed:.2f}s): {e}")
        raise

def transform_to_spark(data):
    print(f"[{datetime.now()}] Transforming JSON hourly arrays into Spark DataFrame...")
    t0 = time.time()
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
    pollutant_lists = {k: hourly.get(k, []) for k in pollutant_keys}

    if len(times) == 0:
        raise ValueError("No 'hourly.time' entries found in API response.")

    n = len(times)
    for k, lst in pollutant_lists.items():
        if len(lst) != n:
            print(f"Warning: length mismatch for {k} (len={len(lst)}) vs time (len={n}). Will safe-index.")

    rows = []
    for i in range(n):
        row = {"time": times[i]}
        for k in pollutant_keys:
            row[k] = safe_get(pollutant_lists.get(k, []), i)
        rows.append(row)

    df = spark.createDataFrame(rows)
    df = df.withColumn("time", F.to_timestamp("time"))
    for k in pollutant_keys:
        df = df.withColumn(k, F.col(k).cast(DoubleType()))
    df = df.withColumn("ingestion_date", F.current_timestamp())
    elapsed = time.time() - t0
    count = df.count()
    print(f"[{datetime.now()}] Transformation done in {elapsed:.2f}s. Hourly rows: {count}")
    return df, elapsed

# Helpers to align column naming between DataFrame and target table
def to_avg_prefix(col_name):
    # Convert pollutant_avg -> avg_pollutant
    if col_name.startswith("avg_"):
        return col_name
    if col_name.endswith("_avg"):
        core = col_name[:-4]
        return f"avg_{core}"
    return col_name

def to_suffix_avg(col_name):
    # Convert avg_pollutant -> pollutant_avg
    if col_name.endswith("_avg"):
        return col_name
    if col_name.startswith("avg_"):
        core = col_name[4:]
        return f"{core}_avg"
    return col_name

def align_df_to_table(df, table_name):
    """
    If table exists: rename df columns to match table column names, add missing table columns as nulls (with proper types),
    and ensure ingestion_date exists when table expects it.
    If table does not exist: rename pollutant_avg cols to avg_<pollutant> convention and add ingestion_date.
    Returns the adjusted DataFrame ready to be written to the table.
    """
    df_current = df
    if spark.catalog.tableExists(table_name):
        target_df = spark.table(table_name)
        target_schema = target_df.schema  # pyspark.sql.types.StructType
        target_cols = [f.name for f in target_schema]

        print(f"[{datetime.now()}] Target table '{table_name}' exists. Schema columns: {target_cols}")

        # Build mapping from current df column -> target column name
        rename_map = {}
        df_cols = df_current.columns

        # Attempt direct matches first (case sensitive)
        for c in df_cols:
            if c in target_cols:
                rename_map[c] = c

        # For unmatched df columns, try swapping suffix/prefix avg naming
        for c in df_cols:
            if c in rename_map:
                continue
            alt = to_avg_prefix(c)  # e.g. pm10_avg -> avg_pm10 or pm10_avg->pm10_avg stays
            if alt in target_cols:
                rename_map[c] = alt
                continue
            # also check suffix form
            alt2 = to_suffix_avg(c)
            if alt2 in target_cols:
                rename_map[c] = alt2
                continue

        # Apply renames
        for src, dst in rename_map.items():
            if src != dst:
                df_current = df_current.withColumnRenamed(src, dst)
                print(f"Renamed column: {src} -> {dst}")

        # Add any target columns that are missing in df as null with proper type
        df_cols_after = df_current.columns
        for field in target_schema:
            if field.name not in df_cols_after:
                print(f"Adding missing target column '{field.name}' as NULL (type: {field.dataType.simpleString()})")
                df_current = df_current.withColumn(field.name, F.lit(None).cast(field.dataType))

        # Reorder columns to match target's column order (helps avoid implicit metadata mismatch)
        df_current = df_current.select(*target_cols)
        return df_current
    else:
        # Table does not exist: create using avg_ prefix naming convention for pollutant averages
        print(f"[{datetime.now()}] Target table '{table_name}' does not exist. Adopting avg_ prefix naming for averages.")
        df_current = df_current.fillna({})  # no-op but explicit
        # Rename pollutant_avg -> avg_pollutant
        for c in df_current.columns:
            if c.endswith("_avg"):
                new = to_avg_prefix(c)
                if new != c:
                    df_current = df_current.withColumnRenamed(c, new)
                    print(f"Renamed column for new table: {c} -> {new}")
        # Ensure ingestion_date exists (we added ingestion_date in transform, but guarantee it)
        if "ingestion_date" not in df_current.columns:
            df_current = df_current.withColumn("ingestion_date", F.current_timestamp())
        # Ensure year and month exist: our aggregated DF will supply them before calling this function
        return df_current

def aggregate_monthly(df_hourly):
    print(f"[{datetime.now()}] Aggregating hourly data into monthly averages...")
    t0 = time.time()
    df2 = df_hourly.withColumn("year", F.year("time")).withColumn("month", F.month("time"))
    pollutant_cols = [
        "pm10",
        "pm2_5",
        "carbon_monoxide",
        "carbon_dioxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",
    ]
    agg_exprs = [F.avg(F.col(c)).alias(f"{c}_avg") for c in pollutant_cols]
    summary = df2.groupBy("year", "month").agg(*agg_exprs).orderBy("year", "month")
    elapsed = time.time() - t0
    count = summary.count()
    print(f"[{datetime.now()}] Aggregation done in {elapsed:.2f}s. {count} monthly rows.")
    return summary, elapsed

def prepare_for_write(df_monthly):
    # Round numeric averages for readability
    select_exprs = [F.col("year"), F.col("month")]
    for c in df_monthly.columns:
        if c.endswith("_avg"):
            select_exprs.append(F.round(F.col(c), 4).alias(c))
    # Keep any other columns (none expected)
    prepared = df_monthly.select(*select_exprs)
    return prepared

def write_aligned_to_delta(df_aligned, table_name=TARGET_TABLE):
    print(f"[{datetime.now()}] Writing DataFrame to Delta table '{table_name}' (append mode)...")
    t0 = time.time()
    try:
        if not spark.catalog.tableExists(table_name):
            # First-run: create table using overwrite (so we control schema naming)
            print(f"[{datetime.now()}] Table '{table_name}' not found. Creating table.")
            df_aligned.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(table_name)
            action = "created"
        else:
            # Table exists: Append. We already aligned schema and column order.
            df_aligned.write.format("delta").mode("append").saveAsTable(table_name)
            action = "appended"
        elapsed = time.time() - t0
        print(f"[{datetime.now()}] Write {action} in {elapsed:.2f}s.")
        return elapsed
    except Exception as e:
        elapsed = time.time() - t0
        print(f"[{datetime.now()}] ERROR during write ({elapsed:.2f}s): {e}")
        raise

def show_results(df_summary, table_name=TARGET_TABLE):
    print(f"[{datetime.now()}] Showing aggregated results and stats...")
    try:
        df_summary.show(truncate=False)
        avg_cols = [c for c in df_summary.columns if c.endswith("_avg") or c.startswith("avg_")]
        if avg_cols:
            df_summary.select(*avg_cols).describe().show(truncate=False)
        if spark.catalog.tableExists(table_name):
            print(f"Latest rows from '{table_name}':")
            spark.table(table_name).orderBy(F.col("year").desc(), F.col("month").desc()).show(10, truncate=False)
    except Exception as e:
        print(f"[{datetime.now()}] ERROR showing results: {e}")

def main():
    overall_start = time.time()
    try:
        # 1) Fetch
        data, fetch_time = fetch_api(API_URL)

        # 2) Transform hourly JSON to Spark DataFrame
        df_hourly, transform_time = transform_to_spark(data)

        # 3) Monthly aggregation (produces columns like pm10_avg, pm2_5_avg, etc.)
        df_monthly, aggregate_time = aggregate_monthly(df_hourly)

        # Prepare (rounding)
        df_monthly_prepared = prepare_for_write(df_monthly)

        # 4) Align to target table schema (rename columns to avg_<pollutant> if needed,
        # add ingestion_date if table expects it, and add missing target columns as nulls).
        df_aligned = align_df_to_table(df_monthly_prepared, TARGET_TABLE)

        # 5) Write to Delta (append if exists)
        write_time = write_aligned_to_delta(df_aligned, TARGET_TABLE)

        # 6) Show results & stats
        show_results(df_aligned, TARGET_TABLE)

        total_elapsed = time.time() - overall_start
        print("\nExecution times (seconds):")
        print(f"  Fetch:      {fetch_time:.2f}s")
        print(f"  Transform:  {transform_time:.2f}s")
        print(f"  Aggregate:  {aggregate_time:.2f}s")
        print(f"  Write:      {write_time:.2f}s")
        print(f"  Total:      {total_elapsed:.2f}s")
        print(f"[{datetime.now()}] Process completed successfully.")
    except Exception as exc:
        print(f"[{datetime.now()}] Pipeline failed: {exc}")
        raise

# Run pipeline
main()
