# Databricks Python notebook script completo
# ETL: Estrazione da Open-Meteo (air-quality & weather), trasformazione, controlli qualità,
#       salvataggio Bronze (raw) e Silver (pulito) come tabelle Delta.
# Progettato per essere eseguito su Databricks (serverless compute).

# ---------------------------
# Import necessari
# ---------------------------
import requests
import datetime
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql import Window

# ---------------------------
# 1) Configurazione e helper
# ---------------------------

# Endpoint API richiesti
AIR_QUALITY_URL = (
    "https://air-quality-api.open-meteo.com/v1/air-quality?"
    "latitude=40.3548&longitude=18.1724&"
    "hourly=pm10,pm2_5,carbon_monoxide,carbon_dioxide,nitrogen_dioxide,sulphur_dioxide,ozone&"
    "past_days=31&forecast_days=1"
)

WEATHER_URL = (
    "https://api.open-meteo.com/v1/forecast?"
    "latitude=40.3548&longitude=18.1724&"
    "hourly=temperature_2m,relative_humidity_2m,dew_point_2m,apparent_temperature,precipitation_probability,rain,wind_speed_10m&"
    "past_days=31&forecast_days=1"
)

# Data di ingestione usata per il partitioning (YYYY-MM-DD)
ingestion_date = datetime.datetime.utcnow().date().isoformat()

def fetch_json(url):
    """Recupera JSON dall'URL usando requests con gestione errori semplice."""
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        raise RuntimeError(f"Errore fetch da {url}: {e}")

# ---------------------------
# 2) Estrazione dati
# ---------------------------

print("Fetching air quality JSON...")
aq_json = fetch_json(AIR_QUALITY_URL)
print("Fetching weather JSON...")
weather_json = fetch_json(WEATHER_URL)

# ---------------------------
# 3) Processing: flatten hourly JSON -> PySpark DataFrames
# ---------------------------

def hourly_dict_to_rows(hourly_dict):
    """
    Converte un oggetto 'hourly' (chiavi -> liste) in una lista di dict (righe),
    usando 'time' come indice. Se una serie è più corta, inserisce None.
    """
    if 'time' not in hourly_dict:
        raise ValueError("L'oggetto 'hourly' non contiene la chiave 'time'")
    times = hourly_dict['time']
    n = len(times)
    rows = []
    keys = [k for k in hourly_dict.keys() if k != 'time']
    for i, t in enumerate(times):
        row = {'time': t}
        for k in keys:
            arr = hourly_dict.get(k)
            val = arr[i] if (arr is not None and i < len(arr)) else None
            row[k] = val
        rows.append(row)
    return rows

# Estrae l'oggetto 'hourly' dai JSON
aq_hourly = aq_json.get('hourly', {})
weather_hourly = weather_json.get('hourly', {})

# Costruisce liste di righe
aq_rows = hourly_dict_to_rows(aq_hourly)
weather_rows = hourly_dict_to_rows(weather_hourly)

# Crea DataFrame Spark dalle liste di dict
print("Creazione DataFrame Spark dalle righe flatten...")
aq_df_raw = spark.createDataFrame(aq_rows)
weather_df_raw = spark.createDataFrame(weather_rows)

# Funzione per normalizzare tipi di colonna e aggiungere ingestion_date
def normalize_df(df, numeric_cols):
    """
    - Converte 'time' in timestamp.
    - Cast delle colonne numeriche a Double.
    - Aggiunge ingestion_date (string) per il partitioning.
    - Se una colonna attesa manca, la aggiunge con valori null per stabilità di schema.
    """
    df2 = df.withColumn("time", F.to_timestamp(F.col("time")))
    for c in numeric_cols:
        if c in df2.columns:
            df2 = df2.withColumn(c, F.col(c).cast(T.DoubleType()))
        else:
            df2 = df2.withColumn(c, F.lit(None).cast(T.DoubleType()))
    df2 = df2.withColumn("ingestion_date", F.lit(ingestion_date).cast(T.StringType()))
    return df2

# Elenco colonne attese per ogni API
aq_expected_cols = [
    "pm10","pm2_5","carbon_monoxide","carbon_dioxide",
    "nitrogen_dioxide","sulphur_dioxide","ozone"
]
weather_expected_cols = [
    "temperature_2m","relative_humidity_2m","dew_point_2m",
    "apparent_temperature","precipitation_probability","rain","wind_speed_10m"
]

# Normalizza DataFrame
aq_df = normalize_df(aq_df_raw, aq_expected_cols)
weather_df = normalize_df(weather_df_raw, weather_expected_cols)

# Riordina colonne per leggibilità
aq_df = aq_df.select(["time"] + aq_expected_cols + ["ingestion_date"])
weather_df = weather_df.select(["time"] + weather_expected_cols + ["ingestion_date"])

# ---------------------------
# 4) Bronze Layer: salva raw DataFrames come tabelle Delta (partitioned by ingestion_date)
# ---------------------------

print("Scrittura air_quality_bronze (Delta, append, partitioned by ingestion_date)...")
aq_df.write.format("delta").mode("append").partitionBy("ingestion_date").saveAsTable("air_quality_bronze")

print("Scrittura weather_bronze (Delta, append, partitioned by ingestion_date)...")
weather_df.write.format("delta").mode("append").partitionBy("ingestion_date").saveAsTable("weather_bronze")

# ---------------------------
# 5) Merge DataFrames (inner join su 'time')
# ---------------------------

# Rinominare ingestion_date per evitare ambiguità
weather_df_renamed = weather_df.select(
    [F.col("time")] +
    [F.col(c).alias(c) for c in weather_expected_cols] +
    [F.col("ingestion_date").alias("weather_ingestion_date")]
)

aq_df_renamed = aq_df.select(
    [F.col("time")] +
    [F.col(c).alias(c) for c in aq_expected_cols] +
    [F.col("ingestion_date").alias("aq_ingestion_date")]
)

# Join inner su 'time'
merged_df = aq_df_renamed.join(
    weather_df_renamed,
    on="time",
    how="inner"
).withColumn("ingestion_date", F.lit(ingestion_date).cast(T.StringType()))

# Seleziona colonne finali in ordine desiderato: time, pollutanti, weather, ingestion_date
merged_columns = ["time"] + aq_expected_cols + weather_expected_cols + ["ingestion_date"]
merged_df = merged_df.select(*merged_columns)

print(f"Righe nel DataFrame unito (post inner join): {merged_df.count()}")

# ---------------------------
# 6) Controlli di qualità sui dati
# ---------------------------

print("Esecuzione controlli di qualità sul DataFrame unito...")

# Null Check: conta null per ogni colonna di interesse
cols_to_check = aq_expected_cols + weather_expected_cols
null_counts_exprs = [F.count(F.when(F.col(c).isNull(), c)).alias(c + "_nulls") for c in cols_to_check]
null_counts_row = merged_df.agg(*null_counts_exprs).collect()[0].asDict()

print("Conteggio valori null per colonna:")
for k, v in null_counts_row.items():
    print(f"  {k}: {v}")

# Duplicate Check: duplicati basati sulla colonna 'time'
total_rows = merged_df.count()
distinct_times = merged_df.select("time").distinct().count()
duplicate_count = total_rows - distinct_times

print(f"Controllo duplicati: total_rows={total_rows}, distinct_times={distinct_times}, duplicates={duplicate_count}")

# Se ci sono duplicati, rimuovere lasciando la prima occorrenza per timestamp
if duplicate_count > 0:
    # Aggiunge id per ordinamento deterministico e mantiene la prima riga per timestamp
    merged_df = merged_df.withColumn("_order_id", F.monotonically_increasing_id())
    win = Window.partitionBy("time").orderBy(F.col("_order_id"))
    merged_df = merged_df.withColumn("_rn", F.row_number().over(win)).filter(F.col("_rn") == 1).drop("_order_id", "_rn")
    print(f"Duplicati rimossi. Nuovo numero di righe: {merged_df.count()}")
else:
    print("Nessun duplicato trovato sulla colonna 'time'.")

# Segnala colonne completamente null (se presenti)
fully_null_cols = [c for c in cols_to_check if null_counts_row.get(c + "_nulls", 0) >= merged_df.count()]
if fully_null_cols:
    print("Attenzione: le seguenti colonne sono completamente null nel DataFrame unito:")
    for c in fully_null_cols:
        print(f"  {c}")

# ---------------------------
# 7) Silver Layer: salva DataFrame pulito come tabella Delta (append)
# ---------------------------

print("Scrittura air_quality_and_weather_silver (Delta, append, partitioned by ingestion_date)...")
merged_df.write.format("delta").mode("append").partitionBy("ingestion_date").saveAsTable("air_quality_and_weather_silver")

print("ETL completato. Tabelle Bronze e Silver persistite come Delta.")

# ---------------------------
# 8) Report finale (stampa)
# ---------------------------

print("\nRIEPILOGO FINALE")
print("----------------")
print(f"Ingestion date (partition): {ingestion_date}")
final_row_count = merged_df.count()
print(f"Righe nel silver scritte (stimate): {final_row_count}")

print("Null counts per column (final):")
final_null_counts = merged_df.agg(*null_counts_exprs).collect()[0].asDict()
for k, v in final_null_counts.items():
    print(f"  {k}: {v}")

print(f"Duplicati rimossi: {duplicate_count}")
print("Script terminato con successo.")
