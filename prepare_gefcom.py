import pandas as pd
import numpy as np
import os

def parse_gefcom_timestamp(ts):
    ts = str(ts).strip()
    date_str, time_str = ts.split(" ")
    hour  = int(time_str.split(":")[0])
    year  = int(date_str[-4:])
    md_str = date_str[:-4]
    parsed = None
    for m_len in range(1, len(md_str)):
        d_len = len(md_str) - m_len
        if d_len < 1 or d_len > 2:
            continue
        month = int(md_str[:m_len])
        day   = int(md_str[m_len:])
        if 1 <= month <= 12 and 1 <= day <= 31:
            parsed = (year, month, day)
            break
    if parsed is None:
        raise ValueError(f"Cannot parse: '{date_str}'")
    year, month, day = parsed
    return pd.Timestamp(year=year, month=month, day=day, hour=hour)

# ── Load all 15 task files ────────────────────────────────────────────────────
base_path = "GEFCom2014-L_V2"
all_dfs   = []
for task_num in range(1, 16):
    fpath = os.path.join(base_path, f"Task {task_num}", f"L{task_num}-train.csv")
    if os.path.exists(fpath):
        df_task = pd.read_csv(fpath)
        all_dfs.append(df_task)
        print(f"  Loaded: {fpath}  ({len(df_task):,} rows)")

# ── Merge & parse ─────────────────────────────────────────────────────────────
df = pd.concat(all_dfs, ignore_index=True)
print(f"\nParsing timestamps...")
df["timestamp"] = df["TIMESTAMP"].apply(parse_gefcom_timestamp)
df = df.set_index("timestamp").sort_index()

# ── Deduplicate ───────────────────────────────────────────────────────────────
df = df[~df.index.duplicated(keep="last")]

# ── Keep only rows with valid load (2005-2011) ────────────────────────────────
df = df.rename(columns={"LOAD": "load"})
df = df.dropna(subset=["load"])
df = df[df["load"] > 0]
print(f"Rows with valid load : {len(df):,}  ({df.index[0].date()} to {df.index[-1].date()})")

# ── Fill timestamp gaps within the valid period ───────────────────────────────
full_idx = pd.date_range(df.index[0], df.index[-1], freq="h")
gaps     = full_idx.difference(df.index)
print(f"Timestamp gaps       : {len(gaps)}")
if len(gaps) > 0:
    df = df.reindex(full_idx)
    # Interpolate load
    df["load"] = df["load"].interpolate(method="time")
    # Interpolate temperature columns
    temp_cols = [f"w{i}" for i in range(1, 26)]
    df[temp_cols] = df[temp_cols].interpolate(method="time").ffill().bfill()

# ── Average temperature ───────────────────────────────────────────────────────
temp_cols = [f"w{i}" for i in range(1, 26)]
df["temp_avg"] = df[temp_cols].mean(axis=1)

# ── Convert Fahrenheit to Celsius ─────────────────────────────────────────────
temp_cols_all = temp_cols + ["temp_avg"]
df[temp_cols_all] = (df[temp_cols_all] - 32) * 5 / 9

# ── Drop unused columns ───────────────────────────────────────────────────────
df = df.drop(columns=["ZONEID", "TIMESTAMP"], errors="ignore")

# ── Final summary ─────────────────────────────────────────────────────────────
print(f"\nFinal shape  : {df.shape}")
print(f"Period       : {df.index[0]}  to  {df.index[-1]}")
print(f"Load range   : {df['load'].min():.1f} – {df['load'].max():.1f} MW")
print(f"Temp range   : {df['temp_avg'].min():.1f} – {df['temp_avg'].max():.1f} °C")
print(f"Missing load : {df['load'].isna().sum()}")
print(f"\nRows per year:")
print(df.groupby(df.index.year).size().to_string())
print(f"\nFirst 3 rows:")
print(df[["load", "temp_avg"]].head(3))

# ── Save ──────────────────────────────────────────────────────────────────────
df.index.name = "timestamp"
df.to_csv("GEFCom2014_clean.csv")
print("\nSaved → GEFCom2014_clean.csv")