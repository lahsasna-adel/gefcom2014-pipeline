import pandas as pd

df = pd.read_csv("AEP_hourly.csv")
df["Datetime"] = pd.to_datetime(df["Datetime"])
df = df.sort_values("Datetime")

# Keep last 2 years (~17,520 rows)
df = df.tail(17520)
df.to_csv("AEP_2years.csv", index=False)
print(f"Saved {len(df)} rows → AEP_2years.csv")
print(df.head())