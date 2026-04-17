from utils.data_loader import load_csv
from utils.feature_engineering_full import build_full_feature_matrix, get_feature_families

df = load_csv("GEFCom2014_clean.csv")

# The loader renames the demand column to "demand"
fe = build_full_feature_matrix(df, target="demand", temp_col="temp_avg", verbose=True)
fams = get_feature_families(fe, target="demand")

print("\nFeatures per family:")
total = 0
for k, v in fams.items():
    print(f"  {k:<16} : {len(v)} features")
    total += len(v)
print(f"  {'TOTAL':<16} : {total} features")