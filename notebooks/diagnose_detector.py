import pandas as pd
import numpy as np

# Load datasets
df = pd.read_csv('../data/fake_discount_detection_results.csv')
prod_summary = pd.read_csv('../data/product_discount_summary.csv')

# 1. TOP 20 observations by fake_discount_score
cols = [
    'product_id', 'date', 'price_eur', 'product_historical_median',
    'product_historical_max', 'product_previous_price', 
    'discount_from_product_median_pct', 'discount_from_product_max_pct',
    'price_change_pct', 'product_price_volatility', 'historical_cv',
    'product_previous_count', 'fake_discount_score', 'discount_status',
    'detection_confidence', 'explanation'
]

# Check if 'product_model' exists
if 'product_model' in df.columns:
    cols.insert(1, 'product_model')

top_20_obs = df.nlargest(20, 'fake_discount_score')[cols]
print("==================================================")
print("1. TOP 20 OBSERVATIONS BY FAKE DISCOUNT SCORE")
print("==================================================")
# Print them in a readable way
for i, row in top_20_obs.iterrows():
    print(f"--- Rank ---")
    for col in cols:
        print(f"{col}: {row[col]}")
    print("")

# 2. TOP 20 products with repeated suspicious observations
# We need to compute highly suspicious vs potentially suspicious.
# Wait, product_summary already has 'suspicious_observations', 'num_observations', 'max_fake_discount_score', etc.
# We will sort by suspicious_observations descending, then max_fake_discount_score descending
print("==================================================")
print("2. TOP 20 PRODUCTS WITH REPEATED SUSPICIOUS OBS")
print("==================================================")
# Recalculate highly suspicious if not in product_summary
highly_susp_counts = df[df['discount_status'] == 'Highly Suspicious'].groupby('product_id').size().rename('highly_suspicious_observations')
prod_summary = prod_summary.merge(highly_susp_counts, on='product_id', how='left')
prod_summary['highly_suspicious_observations'] = prod_summary['highly_suspicious_observations'].fillna(0).astype(int)

top_20_prods = prod_summary.nlargest(20, ['suspicious_observations', 'max_fake_discount_score'])
prod_cols = [
    'product_id', 'num_observations', 'suspicious_observations',
    'highly_suspicious_observations', 'max_fake_discount_score',
    'avg_fake_discount_score', 'historical_median_price', 'historical_max_price'
]
if 'product_model' in df.columns:
    prod_cols.insert(1, 'product_model')

for i, row in top_20_prods.iterrows():
    print(f"--- Product ---")
    for col in prod_cols:
        print(f"{col}: {row[col]}")
    print("")

print("==================================================")
print("3 & 4. DIAGNOSTIC CHECKS")
print("==================================================")

# Do high scores have sufficient evidence?
high_scores = df[df['fake_discount_score'] >= 60]
print(f"Total scores >= 60: {len(high_scores)}")
if len(high_scores) > 0:
    print(f"Min product_previous_count for scores >= 60: {high_scores['product_previous_count'].min()}")
    print(f"Detection confidence counts for scores >= 60:\n{high_scores['detection_confidence'].value_counts().to_string()}")

# Are there missing values driving high scores?
print("\nMissing values in top 20:")
print(top_20_obs.isnull().sum())

# Impossible discount percentages?
print("\nMax discount_from_product_median_pct overall:")
print(df['discount_from_product_median_pct'].max())

print("\nMax discount_from_product_max_pct overall:")
print(df['discount_from_product_max_pct'].max())

print("\nMax price_change_pct overall:")
print(df['price_change_pct'].max())
print("\nMin price_change_pct overall:")
print(df['price_change_pct'].min())

# Check stability (CV) for high scores
print("\nHistorical CV distribution for scores >= 60:")
if len(high_scores) > 0:
    print(high_scores['historical_cv'].describe())

print("==================================================")
