import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
import os

nb = new_notebook()

nb.cells.extend([
    new_markdown_cell("""# Fake Discount Detector: Feature Engineering (Corrected & Diagnosed)

The purpose of this notebook is to create a leakage-safe historical feature dataset that we can later use for the Fake Discount Detector and Tableau dashboard.

**CRITICAL DATA SCIENCE REQUIREMENT:** AVOID DATA LEAKAGE.
For each price observation at date D, historical features must be calculated using ONLY price observations strictly BEFORE date D (i.e. `date < D`).
"""),
    
    new_markdown_cell("""## STEP 1 — LOAD AND PREPARE DATA"""),
    new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

original_df = pd.read_csv('../data/dishwashers_historical_clean.csv')
original_df['date'] = pd.to_datetime(original_df['date'])
original_df = original_df.sort_values(by=['product_id', 'vendor', 'date']).reset_index(drop=True)

print(f"Original row count: {len(original_df)}")
"""),

    new_markdown_cell("""## STEP 2 — CREATE PRODUCT-LEVEL HISTORICAL FEATURES"""),
    new_code_cell("""product_daily = original_df.groupby(['product_id', 'date'])['price_eur'].median().reset_index()
product_daily = product_daily.sort_values(['product_id', 'date'])

grouped_product = product_daily.groupby('product_id')['price_eur']

product_daily['product_previous_count'] = grouped_product.transform(lambda x: x.shift(1).expanding().count())
product_daily['product_historical_min'] = grouped_product.transform(lambda x: x.shift(1).expanding().min())
product_daily['product_historical_max'] = grouped_product.transform(lambda x: x.shift(1).expanding().max())
product_daily['product_historical_mean'] = grouped_product.transform(lambda x: x.shift(1).expanding().mean())
product_daily['product_historical_median'] = grouped_product.transform(lambda x: x.shift(1).expanding().median())
product_daily['product_historical_std'] = grouped_product.transform(lambda x: x.shift(1).expanding().std())

product_daily = product_daily.drop(columns=['price_eur'])
df = original_df.merge(product_daily, on=['product_id', 'date'], how='left')
"""),

    new_markdown_cell("""## STEP 3 — CREATE VENDOR-SPECIFIC HISTORICAL FEATURES"""),
    new_code_cell("""vendor_daily = original_df.groupby(['product_id', 'vendor', 'date'])['price_eur'].median().reset_index()
vendor_daily = vendor_daily.sort_values(['product_id', 'vendor', 'date'])

grouped_vendor = vendor_daily.groupby(['product_id', 'vendor'])['price_eur']

vendor_daily['vendor_previous_count'] = grouped_vendor.transform(lambda x: x.shift(1).expanding().count())
vendor_daily['vendor_historical_min'] = grouped_vendor.transform(lambda x: x.shift(1).expanding().min())
vendor_daily['vendor_historical_max'] = grouped_vendor.transform(lambda x: x.shift(1).expanding().max())
vendor_daily['vendor_historical_mean'] = grouped_vendor.transform(lambda x: x.shift(1).expanding().mean())
vendor_daily['vendor_historical_median'] = grouped_vendor.transform(lambda x: x.shift(1).expanding().median())
vendor_daily['vendor_historical_std'] = grouped_vendor.transform(lambda x: x.shift(1).expanding().std())

vendor_daily['previous_price'] = grouped_vendor.shift(1)
vendor_daily = vendor_daily.drop(columns=['price_eur'])

df = df.merge(vendor_daily, on=['product_id', 'vendor', 'date'], how='left')
"""),

    new_markdown_cell("""## STEP 4 — PRICE CHANGE FEATURES"""),
    new_code_cell("""df['price_change'] = df['price_eur'] - df['previous_price']
df['price_change_pct'] = np.where(
    df['previous_price'] == 0, 
    np.nan, 
    (df['price_change'] / df['previous_price']) * 100
)
"""),

    new_markdown_cell("""## STEP 5 — DIAGNOSING THE ZERO SUFFICIENCY ISSUE

**What we are doing:**
Investigating why `product_history_sufficient` and `vendor_history_sufficient` are both yielding zero True values when requiring a threshold of 3 distinct previous dates.
"""),
    
    new_code_cell("""# 1. Descriptive statistics for counts
print("1. Descriptive statistics:")
display(df[['product_previous_count', 'vendor_previous_count']].describe())

# 2. Maximum values
print("\\n2. Maximum values:")
print("Max product_previous_count:", df["product_previous_count"].max())
print("Max vendor_previous_count:", df["vendor_previous_count"].max())

# 3. Value distributions
print("\\n3. Value distributions:")
print("--- product_previous_count ---")
print(df["product_previous_count"].value_counts().sort_index())
print("--- vendor_previous_count ---")
print(df["vendor_previous_count"].value_counts().sort_index())

# 4. Show 20 rows with the largest product_previous_count
print("\\n4. Top 20 rows with largest product_previous_count:")
display(df.nlargest(20, 'product_previous_count')[['product_id', 'date', 'product_previous_count']])

# 5. Show 20 rows with the largest vendor_previous_count
print("\\n5. Top 20 rows with largest vendor_previous_count:")
display(df.nlargest(20, 'vendor_previous_count')[['product_id', 'vendor', 'date', 'vendor_previous_count']])

# 6. Check data types
print("\\n6. Data types:")
print("product_previous_count dtype:", df["product_previous_count"].dtype)
print("vendor_previous_count dtype:", df["vendor_previous_count"].dtype)

# 7. Check the sufficiency flags (using >= 3)
product_suff = df["product_previous_count"] >= 3
vendor_suff = df["vendor_previous_count"] >= 3

print("\\n7. Sufficiency flags (Threshold >= 3):")
print("product_history_sufficient:")
print(product_suff.value_counts(dropna=False))
print("vendor_history_sufficient:")
print(vendor_suff.value_counts(dropna=False))

# 8. Verify exact calculation and dataset dates
print("\\n8. Investigation into dataset dates:")
print("Total unique dates in dataset:", df['date'].nunique())
print("The dates are:", df['date'].unique())
"""),

    new_markdown_cell("""## STEP 6 — FIXING FEATURE DEPENDENCIES (SUFFICIENCY)

**The Diagnosis:**
The original dataset only contains 3 unique dates in total: `2024-06-26`, `2024-07-26`, and `2024-08-26`. 
Because our rigorous methodology counts **strictly previous distinct dates**, an observation on August 26th can have at most **2** previous dates (June and July). 
Since the maximum mathematical possibility for `previous_count` is 2, the condition `>= 3` is universally `False`. The calculation logic is flawlessly correct, but the dataset slice simply does not span enough months to meet the rigorous 3-previous-month threshold.

**The Fix:**
We will *not* lower the threshold or invent values. We will calculate the sufficiency flags correctly. 
However, any derived statistical features (like median discount percentage) must be strictly `NaN` if the history is insufficient. We will update the derived calculations to depend entirely on these sufficiency flags.
"""),

    new_code_cell("""# Create sufficiency flags
df['product_history_sufficient'] = df['product_previous_count'] >= 3
df['vendor_history_sufficient'] = df['vendor_previous_count'] >= 3

# Calculate derived features ONLY where history is sufficient
# Compare current price vs product median
df['price_vs_product_median_pct'] = np.where(
    df['product_history_sufficient'] & df['product_historical_median'].notna() & (df['product_historical_median'] != 0),
    ((df['price_eur'] - df['product_historical_median']) / df['product_historical_median']) * 100,
    np.nan
)

# Compare current price vs vendor median
df['price_vs_vendor_median_pct'] = np.where(
    df['vendor_history_sufficient'] & df['vendor_historical_median'].notna() & (df['vendor_historical_median'] != 0),
    ((df['price_eur'] - df['vendor_historical_median']) / df['vendor_historical_median']) * 100,
    np.nan
)

# Discount from product median
df['discount_from_product_median_pct'] = np.where(
    df['product_history_sufficient'] & df['product_historical_median'].notna() & (df['product_historical_median'] != 0),
    ((df['product_historical_median'] - df['price_eur']) / df['product_historical_median']) * 100,
    np.nan
)

# Discount from product max
df['discount_from_product_max_pct'] = np.where(
    df['product_history_sufficient'] & df['product_historical_max'].notna() & (df['product_historical_max'] != 0),
    ((df['product_historical_max'] - df['price_eur']) / df['product_historical_max']) * 100,
    np.nan
)

# Volatility features
df['product_price_volatility'] = np.where(df['product_history_sufficient'], df['product_historical_std'], np.nan)
df['vendor_price_volatility'] = np.where(df['vendor_history_sufficient'], df['vendor_historical_std'], np.nan)

# Coefficient of Variation (CV = std / mean)
df['historical_cv'] = np.where(
    df['product_history_sufficient'] & df['product_historical_mean'].notna() & (df['product_historical_mean'] != 0),
    df['product_historical_std'] / df['product_historical_mean'],
    np.nan
)
"""),

    new_markdown_cell("""## STEP 7 — QUALITY CHECKS & ROW COUNT VALIDATION"""),
    new_code_cell("""assert len(original_df) == len(df), f"Row count mismatch!"
print(f"Row count match successful: {len(df)} rows.")

same_day_duplicates = df.duplicated(subset=['product_id', 'vendor', 'date'], keep=False).sum()
print(f"Rows with multiple observations on the same date: {same_day_duplicates}")

print("\\nMissing values in derived features (Expected to be all 26581 due to insufficiency):")
print(df[['price_vs_product_median_pct', 'historical_cv']].isna().sum())
"""),

    new_markdown_cell("""## STEP 8 — SAVE FEATURE DATASET"""),
    new_code_cell("""output_path = '../data/fake_discount_features.csv'
df.to_csv(output_path, index=False)

print(f"Final row count: {len(df)}")
print(f"Final column count: {len(df.columns)}")
print(f"Max product_previous_count: {df['product_previous_count'].max()}")
print(f"Max vendor_previous_count: {df['vendor_previous_count'].max()}")
print(f"Number of product_history_sufficient = True: {df['product_history_sufficient'].sum()}")
print(f"Number of vendor_history_sufficient = True: {df['vendor_history_sufficient'].sum()}")
print(f"Output file path: {output_path}")
""")
])

os.makedirs(r'd:/fake-discount-detector/notebooks', exist_ok=True)
with open(r'd:/fake-discount-detector/notebooks/02_feature_engineering.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
print("Notebook generated successfully!")
