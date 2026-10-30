import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []

# Header
cells.append(nbf.v4.new_markdown_cell("# Fake Discount Detector: Feature Engineering\n\nFollowing the 18-step methodology."))

# STEP 1
cells.append(nbf.v4.new_markdown_cell("## STEP 1 — PRESERVE ORIGINAL DATA"))
cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load the cleaned Retail Rocket price history
df = pd.read_csv('../data/retailrocket_price_history.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by=['product_id', 'date']).reset_index(drop=True)

original_df = df.copy()
original_row_count = len(original_df)

print(f"Original row count: {original_row_count}")"""))

# STEP 2
cells.append(nbf.v4.new_markdown_cell("## STEP 2 — PRODUCT HISTORICAL FEATURES"))
cells.append(nbf.v4.new_code_cell("""# 1. Daily aggregation to establish a stable timeline
product_daily = original_df.groupby(['product_id', 'date'])['price'].median().reset_index()
product_daily = product_daily.sort_values(['product_id', 'date'])

# 2. Shift price to strictly avoid current-day leakage
shifted_price = product_daily.groupby('product_id')['price'].shift(1)
product_daily['product_previous_price'] = shifted_price

# 3. Calculate previous count (0-indexed cumcount on sorted data is exactly the previous count)
product_daily['product_previous_count'] = product_daily.groupby('product_id').cumcount()

# 4. Calculate historical min and max using cumulative operations
product_daily['product_historical_min'] = shifted_price.groupby(product_daily['product_id']).cummin()
product_daily['product_historical_max'] = shifted_price.groupby(product_daily['product_id']).cummax()

# 5. Calculate historical mean and std
cum_sum = shifted_price.groupby(product_daily['product_id']).cumsum()
product_daily['product_historical_mean'] = cum_sum / product_daily['product_previous_count']

cum_sum_sq = (shifted_price ** 2).groupby(product_daily['product_id']).cumsum()
variance = (cum_sum_sq - (cum_sum ** 2) / product_daily['product_previous_count']) / (product_daily['product_previous_count'] - 1)
product_daily['product_historical_std'] = np.sqrt(np.maximum(variance, 0))

# 6. Calculate historical median (optimized vectorized approach)
max_obs = product_daily.groupby('product_id').size().max()
shifted_cols = [product_daily.groupby('product_id')['price'].shift(i) for i in range(1, max_obs)]
product_daily['product_historical_median'] = pd.concat(shifted_cols, axis=1).median(axis=1)

# Drop aggregated 'price' to prevent conflict
product_daily = product_daily.drop(columns=['price'])

# Merge back to observation level
df = original_df.merge(product_daily, on=['product_id', 'date'], how='left')"""))

# STEP 3 - Skipped
cells.append(nbf.v4.new_markdown_cell("## STEP 3 — VENDOR HISTORICAL FEATURES\n*Skipped as dataset does not contain vendor information.*"))

# STEP 4
cells.append(nbf.v4.new_markdown_cell("## STEP 4 — HISTORY SUFFICIENCY"))
cells.append(nbf.v4.new_code_cell("""df['product_history_sufficient'] = (df['product_previous_count'] >= 3).astype(int)
df['vendor_history_sufficient'] = 0 # N/A

print(f"Rows with sufficient product history: {df['product_history_sufficient'].sum():,}")"""))

# STEP 5
cells.append(nbf.v4.new_markdown_cell("## STEP 5 — PRICE CHANGE FEATURES"))
cells.append(nbf.v4.new_code_cell("""df['price_change'] = df['price'] - df['product_previous_price']
df['price_change_pct'] = np.where(
    df['product_previous_price'].isna() | (df['product_previous_price'] == 0),
    np.nan,
    (df['price_change'] / df['product_previous_price']) * 100
)"""))

# STEP 6
cells.append(nbf.v4.new_markdown_cell("## STEP 6 — PRODUCT PRICE VOLATILITY"))
cells.append(nbf.v4.new_code_cell("""df['product_price_volatility'] = np.where(df['product_history_sufficient'] == 1, df['product_historical_std'], np.nan)"""))

# STEP 7 - Skipped
cells.append(nbf.v4.new_markdown_cell("## STEP 7 — VENDOR PRICE VOLATILITY\n*Skipped as dataset does not contain vendor information.*"))
cells.append(nbf.v4.new_code_cell("""df['vendor_price_volatility'] = np.nan"""))

# STEP 8
cells.append(nbf.v4.new_markdown_cell("## STEP 8 — DISCOUNT FEATURES"))
cells.append(nbf.v4.new_code_cell("""df['price_vs_product_median_pct'] = np.where(
    df['product_historical_median'].isna() | (df['product_historical_median'] == 0),
    np.nan,
    ((df['price'] - df['product_historical_median']) / df['product_historical_median']) * 100
)
df['price_vs_vendor_median_pct'] = np.nan

df['discount_from_product_median_pct'] = np.where(
    df['product_historical_median'].isna() | (df['product_historical_median'] == 0),
    np.nan,
    ((df['product_historical_median'] - df['price']) / df['product_historical_median']) * 100
)

df['discount_from_product_max_pct'] = np.where(
    df['product_historical_max'].isna() | (df['product_historical_max'] == 0),
    np.nan,
    ((df['product_historical_max'] - df['price']) / df['product_historical_max']) * 100
)"""))

# STEP 9
cells.append(nbf.v4.new_markdown_cell("## STEP 9 — HISTORICAL CV"))
cells.append(nbf.v4.new_code_cell("""df['historical_cv'] = np.where(
    df['product_history_sufficient'].astype(bool) & df['product_historical_mean'].notna() & (df['product_historical_mean'] != 0),
    df['product_historical_std'] / df['product_historical_mean'],
    np.nan
)"""))

# STEP 10
cells.append(nbf.v4.new_markdown_cell("## STEP 10 — PERFORMANCE REQUIREMENT\n*All transformations utilize fully vectorized operations natively supported by pandas (groupby, shift, cumcount/min/max, concat).*"))

# STEP 11
cells.append(nbf.v4.new_markdown_cell("## STEP 11 — LEAKAGE VALIDATION"))
cells.append(nbf.v4.new_code_cell("""# Select a product with many historical observations
sample_product = df.nlargest(1, 'product_previous_count').iloc[0]['product_id']

inspection_cols = [
    'date', 'price', 'product_previous_price', 'product_historical_median',
    'product_historical_max', 'product_previous_count'
]
leakage_check_df = df[df['product_id'] == sample_product][inspection_cols].head(15)

print(f"Leakage Validation for Product: {sample_product}")
display(leakage_check_df)"""))

# STEP 12
cells.append(nbf.v4.new_markdown_cell("## STEP 12 — AUTOMATED LEAKAGE CHECK"))
cells.append(nbf.v4.new_code_cell("""# For the sample product, run programmatic checks
sample_df = df[df['product_id'] == sample_product].sort_values('date').reset_index(drop=True)
sample_df['unique_dates_before_D'] = sample_df['date'].map(lambda d: sample_df[sample_df['date'] < d]['date'].nunique())

assert (sample_df['product_previous_count'] == sample_df['unique_dates_before_D']).all(), "Mismatch in previous count!"

# Verify product_previous_price
for idx in range(1, len(sample_df)):
    prev_date_price = sample_df.iloc[idx-1]['price']
    curr_prev_price = sample_df.iloc[idx]['product_previous_price']
    assert prev_date_price == curr_prev_price, "Mismatch in previous price!"

print("Automated leakage checks passed successfully!")"""))

# STEP 13
cells.append(nbf.v4.new_markdown_cell("## STEP 13 — ROW COUNT VALIDATION"))
cells.append(nbf.v4.new_code_cell("""assert len(df) == original_row_count, f"Row count changed! Orig: {original_row_count}, New: {len(df)}"
print(f"Original rows: {original_row_count}")
print(f"Final rows: {len(df)}")"""))

# STEP 14
cells.append(nbf.v4.new_markdown_cell("## STEP 14 — DUPLICATE VALIDATION"))
cells.append(nbf.v4.new_code_cell("""duplicated_rows = df.duplicated().sum()
print(f"Total rows: {len(df):,}")
print(f"Duplicated rows: {duplicated_rows:,}")
print(f"Unique products: {df['product_id'].nunique():,}")
print(f"Unique dates: {df['date'].nunique():,}")"""))

# STEP 15
cells.append(nbf.v4.new_markdown_cell("## STEP 15 — FEATURE QUALITY REPORT"))
cells.append(nbf.v4.new_code_cell("""def feature_quality_report(df):
    report = pd.DataFrame({
        'dtype': df.dtypes,
        'non_nulls': df.notnull().sum(),
        'nulls': df.isnull().sum(),
        'null_pct': (df.isnull().sum() / len(df)) * 100
    })
    
    # Infinite values
    inf_counts = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            inf_counts.append(np.isinf(df[col]).sum())
        else:
            inf_counts.append(0)
    report['inf_counts'] = inf_counts
    
    return report

print("Feature Quality Report:")
display(feature_quality_report(df))

print(f"Negative prices: {(df['price'] < 0).sum():,}")
print(f"Zero prices: {(df['price'] == 0).sum():,}")"""))

# STEP 16
cells.append(nbf.v4.new_markdown_cell("## STEP 16 — FINAL FEATURE TABLE"))
cells.append(nbf.v4.new_code_cell("""# Rename price to price_eur if desired, but user just said price
df = df.rename(columns={'price': 'price_eur'})

# Display first few rows of final dataset
display(df.head())"""))

# STEP 17
cells.append(nbf.v4.new_markdown_cell("## STEP 17 — FINAL VALIDATION SUMMARY"))
cells.append(nbf.v4.new_code_cell("""print("========================================")
print("FINAL FEATURE ENGINEERING REPORT")
print("========================================")
print(f"Original row count: {original_row_count}")
print(f"Final row count: {len(df)}")
print("")
print(f"Original column count: {len(original_df.columns)}")
print(f"Final column count: {len(df.columns)}")
print("")
print(f"Unique products: {df['product_id'].nunique()}")
print(f"Unique vendors: 0")
print(f"Unique dates: {df['date'].nunique()}")
print("")
print(f"Products with sufficient history: {df[df['product_history_sufficient'] == 1]['product_id'].nunique()}")
print(f"Vendors with sufficient history: 0")
print("")
print(f"Rows with product volatility: {df['product_price_volatility'].notna().sum()}")
print(f"Rows with vendor volatility: {df['vendor_price_volatility'].notna().sum()}")
print("")
print(f"Rows with discount-from-median feature: {df['discount_from_product_median_pct'].notna().sum()}")
print(f"Rows with discount-from-max feature: {df['discount_from_product_max_pct'].notna().sum()}")
print("")
print(f"Duplicate rows: {df.duplicated().sum()}")
print(f"Infinite values: {np.isinf(df.select_dtypes(include=np.number)).sum().sum()}")
print(f"Negative prices: {(df['price_eur'] < 0).sum()}")
print(f"Zero prices: {(df['price_eur'] == 0).sum()}")
print("")
print("Leakage validation: PASSED")
print("Row preservation: PASSED")
print("========================================")"""))

# STEP 18
cells.append(nbf.v4.new_markdown_cell("## STEP 18 — SAVE FINAL DATASET"))
cells.append(nbf.v4.new_code_cell("""import os

output_path = '../data/retailrocket_fake_discount_features.csv'
df.to_csv(output_path, index=False)

if os.path.exists(output_path):
    print(f"Successfully saved file to: {output_path}")
else:
    print("Error saving file!")"""))

nb.cells = cells

with open('02_feature_engineering.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Created 02_feature_engineering.ipynb")
