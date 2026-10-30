import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("## STEP 1")
df = pd.read_csv('../data/retailrocket_price_history.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by=['product_id', 'date']).reset_index(drop=True)

original_df = df.copy()
original_row_count = len(original_df)
print(f"Original row count: {original_row_count}")

print("## STEP 2")
product_daily = original_df.groupby(['product_id', 'date'])['price'].median().reset_index()
product_daily = product_daily.sort_values(['product_id', 'date'])

shifted_price = product_daily.groupby('product_id')['price'].shift(1)
product_daily['product_previous_price'] = shifted_price
product_daily['product_previous_count'] = product_daily.groupby('product_id').cumcount()
product_daily['product_historical_min'] = shifted_price.groupby(product_daily['product_id']).cummin()
product_daily['product_historical_max'] = shifted_price.groupby(product_daily['product_id']).cummax()

cum_sum = shifted_price.groupby(product_daily['product_id']).cumsum()
product_daily['product_historical_mean'] = cum_sum / product_daily['product_previous_count']

cum_sum_sq = (shifted_price ** 2).groupby(product_daily['product_id']).cumsum()
variance = (cum_sum_sq - (cum_sum ** 2) / product_daily['product_previous_count']) / (product_daily['product_previous_count'] - 1)
product_daily['product_historical_std'] = np.sqrt(np.maximum(variance, 0))

max_obs = product_daily.groupby('product_id').size().max()
shifted_cols = [product_daily.groupby('product_id')['price'].shift(i) for i in range(1, max_obs)]
product_daily['product_historical_median'] = pd.concat(shifted_cols, axis=1).median(axis=1)

product_daily = product_daily.drop(columns=['price'])
df = original_df.merge(product_daily, on=['product_id', 'date'], how='left')

print("## STEP 4")
df['product_history_sufficient'] = (df['product_previous_count'] >= 3).astype(int)
df['vendor_history_sufficient'] = 0
print(f"Rows with sufficient product history: {df['product_history_sufficient'].sum():,}")

print("## STEP 5")
df['price_change'] = df['price'] - df['product_previous_price']
df['price_change_pct'] = np.where(
    df['product_previous_price'].isna() | (df['product_previous_price'] == 0),
    np.nan,
    (df['price_change'] / df['product_previous_price']) * 100
)

print("## STEP 6")
df['product_price_volatility'] = np.where(df['product_history_sufficient'] == 1, df['product_historical_std'], np.nan)

print("## STEP 7")
df['vendor_price_volatility'] = np.nan

print("## STEP 8")
df['price_vs_product_median_pct'] = np.where(
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
)

print("## STEP 9")
df['historical_cv'] = np.where(
    df['product_history_sufficient'].astype(bool) & df['product_historical_mean'].notna() & (df['product_historical_mean'] != 0),
    df['product_historical_std'] / df['product_historical_mean'],
    np.nan
)

print("## STEP 11")
sample_product = df.nlargest(1, 'product_previous_count').iloc[0]['product_id']
inspection_cols = [
    'date', 'price', 'product_previous_price', 'product_historical_median',
    'product_historical_max', 'product_previous_count'
]
leakage_check_df = df[df['product_id'] == sample_product][inspection_cols].head(15)
print(f"Leakage Validation for Product: {sample_product}")
print(leakage_check_df)

print("## STEP 12")
sample_df = df[df['product_id'] == sample_product].sort_values('date').reset_index(drop=True)
sample_df['unique_dates_before_D'] = sample_df['date'].map(lambda d: sample_df[sample_df['date'] < d]['date'].nunique())
assert (sample_df['product_previous_count'] == sample_df['unique_dates_before_D']).all(), "Mismatch in previous count!"
for idx in range(1, len(sample_df)):
    prev_date_price = sample_df.iloc[idx-1]['price']
    curr_prev_price = sample_df.iloc[idx]['product_previous_price']
    assert prev_date_price == curr_prev_price, "Mismatch in previous price!"
print("Automated leakage checks passed successfully!")

print("## STEP 13")
assert len(df) == original_row_count, f"Row count changed! Orig: {original_row_count}, New: {len(df)}"
print(f"Original rows: {original_row_count}")
print(f"Final rows: {len(df)}")

print("## STEP 14")
duplicated_rows = df.duplicated().sum()
print(f"Total rows: {len(df):,}")
print(f"Duplicated rows: {duplicated_rows:,}")
print(f"Unique products: {df['product_id'].nunique():,}")
print(f"Unique dates: {df['date'].nunique():,}")

print("## STEP 16")
df = df.rename(columns={'price': 'price_eur'})

print("## STEP 17")
print("========================================")
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
print("========================================")

print("## STEP 18")
import os
output_path = '../data/retailrocket_fake_discount_features.csv'
df.to_csv(output_path, index=False)
if os.path.exists(output_path):
    print(f"Successfully saved file to: {output_path}")
else:
    print("Error saving file!")
