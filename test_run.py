import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('../data/dishwashers_historical_clean.csv')

# Convert date to datetime
df['date'] = pd.to_datetime(df['date'])

# Sort data by product_id, vendor, and date
df = df.sort_values(by=['product_id', 'vendor', 'date']).reset_index(drop=True)

df.head()

# To ensure strictly chronological product-level history across all vendors, 
# we temporarily sort by product_id and date
df = df.sort_values(['product_id', 'date']).reset_index(drop=True)

# Function to calculate expanding stats with a shift to avoid leakage
def calc_expanding_stat(group, stat):
    shifted = group.shift(1)
    if stat == 'count':
        # Expanding count of shifted gives number of previous observations
        return shifted.expanding().count()
    elif stat == 'min':
        return shifted.expanding().min()
    elif stat == 'max':
        return shifted.expanding().max()
    elif stat == 'mean':
        return shifted.expanding().mean()
    elif stat == 'median':
        return shifted.expanding().median()
    elif stat == 'std':
        return shifted.expanding().std()

# Apply to group
grouped = df.groupby('product_id')['price_eur']

df['product_previous_count'] = grouped.transform(lambda x: calc_expanding_stat(x, 'count'))
df['product_historical_min'] = grouped.transform(lambda x: calc_expanding_stat(x, 'min'))
df['product_historical_max'] = grouped.transform(lambda x: calc_expanding_stat(x, 'max'))
df['product_historical_mean'] = grouped.transform(lambda x: calc_expanding_stat(x, 'mean'))
df['product_historical_median'] = grouped.transform(lambda x: calc_expanding_stat(x, 'median'))
df['product_historical_std'] = grouped.transform(lambda x: calc_expanding_stat(x, 'std'))

df[['product_id', 'date', 'price_eur', 'product_previous_count', 'product_historical_median']].head(10)

# Sort back by product_id, vendor, and date for vendor-specific features
df = df.sort_values(['product_id', 'vendor', 'date']).reset_index(drop=True)

grouped_vendor = df.groupby(['product_id', 'vendor'])['price_eur']

df['vendor_previous_count'] = grouped_vendor.transform(lambda x: calc_expanding_stat(x, 'count'))
df['vendor_historical_min'] = grouped_vendor.transform(lambda x: calc_expanding_stat(x, 'min'))
df['vendor_historical_max'] = grouped_vendor.transform(lambda x: calc_expanding_stat(x, 'max'))
df['vendor_historical_mean'] = grouped_vendor.transform(lambda x: calc_expanding_stat(x, 'mean'))
df['vendor_historical_median'] = grouped_vendor.transform(lambda x: calc_expanding_stat(x, 'median'))
df['vendor_historical_std'] = grouped_vendor.transform(lambda x: calc_expanding_stat(x, 'std'))

df[['product_id', 'vendor', 'date', 'price_eur', 'vendor_previous_count', 'vendor_historical_median']].head(10)

# Previous price is simply the shifted price within the vendor group
df['previous_price'] = grouped_vendor.shift(1)

# Price change
df['price_change'] = df['price_eur'] - df['previous_price']

# Price change percentage (handle division by zero safely)
# We use np.where to avoid division by zero if previous_price is 0
df['price_change_pct'] = np.where(
    df['previous_price'] == 0, 
    np.nan, 
    (df['price_change'] / df['previous_price']) * 100
)

df[['price_eur', 'previous_price', 'price_change', 'price_change_pct']].head()

# Compare current price vs product median
df['price_vs_product_median_pct'] = np.where(
    df['product_historical_median'].isna() | (df['product_historical_median'] == 0),
    np.nan,
    ((df['price_eur'] - df['product_historical_median']) / df['product_historical_median']) * 100
)

# Compare current price vs vendor median
df['price_vs_vendor_median_pct'] = np.where(
    df['vendor_historical_median'].isna() | (df['vendor_historical_median'] == 0),
    np.nan,
    ((df['price_eur'] - df['vendor_historical_median']) / df['vendor_historical_median']) * 100
)

# Discount from product median
df['discount_from_product_median_pct'] = np.where(
    df['product_historical_median'].isna() | (df['product_historical_median'] == 0),
    np.nan,
    ((df['product_historical_median'] - df['price_eur']) / df['product_historical_median']) * 100
)

# Discount from product max
df['discount_from_product_max_pct'] = np.where(
    df['product_historical_max'].isna() | (df['product_historical_max'] == 0),
    np.nan,
    ((df['product_historical_max'] - df['price_eur']) / df['product_historical_max']) * 100
)

df[['price_eur', 'product_historical_median', 'price_vs_product_median_pct', 'discount_from_product_median_pct']].head()

# Product and Vendor Volatility
df['product_price_volatility'] = df['product_historical_std']
df['vendor_price_volatility'] = df['vendor_historical_std']

# Coefficient of Variation (CV = std / mean)
df['historical_cv'] = np.where(
    df['product_historical_mean'].isna() | (df['product_historical_mean'] == 0),
    np.nan,
    df['product_historical_std'] / df['product_historical_mean']
)

df[['product_historical_std', 'product_historical_mean', 'historical_cv']].head()

# We define sufficient history as having at least 3 previous observations at the vendor level.
# This ensures we have a somewhat reliable median and standard deviation.
MIN_OBSERVATIONS = 3

df['history_sufficient'] = df['vendor_previous_count'] >= MIN_OBSERVATIONS

print(f"Observations with sufficient history: {df['history_sufficient'].sum()}")
print(f"Observations without sufficient history: {(~df['history_sufficient']).sum()}")

print("1. Number of rows:", len(df))
print("2. Duplicate rows:", df.duplicated().sum())
print("3. Missing values (selected columns):")
print(df[['price_change_pct', 'product_historical_median', 'historical_cv']].isna().sum())

# 4. Check for infinite values
numeric_cols = df.select_dtypes(include=[np.number]).columns
inf_counts = np.isinf(df[numeric_cols]).sum()
print("\n4. Infinite values:\n", inf_counts[inf_counts > 0])

print("\n5. Date range:", df['date'].min(), "to", df['date'].max())
print("6. Unique products:", df['product_id'].nunique())
print("7. Unique vendors:", df['vendor'].nunique())
print("8. Sufficient history:", df['history_sufficient'].sum())
print("9. Insufficient history:", (~df['history_sufficient']).sum())

# 10. Assertions to confirm no future prices are used
# Check that the historical min is never greater than the current price IF current price was included
# Actually, since it doesn't include current price, the current price CAN be less than historical min.
# The main check is that for the first chronological observation of a vendor/product, historical features are NaN.

first_obs = df.groupby(['product_id', 'vendor']).head(1)
assert first_obs['vendor_historical_median'].isna().all(), "First observations should not have a historical median!"
assert first_obs['previous_price'].isna().all(), "First observations should not have a previous price!"

print("\nAll quality checks passed.")

# Pick a random product and vendor with multiple observations
sample_group = df[df['vendor_previous_count'] > 4].iloc[0]
sample_product = sample_group['product_id']
sample_vendor = sample_group['vendor']

inspection_cols = ['date', 'price_eur', 'previous_price', 'product_historical_median', 'vendor_historical_median']
leakage_check_df = df[(df['product_id'] == sample_product) & (df['vendor'] == sample_vendor)][inspection_cols].head(8)

print(f"Leakage Validation for Product: {sample_product}, Vendor: {sample_vendor}")
display(leakage_check_df)

# Notice how the medians strictly reflect the values from previous rows, and never use the current row's price.

# 1. Distribution of price_change_pct
plt.figure(figsize=(10, 5))
sns.histplot(df['price_change_pct'].dropna(), bins=50, kde=True)
plt.title('Distribution of Price Change Percentage (Current vs Immediate Previous)')
plt.xlabel('Price Change %')
plt.ylabel('Count')
plt.xlim(-100, 100)
plt.show()

# 2. Distribution of discount_from_product_median_pct
plt.figure(figsize=(10, 5))
sns.histplot(df['discount_from_product_median_pct'].dropna(), bins=50, kde=True, color='green')
plt.title('Distribution of Discount from Product Historical Median %')
plt.xlabel('Discount from Product Median %')
plt.ylabel('Count')
plt.xlim(-100, 100)
plt.show()

# 3. Distribution of historical volatility (CV)
plt.figure(figsize=(10, 5))
sns.histplot(df['historical_cv'].dropna(), bins=50, kde=True, color='orange')
plt.title('Distribution of Historical Coefficient of Variation (Volatility)')
plt.xlabel('Coefficient of Variation')
plt.ylabel('Count')
plt.xlim(0, 1)
plt.show()

# 4. Historical price trend for one product/vendor
plt.figure(figsize=(12, 6))
sample_data = df[(df['product_id'] == sample_product) & (df['vendor'] == sample_vendor)]

plt.plot(sample_data['date'], sample_data['price_eur'], marker='o', label='Current Price', color='blue')
plt.plot(sample_data['date'], sample_data['vendor_historical_median'], linestyle='--', label='Vendor Historical Median', color='red')

plt.title(f'Price History and Baseline for {sample_product} at {sample_vendor}')
plt.xlabel('Date')
plt.ylabel('Price (EUR)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

output_path = '../data/fake_discount_features.csv'
df.to_csv(output_path, index=False)

print(f"Final row count: {len(df)}")
print(f"Final column count: {len(df.columns)}")
print(f"Output file path: {output_path}")

print("\nFirst 5 rows:")
display(df.head())
