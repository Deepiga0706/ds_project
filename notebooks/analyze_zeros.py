import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('../data/retailrocket_fake_discount_features.csv')

# 1. Number of zero-price observations
zero_prices = df[df['price_eur'] == 0]
num_zero = len(zero_prices)
print(f"1. Number of zero-price observations: {num_zero:,}")

# 2. Percentage of the complete dataset
total_rows = len(df)
pct_zero = (num_zero / total_rows) * 100
print(f"2. Percentage of complete dataset: {pct_zero:.4f}%")

# 3. Number of unique products containing zero prices
unique_zero_products = zero_prices['product_id'].unique()
print(f"3. Number of unique products containing zero prices: {len(unique_zero_products):,}")

# 4. Number of products where zero is the ONLY observed price
product_price_counts = df.groupby('product_id')['price_eur'].agg(lambda x: (x == 0).all())
only_zero_products = product_price_counts[product_price_counts].index
print(f"4. Number of products where zero is the ONLY observed price: {len(only_zero_products):,}")

# 5. Number of products where zero occurs alongside positive prices
mixed_products = set(unique_zero_products) - set(only_zero_products)
print(f"5. Number of products where zero occurs alongside positive prices: {len(mixed_products):,}")

# 6. Examples of products with zero and positive prices
print("6. Examples of products with zero and positive prices (showing up to 2):")
for pid in list(mixed_products)[:2]:
    example_df = df[df['product_id'] == pid].sort_values('date')[['product_id', 'date', 'price_eur']]
    print(f"\nProduct ID: {pid}")
    print(example_df.to_string(index=False))

# 7. Effect on historical features
print("\n--- Effect of zero prices on historical features ---")

# Let's see if historical medians are 0
zero_median = (df['product_historical_median'] == 0).sum()
print(f"- Rows with product_historical_median == 0: {zero_median:,}")

zero_max = (df['product_historical_max'] == 0).sum()
print(f"- Rows with product_historical_max == 0: {zero_max:,}")

# Check price_change_pct when previous is 0
prev_zero_rows = df[df['product_previous_price'] == 0]
null_change_pct = prev_zero_rows['price_change_pct'].isna().sum()
print(f"- Rows where product_previous_price == 0: {len(prev_zero_rows):,}")
print(f"  -> How many have NaN price_change_pct? {null_change_pct:,} (Expect all, due to div-by-zero handling)")

# Check discount_from_product_median_pct when median is 0
median_zero_rows = df[df['product_historical_median'] == 0]
null_discount = median_zero_rows['discount_from_product_median_pct'].isna().sum()
print(f"- Rows where product_historical_median == 0: {len(median_zero_rows):,}")
print(f"  -> How many have NaN discount_from_product_median_pct? {null_discount:,} (Expect all)")

# Check historical_cv when mean is 0
mean_zero_rows = df[df['product_historical_mean'] == 0]
null_cv = mean_zero_rows['historical_cv'].isna().sum()
print(f"- Rows where product_historical_mean == 0: {len(mean_zero_rows):,}")
print(f"  -> How many have NaN historical_cv? {null_cv:,}")

