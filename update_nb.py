import json

def update_notebook():
    file_path = 'd:/fake-discount-detector/notebooks/04_feature_engineering.ipynb'
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    new_code = '''import time

print("--- Starting STEP 3 ---")
step3_start_time = time.time()

# 1. Sort product_daily to ensure correct chronological order for cumulative operations
product_daily = product_daily.sort_values(['product_id', 'date'])

# 2. Shift price within each product group
shifted_price = product_daily.groupby('product_id')['price'].shift(1)
product_daily['product_previous_price'] = shifted_price

# 3. Calculate previous count using cumcount on the sorted dataframe
# cumcount starts at 0, which matches the count of *strictly previous* observations
product_daily['product_previous_count'] = product_daily.groupby('product_id').cumcount()
# For the first observation (where previous count should be 0), we want it to be 0 instead of NaN.
# Wait, expanding().count() returns 1 when there's 1 non-NaN element.
# The original code:
# product_daily['product_previous_count'] = grouped_product.transform(lambda x: x.shift(1).expanding().count())
# This would return 0.0 or NaN for the first element, then 1.0, 2.0...
# product_daily.groupby('product_id').cumcount() gives 0, 1, 2... exactly. 

# 4. Calculate historical min and max using cumulative operations on the shifted price
product_daily['product_historical_min'] = shifted_price.groupby(product_daily['product_id']).cummin()
product_daily['product_historical_max'] = shifted_price.groupby(product_daily['product_id']).cummax()

# 5. Calculate historical mean and std using cumulative sums
cum_sum = shifted_price.groupby(product_daily['product_id']).cumsum()
product_daily['product_historical_mean'] = cum_sum / product_daily['product_previous_count']

# Calculate cumulative sum of squares for standard deviation
cum_sum_sq = (shifted_price ** 2).groupby(product_daily['product_id']).cumsum()
# Variance = (sum(x^2) - (sum(x))^2 / n) / (n - 1)
variance = (cum_sum_sq - (cum_sum ** 2) / product_daily['product_previous_count']) / (product_daily['product_previous_count'] - 1)
# Prevent negative variance due to floating point precision
product_daily['product_historical_std'] = np.sqrt(np.maximum(variance, 0))

# 6. Calculate historical median
# Using groupby.expanding.median avoids the very slow transform(lambda ...)
product_daily['product_historical_median'] = shifted_price.groupby(product_daily['product_id']).expanding().median().reset_index(level=0, drop=True)

# Drop the aggregated price so we don't conflict with the original price column
product_daily = product_daily.drop(columns=['price'])

# Merge back into the original observation-level dataframe
df = original_df.merge(product_daily, on=['product_id', 'date'], how='left')

step3_end_time = time.time()
print(f"STEP 3 completed in {step3_end_time - step3_start_time:.2f} seconds.")

# Performance diagnostics
print("Rows in product_daily:", len(product_daily))
print("Rows in original:", len(original_df))
print("Rows after merge:", len(df))

assert len(original_df) == len(df)
'''
    # Assign new lines to the source
    lines = [line + '\\n' for line in new_code.split('\\n')]
    if lines[-1] == '\\n':
        lines.pop()
    
    nb['cells'][6]['source'] = lines

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

if __name__ == '__main__':
    update_notebook()
    print("Notebook cell updated.")
