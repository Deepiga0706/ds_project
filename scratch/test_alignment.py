import pandas as pd
import numpy as np

df = pd.DataFrame({
    'product_id': [2, 2, 1, 1],
    'price': [10, 20, 100, 200]
})
# index is 0, 1, 2, 3

shifted_price = df.groupby('product_id')['price'].shift(1)
grouped_shifted = shifted_price.groupby(df['product_id'])

# expanding mean
mean_s = grouped_shifted.expanding().mean()
print("MultiIndex:\n", mean_s.index)

# droplevel
mean_s = mean_s.droplevel(0)
print("After droplevel:\n", mean_s.index)

df['mean'] = mean_s
print(df)
