import pandas as pd
import numpy as np
import time

# Create dummy data
df = pd.DataFrame({
    'product_id': np.repeat(np.arange(1000), 50),
    'price': np.random.rand(50000)
})

start = time.time()
shifted_price = df.groupby('product_id')['price'].shift(1)
grouped_shifted = shifted_price.groupby(df['product_id'])

df['count'] = grouped_shifted.cumcount()
df['min'] = grouped_shifted.cummin()
df['max'] = grouped_shifted.cummax()

mean_s = grouped_shifted.expanding().mean()
if isinstance(mean_s.index, pd.MultiIndex):
    mean_s = mean_s.reset_index(level=0, drop=True)
df['mean'] = mean_s

std_s = grouped_shifted.expanding().std()
if isinstance(std_s.index, pd.MultiIndex):
    std_s = std_s.reset_index(level=0, drop=True)
df['std'] = std_s

median_s = grouped_shifted.expanding().median()
if isinstance(median_s.index, pd.MultiIndex):
    median_s = median_s.reset_index(level=0, drop=True)
df['median'] = median_s

print("Vectorized with expanding:", time.time() - start)

start = time.time()
gp = df.groupby('product_id')['price']
df['min2'] = gp.transform(lambda x: x.shift(1).expanding().min())
print("Lambda transform:", time.time() - start)
