import pandas as pd
import numpy as np
import time

df = pd.DataFrame({
    'product_id': [2, 2, 1, 1],
    'price': [10, 20, 100, 200]
})
df.index = [3, 4, 1, 2] # Arbitrary non-default index

shifted_price = df.groupby('product_id')['price'].shift(1)
grouped_shifted = shifted_price.groupby(df['product_id'])

df['prev'] = shifted_price
df['count'] = grouped_shifted.expanding().count().droplevel(0)
df['min'] = grouped_shifted.cummin()
df['max'] = grouped_shifted.cummax()
df['mean'] = grouped_shifted.expanding().mean().droplevel(0)
df['median'] = grouped_shifted.expanding().median().droplevel(0)
df['std'] = grouped_shifted.expanding().std().droplevel(0)

print(df)
