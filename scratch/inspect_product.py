import pandas as pd

# Load chunk or specific product
chunks = []
for chunk in pd.read_csv('data/retailrocket_ml_anomaly_results.csv', chunksize=100000):
    match = chunk[chunk['product_id'] == 650]
    if len(match) > 0:
        chunks.append(match)

if chunks:
    df_p = pd.concat(chunks).sort_values('date')
    print("Product 650 details:")
    print(df_p[['date', 'price_eur', 'product_historical_median', 'product_historical_max', 'fake_discount_score', 'discount_status', 'ml_anomaly_status', 'explanation']])
