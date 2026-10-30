import pandas as pd

# Check top anomalies and products with suspicious observations
top_anom = pd.read_csv('data/ml_top_anomalies.csv', nrows=20)
print('Top anomalies product_ids:', top_anom['product_id'].tolist()[:10])

# Let's inspect product_discount_summary
p_sum = pd.read_csv('data/product_discount_summary.csv')
print('\nTop products by suspicious_observations in product_discount_summary:')
top_susp_p = p_sum.sort_values(by=['suspicious_observations', 'num_observations'], ascending=False).head(10)
print(top_susp_p)

# Let's check a product with 10-18 observations and suspicious discounts
cands = p_sum[(p_sum['suspicious_observations'] >= 2) & (p_sum['num_observations'] >= 10)].head(10)
print('\nCandidates with >=2 suspicious and >= 10 observations:')
print(cands)
