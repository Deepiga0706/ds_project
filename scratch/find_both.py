import pandas as pd

df = pd.read_csv('data/ml_rule_comparison.csv')
rule_susp = df['discount_status'].isin(['Potentially Suspicious', 'Highly Suspicious'])
ml_susp = (df['ml_anomaly_status'] == 'ML Anomaly')

both_df = df[rule_susp & ml_susp]
print("Both detected count:", len(both_df))
print("Top products in both:")
top_both = both_df['product_id'].value_counts().head(10)
print(top_both)
