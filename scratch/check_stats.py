import pandas as pd
import numpy as np

print('Reading comparison CSV...')
df = pd.read_csv('data/ml_rule_comparison.csv')
print(f'Total rows: {len(df):,}')
print(f'Unique products: {df["product_id"].nunique():,}')
print(f'Min date: {df["date"].min()}, Max date: {df["date"].max()}')
print(f'Unique dates: {df["date"].nunique():,}')

print('\nDiscount Status counts:')
status_counts = df['discount_status'].value_counts()
print(status_counts)
print(status_counts / len(df) * 100)

print('\nML Anomaly Status counts:')
ml_counts = df['ml_anomaly_status'].value_counts()
print(ml_counts)
print(ml_counts / len(df) * 100)

rule_suspicious = df['discount_status'].isin(['Potentially Suspicious', 'Highly Suspicious'])
ml_suspicious = (df['ml_anomaly_status'] == 'ML Anomaly')

both = int((rule_suspicious & ml_suspicious).sum())
rule_only = int((rule_suspicious & ~ml_suspicious).sum())
ml_only = int((~rule_suspicious & ml_suspicious).sum())
neither = int((~rule_suspicious & ~ml_suspicious).sum())
union = both + rule_only + ml_only
jaccard = (both / union) * 100 if union > 0 else 0
agreement = (rule_suspicious == ml_suspicious).mean() * 100

print('\nComparison categories:')
print(f'Both Detected: {both:,} ({both/len(df)*100:.4f}%)')
print(f'Rule-Based Only: {rule_only:,} ({rule_only/len(df)*100:.4f}%)')
print(f'ML Only: {ml_only:,} ({ml_only/len(df)*100:.4f}%)')
print(f'Neither Detected: {neither:,} ({neither/len(df)*100:.4f}%)')
print(f'Total: {both + rule_only + ml_only + neither:,}')
print(f'Agreement rate: {agreement:.2f}%')
print(f'Jaccard similarity: {jaccard:.2f}%')

print('\nScore summaries:')
print('fake_discount_score min, max, mean, median:')
print(df['fake_discount_score'].min(), df['fake_discount_score'].max(), df['fake_discount_score'].mean(), df['fake_discount_score'].median())
print('ml_anomaly_score min, max, mean, median:')
print(df['ml_anomaly_score'].min(), df['ml_anomaly_score'].max(), df['ml_anomaly_score'].mean(), df['ml_anomaly_score'].median())
