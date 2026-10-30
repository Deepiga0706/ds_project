import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

df_raw = pd.read_csv('../data/retailrocket_fake_discount_features.csv')
df = df_raw[df_raw['price_eur'] > 0].copy()

sig_A = np.clip(df['discount_from_product_median_pct'].fillna(0), 0, 100) / 100.0
sig_B = np.clip(df['discount_from_product_max_pct'].fillna(0), 0, 100) / 100.0
sig_C = np.clip(-df['price_change_pct'].fillna(0), 0, 100) / 100.0
sig_D = 1.0 - np.clip(df['historical_cv'].fillna(0.25), 0, 0.5) / 0.5
raw_score = (0.4 * sig_A + 0.2 * sig_B + 0.2 * sig_C + 0.2 * sig_D) * 100

df['fake_discount_score'] = np.where(df['product_history_sufficient'] == 1, raw_score, raw_score * 0.2)
df['fake_discount_score'] = df['fake_discount_score'].fillna(0).round(2)

conditions = [
    df['product_previous_count'] < 3,
    (df['product_previous_count'] >= 3) & (df['product_previous_count'] <= 5),
    df['product_previous_count'] > 5
]
df['detection_confidence'] = np.select(conditions, ['Low', 'Medium', 'High'], default='Low')

conditions = [
    df['product_history_sufficient'] == 0,
    df['fake_discount_score'] >= 60,
    df['fake_discount_score'] >= 40
]
df['discount_status'] = np.select(conditions, ['Insufficient History', 'Highly Suspicious', 'Potentially Suspicious'], default='Normal')

df['explanation'] = 'N/A'
mask = df['discount_status'].isin(['Potentially Suspicious', 'Highly Suspicious'])

med_str = np.where(df['discount_from_product_median_pct'] > 20, 
                   df['discount_from_product_median_pct'].round().astype(str) + "% below median", "")
drop_str = np.where(df['price_change_pct'] < -20, 
                    "sudden drop of " + (-df['price_change_pct']).round().astype(str) + "%", "")
stab_str = np.where(df['historical_cv'] < 0.1, 
                    "highly stable historical price", "")

def combine_strings(s1, s2, s3):
    parts = [s for s in [s1, s2, s3] if s != ""]
    if not parts:
        return "Suspicious combination of weak signals"
    return " and ".join(parts)

if mask.sum() > 0:
    explanations = [combine_strings(m, d, s) for m, d, s in zip(med_str[mask], drop_str[mask], stab_str[mask])]
    df.loc[mask, 'explanation'] = explanations

product_summary = df.groupby('product_id').agg(
    num_observations=('date', 'count'),
    suspicious_observations=('discount_status', lambda x: x.isin(['Potentially Suspicious', 'Highly Suspicious']).sum()),
    max_fake_discount_score=('fake_discount_score', 'max'),
    avg_fake_discount_score=('fake_discount_score', 'mean'),
    historical_median_price=('price_eur', 'median'),
    historical_max_price=('price_eur', 'max'),
    historical_min_price=('price_eur', 'min')
).reset_index()

df.to_csv('../data/fake_discount_detection_results.csv', index=False)
product_summary.to_csv('../data/product_discount_summary.csv', index=False)

print("========================================")
print("FAKE DISCOUNT DETECTION REPORT")
print("========================================")
print(f"Total observations: {len(df):,}")
print(f"Unique products: {df['product_id'].nunique():,}")
print("")
for status in ['Normal', 'Potentially Suspicious', 'Highly Suspicious', 'Insufficient History']:
    count = (df['discount_status'] == status).sum()
    pct = (count / len(df)) * 100
    print(f"{status} observations: {count:,} ({pct:.2f}%)")
print("")
print(f"Average Fake Discount Score: {df['fake_discount_score'].mean():.2f}")
print(f"Maximum Fake Discount Score: {df['fake_discount_score'].max():.2f}")
print("")
multi_suspicious = (product_summary['suspicious_observations'] > 1).sum()
print(f"Products with repeated suspicious observations: {multi_suspicious:,}")
print("========================================")
