import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []

# Header
cells.append(nbf.v4.new_markdown_cell("# Fake Discount Detector: Detection Phase\n\nThis notebook implements an interpretable anomaly-based Fake Discount Detector using vectorized operations."))

# STEP 1
cells.append(nbf.v4.new_markdown_cell("## STEP 1 — LOAD FEATURE DATA"))
cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load the feature-engineered dataset
df_raw = pd.read_csv('../data/retailrocket_fake_discount_features.csv')

# Exclude price_eur == 0
df = df_raw[df_raw['price_eur'] > 0].copy()

print(f"Total raw observations: {len(df_raw):,}")
print(f"Valid observations (price > 0): {len(df):,}")"""))

# STEP 2 & 3 & 4
cells.append(nbf.v4.new_markdown_cell("## STEP 2, 3, & 4 — DEFINE SIGNALS, NORMALIZE, AND DESIGN SCORE\n\nWe define the score based on:\n- Signal A: Discount from Median (40% weight)\n- Signal B: Discount from Maximum (20% weight)\n- Signal C: Recent Price Drop (20% weight)\n- Signal D: Price Stability (20% weight)"))
cells.append(nbf.v4.new_code_cell("""# SIGNAL A: Discount from median (normalized 0-1)
sig_A = np.clip(df['discount_from_product_median_pct'].fillna(0), 0, 100) / 100.0

# SIGNAL B: Discount from maximum (normalized 0-1)
sig_B = np.clip(df['discount_from_product_max_pct'].fillna(0), 0, 100) / 100.0

# SIGNAL C: Recent Price Drop (normalized 0-1)
# A negative price change percentage indicates a drop.
sig_C = np.clip(-df['price_change_pct'].fillna(0), 0, 100) / 100.0

# SIGNAL D: Price Stability
# historical_cv = std / mean. Lower CV means higher stability.
# We normalize so that CV=0 gives stability=1, and CV>=0.5 gives stability=0
sig_D = 1.0 - np.clip(df['historical_cv'].fillna(0.25), 0, 0.5) / 0.5

# Calculate Raw Score (0-100)
raw_score = (0.4 * sig_A + 0.2 * sig_B + 0.2 * sig_C + 0.2 * sig_D) * 100

# Apply history sufficiency penalty
# If insufficient history, score is severely penalized to prevent false positives on new products
df['fake_discount_score'] = np.where(df['product_history_sufficient'] == 1, raw_score, raw_score * 0.2)
df['fake_discount_score'] = df['fake_discount_score'].fillna(0).round(2)
"""))

# STEP 5
cells.append(nbf.v4.new_markdown_cell("## STEP 5 — CONFIDENCE"))
cells.append(nbf.v4.new_code_cell("""# Confidence based on amount of historical evidence
conditions = [
    df['product_previous_count'] < 3,
    (df['product_previous_count'] >= 3) & (df['product_previous_count'] <= 5),
    df['product_previous_count'] > 5
]
choices = ['Low', 'Medium', 'High']
df['detection_confidence'] = np.select(conditions, choices, default='Low')
"""))

# STEP 6
cells.append(nbf.v4.new_markdown_cell("## STEP 6 — DETECTION CATEGORY"))
cells.append(nbf.v4.new_code_cell("""# Categorize based on score and history sufficiency
conditions = [
    df['product_history_sufficient'] == 0,
    df['fake_discount_score'] >= 60,
    df['fake_discount_score'] >= 40
]
choices = ['Insufficient History', 'Highly Suspicious', 'Potentially Suspicious']
df['discount_status'] = np.select(conditions, choices, default='Normal')
"""))

# STEP 7
cells.append(nbf.v4.new_markdown_cell("## STEP 7 — EXPLAINABILITY"))
cells.append(nbf.v4.new_code_cell("""# Vectorized explanation generation
df['explanation'] = 'N/A'
mask = df['discount_status'].isin(['Potentially Suspicious', 'Highly Suspicious'])

# Construct explanation components
med_str = np.where(df['discount_from_product_median_pct'] > 20, 
                   df['discount_from_product_median_pct'].round().astype(str) + "% below median", "")

drop_str = np.where(df['price_change_pct'] < -20, 
                    "sudden drop of " + (-df['price_change_pct']).round().astype(str) + "%", "")

stab_str = np.where(df['historical_cv'] < 0.1, 
                    "highly stable historical price", "")

# Combine components
def combine_strings(s1, s2, s3):
    parts = [s for s in [s1, s2, s3] if s != ""]
    if not parts:
        return "Suspicious combination of weak signals"
    return " and ".join(parts)

# Apply only to suspicious rows for performance
if mask.sum() > 0:
    explanations = [combine_strings(m, d, s) for m, d, s in zip(med_str[mask], drop_str[mask], stab_str[mask])]
    df.loc[mask, 'explanation'] = explanations
"""))

# STEP 8
cells.append(nbf.v4.new_markdown_cell("## STEP 8 — VALIDATE THE DETECTOR"))
cells.append(nbf.v4.new_code_cell("""cols_to_show = [
    'product_id', 'date', 'price_eur', 'product_historical_median', 
    'product_historical_max', 'discount_from_product_median_pct', 
    'discount_from_product_max_pct', 'price_change_pct', 
    'product_price_volatility', 'product_previous_count', 
    'fake_discount_score', 'discount_status', 'detection_confidence', 'explanation'
]

top_20 = df.nlargest(20, 'fake_discount_score')[cols_to_show]
display(top_20)"""))

# STEP 9
cells.append(nbf.v4.new_markdown_cell("## STEP 9 — CHECK WHETHER RESULTS MAKE SENSE"))
cells.append(nbf.v4.new_code_cell("""status_counts = df['discount_status'].value_counts()
status_pcts = df['discount_status'].value_counts(normalize=True) * 100

dist_df = pd.DataFrame({
    'Count': status_counts,
    'Percentage': status_pcts.round(2)
})
display(dist_df)"""))

# STEP 10
cells.append(nbf.v4.new_markdown_cell("## STEP 10 — PRODUCT-LEVEL ANALYSIS"))
cells.append(nbf.v4.new_code_cell("""# Product level aggregations
product_summary = df.groupby('product_id').agg(
    num_observations=('date', 'count'),
    suspicious_observations=('discount_status', lambda x: x.isin(['Potentially Suspicious', 'Highly Suspicious']).sum()),
    max_fake_discount_score=('fake_discount_score', 'max'),
    avg_fake_discount_score=('fake_discount_score', 'mean'),
    historical_median_price=('price_eur', 'median'),
    historical_max_price=('price_eur', 'max'),
    historical_min_price=('price_eur', 'min')
).reset_index()

display(product_summary.nlargest(10, 'suspicious_observations'))"""))

# STEP 11
cells.append(nbf.v4.new_markdown_cell("## STEP 11 — VISUALIZATIONS"))
cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(10, 5))
sns.histplot(df[df['product_history_sufficient'] == 1]['fake_discount_score'], bins=50, kde=True)
plt.title('Distribution of Fake Discount Score (Sufficient History Only)')
plt.xlabel('Fake Discount Score')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='discount_status', order=['Normal', 'Potentially Suspicious', 'Highly Suspicious', 'Insufficient History'])
plt.title('Number of Observations by Discount Status')
plt.xticks(rotation=45)
plt.show()"""))

# STEP 12
cells.append(nbf.v4.new_markdown_cell("## STEP 12 — IMPORTANT LIMITATION\n\n**Limitation Statement:**\nThere is no ground-truth fake-discount label in the Retail Rocket dataset. Therefore, the Fake Discount Score does not equal proof that a discount is fraudulent. The system identifies pricing patterns that are unusually inconsistent with historical behavior.\n\nThis is an **interpretable anomaly-based Fake Discount Detection system**, not a supervised classification model."))

# STEP 13
cells.append(nbf.v4.new_markdown_cell("## STEP 13 — SAVE FINAL OUTPUT"))
cells.append(nbf.v4.new_code_cell("""df.to_csv('../data/fake_discount_detection_results.csv', index=False)
product_summary.to_csv('../data/product_discount_summary.csv', index=False)
print("Saved detection results and product summary.")"""))

# STEP 14
cells.append(nbf.v4.new_markdown_cell("## STEP 14 — FINAL REPORT"))
cells.append(nbf.v4.new_code_cell("""print("========================================")
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

print("\\nTop 10 most suspicious observations:")
display(df.nlargest(10, 'fake_discount_score')[cols_to_show])"""))


nb.cells = cells
with open('03_fake_discount_detection.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Created 03_fake_discount_detection.ipynb")
