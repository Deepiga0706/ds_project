# Fake Discount Detector

An end-to-end e-commerce data science and machine learning pipeline designed to detect deceptive pricing practices, artificial pre-sale price inflation, and phantom discounts across multi-vendor and historical retail datasets.

---

## 📌 Overview

Retailers frequently advertise major promotional discounts that may be deceptive:
1. **Artificial Inflation:** Raising prices shortly before a promotional campaign and then dropping them back down to "regular" levels while advertising a large percentage discount.
2. **Phantom Discounts:** Using exaggerated historical reference prices (or fictitious MSRPs) rather than genuine past selling prices.
3. **Cross-Vendor Discrepancies:** Divergent price trajectories that artificially inflate discount claims compared to market-wide trends.

The **Fake Discount Detector** system provides an automated, dual-engine approach combining:
- **Rule-Based Heuristic Scorer:** Deterministic business logic based on expanding historical medians, pre-promotion price spikes, and volatility thresholds.
- **Unsupervised Machine Learning Engine:** An Isolation Forest anomaly detector trained on leakage-free price series features to flag statistically abnormal discount behavior without relying on labeled ground-truth data.

---

## 🏗️ Architecture & Methodology

```
+-------------------------------------------------------+
|                 Raw E-Commerce Data                   |
|   (Retail Rocket dataset / Product Price Histories)   |
+---------------------------+---------------------------+
                            |
                            v
+-------------------------------------------------------+
|        Phase 1: Ingestion & Validation                |
|  - Data cleaning & timestamp parsing                  |
|  - Product-level & vendor-level sorting               |
+---------------------------+---------------------------+
                            |
                            v
+-------------------------------------------------------+
|        Phase 2: Leakage-Free Feature Engineering      |
|  - Expanding min, max, mean, median (shifted by 1)    |
|  - Volatility & Coefficient of Variation (CV)         |
|  - Price delta percentages & discount depth           |
+---------------------------+---------------------------+
                            |
             +--------------+--------------+
             |                             |
             v                             v
+---------------------------+ +---------------------------+
|  Phase 3: Rule-Based      | | Phase 4: Machine Learning |
|  Heuristic Detector       | | Anomaly Detector          |
|  - Pre-sale spike check   | | - Isolation Forest model  |
|  - Reference price checks | | - Standardized features   |
|  - Heuristic score (0-100)| | - Unsupervised anomaly    |
+-------------+-------------+ +-------------+-------------+
              |                             |
              +--------------+--------------+
                             |
                             v
+-------------------------------------------------------+
|        Phase 5: Comparative Evaluation & Reporting    |
|  - Cross-validation between ML & rule-based flags     |
|  - Publication-ready 300 DPI report visualizations   |
+-------------------------------------------------------+
```

---

## 📁 Repository Structure

```
fake-discount-detector/
├── data/                               # Sample datasets & processed summary outputs
│   ├── dishwashers_historical_clean.csv
│   ├── dishwashers_historical_features.csv
│   ├── electronics_dataset.csv
│   ├── fake_discount_features.csv
│   ├── product_discount_summary.csv
│   ├── repeated_suspicious_products.csv
│   ├── ml_top_anomalies.csv
│   └── top_50_suspicious_observations.csv
├── notebooks/                          # Jupyter Notebooks detailing research & experiments
│   ├── 01_data_analysis.ipynb          # Exploratory data analysis on price series
│   ├── 02_feature_engineering.ipynb    # Feature engineering pipeline
│   ├── 03_fake_discount_detection.ipynb# Heuristic rule-based detection logic
│   ├── 03_retailrocket_data_validation.ipynb # Validation against Retail Rocket dataset
│   ├── 04_feature_engineering.ipynb    # Scaled feature creation
│   ├── 04_evaluation.ipynb             # Performance & sanity evaluation
│   └── 05_ml_anomaly_detection.ipynb   # Isolation Forest unsupervised learning
├── report_figures/                     # Publication-ready figures (300 DPI)
│   ├── fig1_dataset_summary.png
│   ├── fig2_rule_based_results.png
│   ├── fig3_ml_anomaly_distribution.png
│   ├── fig4_fake_discount_score_distribution.png
│   ├── fig5_ml_anomaly_score_distribution.png
│   ├── fig6_detector_comparison.png
│   ├── fig7_product_price_history.png
│   └── fig8_system_architecture.png
├── scripts/                            # Pipeline execution & report generator scripts
│   └── generate_report_figures.py      # Standalone 300 DPI figure generation script
├── scratch/                            # Diagnostic & prototyping scripts
├── .gitignore                          # Git ignore rules
├── requirements.txt                    # Python package dependencies
└── README.md                           # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- `pip` or virtual environment manager

### 2. Installation
Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/Deepiga0706/ds_project.git
cd ds_project

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Dataset Setup
- Small sample datasets (e.g. `dishwashers_historical_clean.csv`, `electronics_dataset.csv`) are included in `data/`.
- For the full large-scale Retail Rocket dataset (~1.7M interactions), download the CSV files from Kaggle:
  [Retail Rocket E-Commerce Dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) and place the files into `retail-rocketery/`.

### 4. Running the Notebooks
Launch Jupyter Lab or Notebook:
```bash
jupyter notebook notebooks/
```

### 5. Generating Publication Figures
To regenerate the high-resolution report figures in `report_figures/`:
```bash
python scripts/generate_report_figures.py
```

---

## 📊 Key Highlights & Results

- **Data-Leakage Prevention:** All rolling and expanding statistics (median, min, max, standard deviation) employ strict lag shifts (`shift(1)`) so no future pricing information is present when scoring an observation.
- **Dual Verification:** Compares unsupervised machine learning anomaly scores (Isolation Forest) with domain-specific heuristic rules, identifying false promotional patterns with high agreement.
- **Explainability:** Each flagged discount comes with explicit human-interpretable reasoning (e.g., historical median comparison, pre-discount price spike duration).
