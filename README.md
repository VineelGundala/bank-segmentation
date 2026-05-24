# Bank Customer Segmentation & CLV Analysis

Unsupervised machine learning project to segment 307,511 real banking customers using behavioral and financial features engineered from 5 source files. Identified 4 distinct customer segments and estimated Customer Lifetime Value per segment to support targeted marketing and retention strategy.

---

## Dataset

**Source:** [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data) — real consumer lending data from Home Credit Group.

| File | Description |
|---|---|
| application_train.csv | Main customer application data — income, demographics, credit |
| installments_payments.csv | Payment history per installment |
| credit_card_balance.csv | Monthly credit card statement data |
| POS_CASH_balance.csv | Point of sale and cash loan monthly snapshots |
| bureau.csv | External credit bureau history |

---

## Project Structure

```
bank-segmentation/
├── data/                          # Raw CSV files (not tracked in git)
├── notebooks/
│   ├── 01_eda.ipynb               # EDA and data cleaning
│   ├── 02_feature_engineering.ipynb  # RFM and multi-source feature merge
│   ├── 03_clustering.ipynb        # All unsupervised methods compared
│   └── 04_clv_profiling.ipynb     # CLV estimation and business recommendations
├── outputs/                       # Saved plots and figures
├── requirements.txt
└── README.md
```

---

## Methodology

### 1. Feature Engineering
- Merged 5 source files on customer ID
- Engineered RFM features from actual transaction history:
  - **Recency** — days since last payment
  - **Frequency** — total number of transactions
  - **Monetary** — total amount paid
- Added credit utilization ratio, payment delay features, bureau history depth
- Final feature matrix: 51 features across 307,511 customers

### 2. Preprocessing
- Log transformation of skewed features (CC balance, total paid, frequency)
- Outlier clipping at 99th percentile
- RobustScaler for final normalization
- PCA with 10 components explaining 94.5% variance

### 3. Clustering Methods Compared

| Method | Role | Result |
|---|---|---|
| KMeans K=4 | Primary segmentation model | Silhouette 0.64 |
| Gaussian Mixture Models | Soft assignment validation | 98.9% confidence |
| Hierarchical Clustering | Structure validation via dendrogram | ARI 0.95 vs KMeans |
| DBSCAN | Evaluated and rejected | 58.8% noise — unsuitable for dense customer data |

### 4. Evaluation Metrics

| Metric | Score | Interpretation |
|---|---|---|
| Silhouette Score | 0.64 | Well separated clusters |
| Davies Bouldin Index | 0.53 | Strong cluster compactness |
| Calinski Harabasz Score | 466,560 | High between vs within cluster variance |
| ARI KMeans vs GMM | 0.86 | Strong method agreement |
| ARI KMeans vs Hierarchical | 0.95 | Near perfect method agreement |
| GMM Confidence | 98.9% | Almost certain cluster assignments |

---

## Results

### Customer Segments

| Segment | Customers | Key Characteristics |
|---|---|---|
| Mass Market Mainstream | 222,714 | Moderate income, average transaction activity |
| Active High Value | 58,394 | High frequency, high CC balance, most active |
| Dormant / New Customers | 15,884 | Zero transaction activity, unknown risk |
| Disciplined Transactors | 10,519 | Highest payment discipline, lowest late payments |

### Business Recommendations

**Active High Value** — Retention priority. Loyalty rewards, premium credit card upgrades, relationship manager assignment. Cross-sell wealth management and insurance products.

**Disciplined Transactors** — Upsell priority. Offer higher credit limits and premium loan products at preferential rates. Most reliable payers in the portfolio.

**Mass Market Mainstream** — Engagement priority. Digital campaigns, auto-debit enrollment, savings product bundling. Monitor for early delinquency signals.

**Dormant / New Customers** — Activation priority. Onboarding campaigns, first transaction incentives. Starter credit cards and small personal loans to build history.

---

## Key Learnings

- Feature distribution quality matters more than algorithm choice — switching from StandardScaler to log transformation + RobustScaler improved silhouette from 0.11 to 0.64
- DBSCAN is unsuitable for dense banking customer data — produced 58.8% noise points
- High method agreement (ARI 0.95) confirms cluster structure is stable and not an artifact of any single algorithm
- Real customer segments overlap naturally — silhouette scores below 0.3 are typical for raw banking data without proper preprocessing

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/VineelGundala/bank-segmentation.git
cd bank-segmentation

# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows Git Bash

# Install dependencies
pip install -r requirements.txt

# Download data from Kaggle
# Place CSV files in data/ folder

# Run notebooks in order
# 01_eda.ipynb → 02_feature_engineering.ipynb → 03_clustering.ipynb → 04_clv_profiling.ipynb
```

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
jupyter
ipykernel
```

---

## Author

Vineel Gundala
