📘 Median PFS Prediction
 
This project predicts Median Progression-Free Survival (PFS) for oncology clinical trial arms using structured trial-arm–level datasets.
We combine tabular features, engineered oncology response metrics, disease & drug embeddings, and advanced ML models with strict Trial ID–based leakage prevention.
 
 
---
 
📌 Project Overview
 
The goal is to build a robust, generalizable ML model that can estimate Median PFS using clinical trial metadata and response variables.
 
We explored:
 
LightGBM (GBDT + DART)
 
CatBoost (with Bayesian Optimization)
 
XGBoost
 
Stacking models
 
Disease & Drug text embeddings (BERT)
 
Feature engineering
 
Group-aware validation (Trial ID)
 
 
 
---
 
📂 Project Structure
 
MedianPFS/
│
├── data/                             # Cleaned datasets, merged files, embeddings
│
├── src/
│   ├── preprocess/                   # Data cleaning, clustering, feature engineering
│   ├── explore/                      # EDA, scatter plots, correlation analysis, PPT generation
│   ├── new_features/
│   │   ├── product_decompose.py
│   │   └── embed/
│   │       ├── disease_embed.py
│   │       ├── drug_embed.py
│   │       ├── merge_embed_DP.py
│   │       ├── train_embed_hybrid_77.py
│   │       ├── train_nopca.py
│   │       ├── train_cv.py
│   │       └── tuning_lgbm_gbdt_optuna.py
│   ├── train/                        # All ML training scripts
│   └── tuning/                       # Optuna, BO, DART tuning
│
├── outputs/                          # SHAP, CV results, pred-vs-actual, tuning outputs
├── shap_summary.png
├── shap_bar.png
└── README.md
 
 
---
 
🧹 1. Data Cleaning & Preprocessing
 
Performed using scripts in src/preprocess/clean/.
 
✔ Key steps:
 
Standardized column names
 
Removed invalid rows
 
Converted numeric features (ORR%, DoR, Arm N, PFS)
 
Removed Median PFS > 40 months as outliers
 
Filled missing values appropriately
 
 
 
---
 
🧬 2. Engineered Features
 
Feature engineering played a major role.
We generated:
 
Feature	Description
 
ORR×DoR	ORR (%) × Duration of Response
log(ORR×DoR)	Log-normalized metric
Response_Count_Duration	Derived oncology response variable
Response_Percentage_Duration	% response × duration
Product_Category	Immunotherapy / Targeted / Chemo / Hormonal / ADC / Other
MOA Cluster	KMeans grouping of mechanisms of action
Precise Area Cluster	KMeans grouping of disease areas
 
 
These improved interpretability and downstream modelling performance.
 
 
---
 
🧬 3. Disease & Drug Text Embeddings
 
We generated transformer-based 768-dim text embeddings for:
 
✔ Disease Embeddings
 
From combined text:
 
Precise_Area_Name
 
Primary_MOA_all
 
Type
 
 
✔ Drug Embeddings
 
Extracted active drug names → embedded using the same biomedical BERT model.
 
Saved as:
 
disease_emb_0 ... disease_emb_767
 
drug_emb_0 ... drug_emb_767
 
 
✔ Merged dataset
 
Merged by Trial ID + Arm ID → Saved as:
 
MedianPFS_training_merged_with_embeddings.xlsx
 
 
---
 
⚙️ 4. PCA Compression (Optional)
 
To reduce embedding dimensions:
 
768 → 50 components
 
PCA used when needed for speed or regularization
 
Non-PCA version gave better performance, so we keep both.
 
 
 
---
 
📊 5. Exploratory Analysis
 
Scripts under src/explore/ generate:
 
ORR vs PFS plots
 
DoR vs PFS plots
 
ORR×DoR vs PFS
 
log(ORR×DoR) vs PFS
 
OS comparison plots
 
Automated PowerPoint report
 
Correlation heatmaps
 
 
These were packaged into:
 
Median_PFS_OS_Report.pptx
 
 
---
 
🤖 6. Machine Learning Models
 
We trained multiple models:
 
✔ LightGBM GBDT
 
Best baseline tabular model
→ R² = 0.74
 
✔ LightGBM + Disease/Drug Embeddings
 
Best single-split performance
→ R² = 0.79
 
✔ CatBoost (BO tuned)
 
→ R² ≈ 0.72
 
✔ XGBoost, DART, Stacking
 
Underperformed relative to LightGBM
 
 
---
 
🔍 7. Hyperparameter Tuning
 
✔ Optuna (TPE)
 
Used extensively for:
 
LightGBM tuning
 
CatBoost tuning
 
PCA vs non-PCA selection
 
 
Best LightGBM parameters:
 
learning_rate=0.0037
n_estimators=1351
num_leaves=227
max_depth=6
min_child_samples=198
subsample=0.74
colsample_bytree=0.49
reg_alpha=1.09
reg_lambda=0.22
 
✔ Bayesian Optimization for CatBoost
 
Best R² ≈ 0.72.
 
 
---
 
🧪 8. Cross-Validation (Leakage-Free)
 
Used GroupKFold with Trial ID
 
→ Ensures trial arms from same trial never leak across folds.
 
Final CV results:
 
Model Type	Single-Split	5-Fold CV
 
Baseline (no embeddings)	0.74	0.68
With Embeddings (disease + drug)	0.79	0.55
 
 
Interpretation:
Embeddings help when training/validation distribution match (single-split)
But decrease generalization in strict Trial ID–based CV.
 
 
---
 
🔍 9. SHAP Explainability
 
Generated:
 
SHAP summary plot
 
SHAP bar importance
 
Feature ranking
 
Embedding component importance
 
 
Helps understand:
 
Which disease descriptors matter
 
Which drug properties matter
 
Which engineered features correlate with increase in PFS
 
 
 
---
 
📦 10. Outputs
 
Everything saved under /outputs:
 
prediction_vs_actual.png
 
feature_importance.png
 
shap_summary.png
 
shap_bar.png
 
cv_summary.xlsx
 
Fold-wise prediction Excel files
 
Tuned parameter JSON
 
Final .pkl model files
 
 
 
---
 
🚀 11. Final Model Summary
 
✔ Best single-split model (usable):
 
LightGBM + Disease Embeddings
➡ R² = 0.79, MAE ≈ 2.05
 
✔ Best generalizable model (recommended):
 
Baseline LightGBM without embeddings
➡ 5-fold CV R² = 0.68
 