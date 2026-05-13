# 05_ml_models.py
# ML analysis on FDA drug label data
# Model 1: XGBoost — predict black box warning (classification)
# Model 2: TF-IDF + K-Means — drug clustering by indication (NLP)
# Model 3: SHAP explainability on Model 1
# Outputs: ML results pushed back to BigQuery for Power BI

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving PNGs
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, ConfusionMatrixDisplay)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import shap
from google.cloud import bigquery

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "healthcare-rag-prachi")
DATASET_ID = "healthcare_rag"
OUTPUT_DIR = "outputs/ml"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("data/drug_labels_full.csv")
print(f"Loaded {len(df)} records")
print(f"Columns: {list(df.columns)}")

# ── Clean for ML ──────────────────────────────────────────────────────────────
df = df.dropna(subset=["generic_name"]).reset_index(drop=True)
df["warning_length"] = df["warning_length"].fillna(0).astype(int)
df["interaction_length"] = df["interaction_length"].fillna(0).astype(int)
df["has_black_box"] = df["has_black_box"].fillna(0).astype(int)
df["is_prescription"] = df["is_prescription"].fillna(0).astype(int)
df["has_interactions"] = df["has_interactions"].fillna(0).astype(int)
df["adverse_length"] = df.get("adverse_length", pd.Series([0]*len(df))).fillna(0).astype(int)
df["has_pediatric_info"] = df.get("has_pediatric_info", pd.Series([0]*len(df))).fillna(0).astype(int)
df["has_geriatric_info"] = df.get("has_geriatric_info", pd.Series([0]*len(df))).fillna(0).astype(int)

print(f"\nBlack box distribution:\n{df['has_black_box'].value_counts()}")
print(f"Prescription distribution:\n{df['is_prescription'].value_counts()}")

# ════════════════════════════════════════════════════════════════════════════════
# MODEL 1: XGBoost — Predict Black Box Warning
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MODEL 1: XGBoost Black Box Warning Classifier")
print("="*60)

FEATURES = [
    "warning_length",
    "interaction_length",
    "adverse_length",
    "has_interactions",
    "is_prescription",
    "has_pediatric_info",
    "has_geriatric_info",
]
TARGET = "has_black_box"

# Only use features that exist in the dataframe
available_features = [f for f in FEATURES if f in df.columns]
print(f"Features used: {available_features}")

X = df[available_features].copy()
y = df[TARGET].copy()

# Handle class imbalance
black_box_rate = y.mean()
scale_pos_weight = (1 - black_box_rate) / black_box_rate
print(f"Black box rate: {black_box_rate:.1%} | scale_pos_weight: {scale_pos_weight:.1f}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="logloss",
    verbosity=0,
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Black Box", "Black Box"]))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
print(f"5-Fold CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Black Box", "Black Box"])
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("XGBoost: Black Box Warning Prediction\nConfusion Matrix", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUTPUT_DIR}/confusion_matrix.png")

# Feature importance plot
feat_imp = pd.Series(model.feature_importances_, index=available_features).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(7, 4))
feat_imp.plot(kind="barh", ax=ax, color="#1565C0")
ax.set_title("XGBoost Feature Importance\nPredicting Black Box Warning", fontsize=12, fontweight="bold")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUTPUT_DIR}/feature_importance.png")

# ── SHAP Explainability ───────────────────────────────────────────────────────
print("\nGenerating SHAP explainability...")
try:
    explainer = shap.TreeExplainer(model)
    # Use a sample for speed
    X_sample = X_test.sample(min(200, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(X_sample)

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_values, X_sample, show=False, plot_size=None)
    plt.title("SHAP Feature Impact on Black Box Warning Prediction", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/shap_summary.png")
except Exception as e:
    print(f"SHAP error (non-critical): {e}")

# Add ML predictions back to dataframe
df["black_box_predicted"] = model.predict(X)
df["black_box_probability"] = model.predict_proba(X)[:, 1].round(4)

# ════════════════════════════════════════════════════════════════════════════════
# MODEL 2: TF-IDF + K-Means Drug Clustering
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MODEL 2: TF-IDF + K-Means Drug Clustering by Indication")
print("="*60)

# Use indications text for clustering
df_cluster = df.dropna(subset=["indications"]).copy()
df_cluster = df_cluster[df_cluster["indications"].str.len() > 50]
print(f"Records for clustering: {len(df_cluster)}")

tfidf = TfidfVectorizer(
    max_features=500,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=3,
)
X_tfidf = tfidf.fit_transform(df_cluster["indications"].fillna(""))

N_CLUSTERS = 8
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
df_cluster["cluster"] = kmeans.fit_predict(X_tfidf)

# Get top keywords per cluster
print("\nTop keywords per cluster:")
feature_names = tfidf.get_feature_names_out()
cluster_labels = {}
for cluster_id in range(N_CLUSTERS):
    center = kmeans.cluster_centers_[cluster_id]
    top_indices = center.argsort()[-5:][::-1]
    top_words = [feature_names[i] for i in top_indices]
    cluster_labels[cluster_id] = ", ".join(top_words)
    count = (df_cluster["cluster"] == cluster_id).sum()
    print(f"  Cluster {cluster_id} ({count} drugs): {cluster_labels[cluster_id]}")

# PCA visualization
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_tfidf.toarray())
df_cluster["pca_x"] = coords[:, 0]
df_cluster["pca_y"] = coords[:, 1]

fig, ax = plt.subplots(figsize=(10, 7))
colors = plt.cm.Set2(np.linspace(0, 1, N_CLUSTERS))
for c in range(N_CLUSTERS):
    mask = df_cluster["cluster"] == c
    ax.scatter(
        df_cluster[mask]["pca_x"],
        df_cluster[mask]["pca_y"],
        color=colors[c],
        label=f"C{c}: {cluster_labels[c][:30]}",
        alpha=0.5, s=8
    )
ax.set_title("FDA Drug Clusters by Indication Text\n(TF-IDF + K-Means + PCA)", fontsize=12, fontweight="bold")
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/drug_clusters.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {OUTPUT_DIR}/drug_clusters.png")

# Add cluster back to main df
df = df.merge(
    df_cluster[["set_id", "cluster", "pca_x", "pca_y"]],
    on="set_id", how="left"
)
df["cluster"] = df["cluster"].fillna(-1).astype(int)

# ════════════════════════════════════════════════════════════════════════════════
# Push ML results back to BigQuery
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Pushing ML results to BigQuery...")
print("="*60)

try:
    client = bigquery.Client(project=PROJECT_ID)

    ml_df = df[[
        "set_id", "brand_name", "generic_name", "search_category",
        "has_black_box", "black_box_predicted", "black_box_probability",
        "warning_length", "interaction_length", "is_prescription",
        "cluster"
    ]].copy()

    table_ref = f"{PROJECT_ID}.{DATASET_ID}.drug_labels_ml"
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        autodetect=True,
    )
    job = client.load_table_from_dataframe(ml_df, table_ref, job_config=job_config)
    job.result()
    print(f"Pushed {len(ml_df)} rows to BigQuery: {table_ref}")
    print("Power BI can now connect to drug_labels_ml for ML visuals!")

except Exception as e:
    print(f"BigQuery push error: {e}")
    print("Saving ML results locally instead...")
    df.to_csv("data/drug_labels_with_ml.csv", index=False)
    print("Saved to data/drug_labels_with_ml.csv")

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("ML COMPLETE — Summary")
print("="*60)
print(f"Total records processed: {len(df)}")
print(f"Black box prediction AUC: {roc_auc_score(y_test, y_prob):.4f}")
print(f"Drug clusters created: {N_CLUSTERS}")
print(f"\nOutput files saved to: {OUTPUT_DIR}/")
print("  confusion_matrix.png  — paste in README + LinkedIn")
print("  feature_importance.png — paste in README + LinkedIn")
print("  shap_summary.png       — paste in README + LinkedIn")
print("  drug_clusters.png      — paste in README + LinkedIn")
print("\nBigQuery table: drug_labels_ml")
print("Connect Power BI to drug_labels_ml for ML dashboard page!")