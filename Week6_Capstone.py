"""
Week 6 Capstone: Integrative Data Science Project
Problem: Customer Churn Analysis & Retention Strategy for a Telecommunications Company

Dataset: IBM Telco Customer Churn dataset (public; originally distributed by IBM as a
sample dataset for its Watson Studio / Cognos Analytics tutorials; mirrored at
https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d). 7,043 customers,
21 columns describing demographics, account information, subscribed services, and
whether the customer churned (canceled service).

Pipeline: data acquisition -> cleaning -> EDA -> feature engineering ->
UNSUPERVISED customer segmentation (K-Means) -> SUPERVISED churn prediction
(multiple classifiers, cross-validation, hyperparameter tuning) -> combined
evaluation and business recommendations.

All figures are saved to /home/claude/wk6/figs/ for embedding in the report.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_rand_score
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                      cross_validate, GridSearchCV)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, roc_curve, precision_recall_curve,
                              confusion_matrix, classification_report)

import os
FIG = "/home/claude/wk6/figs"
os.makedirs(FIG, exist_ok=True)
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 140
SEED = 42
np.random.seed(SEED)

# ===========================================================================
# PHASE 1: DATA ACQUISITION
# ===========================================================================
df_raw = pd.read_csv("/home/claude/wk6/telco_customer_churn.csv")
print("Raw shape:", df_raw.shape)
print(df_raw.head())

# ===========================================================================
# PHASE 2: DATA CLEANING
# ===========================================================================
df = df_raw.copy()

# 2.1 TotalCharges is stored as a string and contains 11 blank entries for
#     brand-new customers (tenure == 0), which pandas silently read as object
#     dtype rather than numeric. Coerce to numeric and inspect the resulting NaNs.
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
n_missing_total_charges = df["TotalCharges"].isna().sum()
print(f"\nMissing TotalCharges after coercion: {n_missing_total_charges}")
print("Tenure of affected rows:", df.loc[df['TotalCharges'].isna(), 'tenure'].unique())

# All missing TotalCharges correspond to tenure == 0 (brand-new customers who
# have not yet been billed) -> impute as 0, which is the correct domain value,
# not a statistical guess.
df.loc[df["TotalCharges"].isna(), "TotalCharges"] = 0.0

# 2.2 Drop the customer ID (a unique identifier with no predictive value)
df = df.drop(columns=["customerID"])

# 2.3 Standardize the "No internet service" / "No phone service" categories:
#     these are functionally equivalent to "No" for modeling purposes and
#     collapsing them reduces unnecessary cardinality in one-hot encoding.
service_cols_with_no_service = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                                 "TechSupport", "StreamingTV", "StreamingMovies"]
for col in service_cols_with_no_service:
    df[col] = df[col].replace({"No internet service": "No"})
df["MultipleLines"] = df["MultipleLines"].replace({"No phone service": "No"})

# 2.4 Encode target and binary Yes/No fields as 0/1
binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"] + service_cols_with_no_service + ["MultipleLines"]
for col in binary_cols:
    df[col] = df[col].map({"Yes": 1, "No": 0})
df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

print(f"\nFinal cleaned shape: {df.shape}, missing values remaining: {df.isnull().sum().sum()}")
print("Churn rate:", df["Churn"].mean())

# ===========================================================================
# PHASE 3: EXPLORATORY DATA ANALYSIS
# ===========================================================================
# 3.1 Churn rate
plt.figure(figsize=(4.5, 4))
counts = df["Churn"].value_counts().sort_index()
bars = plt.bar(["Stayed", "Churned"], counts.values, color=["#55A868", "#C44E52"])
for b, v in zip(bars, counts.values):
    plt.text(b.get_x() + b.get_width() / 2, v + 40, f"{v}\n({v/len(df):.1%})", ha="center")
plt.ylabel("Number of customers")
plt.title("Figure 1. Overall Churn Rate")
plt.tight_layout()
plt.savefig(f"{FIG}/fig1_churn_rate.png")
plt.close()

# 3.2 Numeric feature distributions by churn
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, col in zip(axes, ["tenure", "MonthlyCharges", "TotalCharges"]):
    sns.kdeplot(data=df, x=col, hue="Churn", fill=True, alpha=0.4, ax=ax,
                palette=["#55A868", "#C44E52"], common_norm=False)
    ax.set_title(col)
    ax.legend(["Churned", "Stayed"], loc="upper right", fontsize=8)
plt.suptitle("Figure 2. Distributions of Key Numeric Features by Churn Status")
plt.tight_layout()
plt.savefig(f"{FIG}/fig2_numeric_by_churn.png")
plt.close()

# 3.3 Churn rate by key categorical drivers
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
for ax, col in zip(axes, ["Contract", "InternetService", "PaymentMethod"]):
    rates = df.groupby(col)["Churn"].mean().sort_values(ascending=False)
    rates.plot(kind="bar", ax=ax, color="#4C72B0")
    ax.set_ylabel("Churn rate")
    ax.set_title(col)
    ax.set_ylim(0, 0.6)
    ax.tick_params(axis="x", rotation=30)
plt.suptitle("Figure 3. Churn Rate by Contract Type, Internet Service, and Payment Method")
plt.tight_layout()
plt.savefig(f"{FIG}/fig3_categorical_churn_rates.png")
plt.close()

# 3.4 Correlation heatmap of numeric + binary features
corr_cols = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
             "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
             "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling",
             "MonthlyCharges", "TotalCharges", "Churn"]
plt.figure(figsize=(10, 8))
sns.heatmap(df[corr_cols].corr(), cmap="coolwarm", center=0, square=True, cbar_kws={"shrink": .8})
plt.title("Figure 4. Correlation Heatmap (Numeric & Binary Features)")
plt.tight_layout()
plt.savefig(f"{FIG}/fig4_correlation_heatmap.png")
plt.close()

eda_summary = {
    "overall_churn_rate": round(float(df["Churn"].mean()), 4),
    "n_customers": int(len(df)),
    "month_to_month_churn_rate": round(float(df.loc[df["Contract"] == "Month-to-month", "Churn"].mean()), 4),
    "two_year_churn_rate": round(float(df.loc[df["Contract"] == "Two year", "Churn"].mean()), 4),
    "fiber_churn_rate": round(float(df.loc[df["InternetService"] == "Fiber optic", "Churn"].mean()), 4),
    "avg_tenure_churned": round(float(df.loc[df["Churn"] == 1, "tenure"].mean()), 1),
    "avg_tenure_stayed": round(float(df.loc[df["Churn"] == 0, "tenure"].mean()), 1),
}
print("\nEDA summary:", json.dumps(eda_summary, indent=2))

# ===========================================================================
# PHASE 4: FEATURE ENGINEERING
# ===========================================================================
# 4.1 Tenure groups (business-interpretable bucketing)
def tenure_group(t):
    if t <= 12: return "0-1yr"
    elif t <= 24: return "1-2yr"
    elif t <= 48: return "2-4yr"
    else: return "4yr+"
df["TenureGroup"] = df["tenure"].apply(tenure_group)

# 4.2 Total number of subscribed add-on services (engineered aggregate feature)
addon_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
df["NumAddonServices"] = df[addon_cols].sum(axis=1)

# 4.3 Average monthly spend per tenure month (captures spend intensity vs. loyalty)
df["AvgChargePerTenure"] = df["TotalCharges"] / df["tenure"].replace(0, 1)

# 4.4 One-hot encode remaining multi-category categorical columns
multi_cat_cols = ["InternetService", "Contract", "PaymentMethod", "TenureGroup"]
df_encoded = pd.get_dummies(df, columns=multi_cat_cols, drop_first=False)
bool_cols = df_encoded.select_dtypes(include="bool").columns
df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

print(f"\nAfter feature engineering & encoding: {df_encoded.shape}")
feature_cols_model = [c for c in df_encoded.columns if c != "Churn"]

# ===========================================================================
# PHASE 5: UNSUPERVISED LEARNING — CUSTOMER SEGMENTATION (K-MEANS)
# ===========================================================================
# Use a curated feature set capturing tenure, spend, and service adoption —
# the dimensions most relevant to a business segmentation, standardized prior
# to distance-based clustering. The Churn label itself is excluded from the
# clustering input so segments reflect behavior/usage, not the outcome we
# want to later explain with them.
segment_features = ["tenure", "MonthlyCharges", "TotalCharges", "NumAddonServices",
                     "SeniorCitizen", "Partner", "Dependents", "PaperlessBilling"]
X_seg = df_encoded[segment_features].values
scaler_seg = StandardScaler()
X_seg_scaled = scaler_seg.fit_transform(X_seg)

# 5.1 Determine optimal k via elbow + silhouette
k_range = range(2, 9)
inertias, silhouettes = [], []
for k in k_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED)
    labels_k = km.fit_predict(X_seg_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_seg_scaled, labels_k))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
axes[0].plot(list(k_range), inertias, marker="o", color="#4C72B0")
axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia (WCSS)"); axes[0].set_title("Elbow Method")
axes[1].plot(list(k_range), silhouettes, marker="o", color="#DD8452")
axes[1].set_xlabel("k"); axes[1].set_ylabel("Silhouette Score"); axes[1].set_title("Silhouette Analysis")
plt.suptitle("Figure 5. Selecting the Optimal Number of Customer Segments")
plt.tight_layout()
plt.savefig(f"{FIG}/fig5_elbow_silhouette.png")
plt.close()

best_k = list(k_range)[int(np.argmax(silhouettes))]
print(f"\nBest k by silhouette: {best_k}")
K_FINAL = 4  # chosen for business interpretability alongside the statistical signal
kmeans = KMeans(n_clusters=K_FINAL, n_init=10, random_state=SEED)
seg_labels = kmeans.fit_predict(X_seg_scaled)
df_encoded["Segment"] = seg_labels
df["Segment"] = seg_labels
sil_final = silhouette_score(X_seg_scaled, seg_labels)
print(f"Final K-Means (k={K_FINAL}) silhouette score: {sil_final:.3f}")

# 5.2 PCA visualization of segments
pca = PCA(n_components=2, random_state=SEED)
X_pca = pca.fit_transform(X_seg_scaled)
palette = sns.color_palette("Set2", K_FINAL)
plt.figure(figsize=(6.5, 5.5))
for c in range(K_FINAL):
    mask = seg_labels == c
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], s=18, alpha=0.6, color=palette[c], label=f"Segment {c}")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
plt.title(f"Figure 6. Customer Segments in PCA Space (k={K_FINAL})")
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIG}/fig6_segments_pca.png")
plt.close()

# 5.3 Segment profiling
profile_raw = df.groupby("Segment")[["tenure", "MonthlyCharges", "TotalCharges", "NumAddonServices"]].mean()
profile_raw["n_customers"] = df.groupby("Segment").size()
profile_raw["churn_rate"] = df.groupby("Segment")["Churn"].mean()
profile_raw.to_csv("/home/claude/wk6/segment_profiles.csv")
print("\nSegment profiles:\n", profile_raw)

plt.figure(figsize=(8, 4.5))
profile_raw["churn_rate"].plot(kind="bar", color=[palette[i] for i in profile_raw.index])
plt.ylabel("Churn rate")
plt.xlabel("Segment")
plt.title("Figure 7. Churn Rate by Customer Segment (Unsupervised Clusters)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{FIG}/fig7_segment_churn_rates.png")
plt.close()

# Segment feature heatmap (z-scored)
z_profile = pd.DataFrame(X_seg_scaled, columns=segment_features)
z_profile["Segment"] = seg_labels
z_means = z_profile.groupby("Segment")[segment_features].mean()
plt.figure(figsize=(10, 4))
sns.heatmap(z_means, cmap="coolwarm", center=0, annot=True, fmt=".2f", cbar_kws={"label": "Mean (standardized)"})
plt.title("Figure 8. Segment Feature Profile Heatmap (Standardized Means)")
plt.tight_layout()
plt.savefig(f"{FIG}/fig8_segment_profile_heatmap.png")
plt.close()

# ===========================================================================
# PHASE 6: SUPERVISED LEARNING — CHURN PREDICTION
# ===========================================================================
X = df_encoded[feature_cols_model]  # includes engineered Segment as a feature too
y = df_encoded["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
print("Train churn rate:", y_train.mean(), " Test churn rate:", y_test.mean())

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, class_weight="balanced", random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED),
    "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
    "SVM (RBF)": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED),
}

cv_results = {}
for name, model in models.items():
    res = cross_validate(model, X_train_scaled, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    cv_results[name] = {m: (res[f"test_{m}"].mean(), res[f"test_{m}"].std()) for m in scoring}
    print(f"\n{name}:")
    for m in scoring:
        print(f"  {m}: {res[f'test_{m}'].mean():.4f} (+/- {res[f'test_{m}'].std():.4f})")

plt.figure(figsize=(8, 4.5))
names = list(cv_results.keys())
roc_means = [cv_results[n]["roc_auc"][0] for n in names]
f1_means = [cv_results[n]["f1"][0] for n in names]
xpos = np.arange(len(names))
w = 0.35
plt.bar(xpos - w/2, roc_means, width=w, label="ROC-AUC", color="#4C72B0")
plt.bar(xpos + w/2, f1_means, width=w, label="F1 (churn class)", color="#DD8452")
plt.xticks(xpos, names, rotation=20, ha="right")
plt.ylabel("Cross-validated score")
plt.title("Figure 9. 5-Fold Cross-Validation: Churn Model Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIG}/fig9_model_comparison.png")
plt.close()

# Hyperparameter tuning for the two strongest candidates
logreg_grid = {"C": [0.01, 0.1, 1, 10], "penalty": ["l2"], "solver": ["lbfgs"]}
gs_logreg = GridSearchCV(LogisticRegression(max_iter=5000, class_weight="balanced", random_state=SEED),
                          logreg_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
gs_logreg.fit(X_train_scaled, y_train)

rf_grid = {"n_estimators": [200, 400], "max_depth": [6, 10, None], "min_samples_leaf": [1, 3, 5]}
gs_rf = GridSearchCV(RandomForestClassifier(class_weight="balanced", random_state=SEED),
                      rf_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
gs_rf.fit(X_train_scaled, y_train)

gb_grid = {"n_estimators": [100, 200], "max_depth": [2, 3, 4], "learning_rate": [0.05, 0.1]}
gs_gb = GridSearchCV(GradientBoostingClassifier(random_state=SEED),
                      gb_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
gs_gb.fit(X_train_scaled, y_train)

print("\nBest Logistic Regression:", gs_logreg.best_params_, gs_logreg.best_score_)
print("Best Random Forest:", gs_rf.best_params_, gs_rf.best_score_)
print("Best Gradient Boosting:", gs_gb.best_params_, gs_gb.best_score_)

candidates = {"Logistic Regression": gs_logreg, "Random Forest": gs_rf, "Gradient Boosting": gs_gb}
final_name = max(candidates, key=lambda k: candidates[k].best_score_)
final_model = candidates[final_name].best_estimator_
print(f"\n>>> Final selected churn model: {final_name}")

# Final test-set evaluation
y_pred = final_model.predict(X_test_scaled)
y_proba = final_model.predict_proba(X_test_scaled)[:, 1]

test_acc = accuracy_score(y_test, y_pred)
test_prec = precision_score(y_test, y_pred)
test_rec = recall_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)
test_auc = roc_auc_score(y_test, y_proba)
report_txt = classification_report(y_test, y_pred, target_names=["Stayed", "Churned"], digits=4)
print(f"\nFinal Test Metrics ({final_name}):")
print(f"  Accuracy: {test_acc:.4f}  Precision: {test_prec:.4f}  Recall: {test_rec:.4f}  F1: {test_f1:.4f}  ROC-AUC: {test_auc:.4f}")
print(report_txt)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5.5, 4.7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Stayed", "Churned"], yticklabels=["Stayed", "Churned"])
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.title(f"Figure 10. Confusion Matrix \u2014 {final_name} (Test Set)")
plt.tight_layout()
plt.savefig(f"{FIG}/fig10_confusion_matrix.png")
plt.close()

fpr, tpr, _ = roc_curve(y_test, y_proba)
prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_proba)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].plot(fpr, tpr, color="#4C72B0", lw=2, label=f"AUC = {test_auc:.3f}")
axes[0].plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--")
axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve"); axes[0].legend(loc="lower right")
axes[1].plot(rec_arr, prec_arr, color="#DD8452", lw=2)
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve")
plt.suptitle(f"Figure 11. ROC and Precision-Recall Curves \u2014 {final_name} (Test Set)")
plt.tight_layout()
plt.savefig(f"{FIG}/fig11_roc_pr_curves.png")
plt.close()

# Feature importance
if hasattr(final_model, "feature_importances_"):
    importances = pd.Series(final_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    imp_label = "Feature Importance"
else:
    importances = pd.Series(final_model.coef_[0], index=X_train.columns).sort_values(key=abs, ascending=False)
    imp_label = "Coefficient Magnitude (standardized)"
importances.to_csv("/home/claude/wk6/feature_importance.csv")
top15 = importances.head(15)
plt.figure(figsize=(8, 7))
colors = ["#C44E52" if v > 0 else "#4C72B0" for v in top15.values] if "Coef" in imp_label else "#4C72B0"
sns.barplot(x=top15.values, y=top15.index, palette=colors if isinstance(colors, list) else None,
            color=None if isinstance(colors, list) else colors)
plt.xlabel(imp_label)
plt.title(f"Figure 12. Top 15 Churn Drivers \u2014 {final_name}")
plt.tight_layout()
plt.savefig(f"{FIG}/fig12_feature_importance.png")
plt.close()

# ===========================================================================
# PHASE 7: COMBINING UNSUPERVISED + SUPERVISED INSIGHTS
# ===========================================================================
# Cross-tabulate the unsupervised segments against predicted churn probability,
# to translate the churn model into segment-level, business-actionable priorities.
df_encoded["churn_proba_full"] = final_model.predict_proba(
    pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
)[:, 1]
segment_risk = df_encoded.groupby("Segment").agg(
    n_customers=("Churn", "size"),
    actual_churn_rate=("Churn", "mean"),
    predicted_churn_risk=("churn_proba_full", "mean"),
).round(4)
segment_risk.to_csv("/home/claude/wk6/segment_risk.csv")
print("\nSegment risk (unsupervised segments x supervised churn risk):\n", segment_risk)

plt.figure(figsize=(7.5, 4.8))
x = np.arange(K_FINAL)
w = 0.35
plt.bar(x - w/2, segment_risk["actual_churn_rate"], width=w, label="Actual churn rate", color="#55A868")
plt.bar(x + w/2, segment_risk["predicted_churn_risk"], width=w, label="Model-predicted avg. churn risk", color="#C44E52")
plt.xticks(x, [f"Segment {i}" for i in segment_risk.index])
plt.ylabel("Rate / Probability")
plt.title("Figure 13. Segment Risk: Actual vs. Model-Predicted Churn", fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIG}/fig13_combined_segment_risk.png")
plt.close()

# ===========================================================================
# SAVE SUMMARY
# ===========================================================================
summary = {
    "n_customers": int(len(df)),
    "n_features_raw": int(df_raw.shape[1] - 1),
    "n_features_engineered": int(len(feature_cols_model)),
    "missing_total_charges_found": int(n_missing_total_charges),
    "overall_churn_rate": eda_summary["overall_churn_rate"],
    "month_to_month_churn_rate": eda_summary["month_to_month_churn_rate"],
    "two_year_churn_rate": eda_summary["two_year_churn_rate"],
    "fiber_churn_rate": eda_summary["fiber_churn_rate"],
    "avg_tenure_churned": eda_summary["avg_tenure_churned"],
    "avg_tenure_stayed": eda_summary["avg_tenure_stayed"],
    "cluster_k_final": K_FINAL,
    "cluster_silhouette": round(float(sil_final), 4),
    "segment_profiles": profile_raw.round(2).to_dict(orient="index"),
    "cv_results": {name: {m: round(v[0], 4) for m, v in vals.items()} for name, vals in cv_results.items()},
    "best_logreg_params": gs_logreg.best_params_,
    "best_logreg_cv_auc": round(float(gs_logreg.best_score_), 4),
    "best_rf_params": gs_rf.best_params_,
    "best_rf_cv_auc": round(float(gs_rf.best_score_), 4),
    "best_gb_params": gs_gb.best_params_,
    "best_gb_cv_auc": round(float(gs_gb.best_score_), 4),
    "final_model": final_name,
    "test_accuracy": round(float(test_acc), 4),
    "test_precision": round(float(test_prec), 4),
    "test_recall": round(float(test_rec), 4),
    "test_f1": round(float(test_f1), 4),
    "test_roc_auc": round(float(test_auc), 4),
    "confusion_matrix": cm.tolist(),
    "top5_features": importances.head(5).index.tolist(),
    "segment_risk": segment_risk.reset_index().to_dict(orient="records"),
    "train_size": int(len(X_train)),
    "test_size": int(len(X_test)),
}
with open("/home/claude/wk6/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\n\nFINAL SUMMARY:\n", json.dumps(summary, indent=2, default=str))
print("\nDone.")
