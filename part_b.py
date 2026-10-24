import os
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, silhouette_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Ensure outputs directory exists
os.makedirs("outputs", exist_ok=True)

print("--- Starting Part B ---")

# 1. Load clean data from Part A
customers = pd.read_csv("data/clean/customers.csv")
products = pd.read_csv("data/clean/products.csv")
orders = pd.read_csv("data/clean/orders.csv")

orders["order_ts"] = pd.to_datetime(orders["order_ts"])
customers["signup_date"] = pd.to_datetime(customers["signup_date"])

completed_orders = orders[orders["status"] == "completed"].copy()

# ==========================================
# Task 1: Build the RFM Table
# ==========================================
snapshot = pd.Timestamp("2025-10-01 00:00:00")

rfm = (
    completed_orders.groupby("customer_id")
    .agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
    )
    .reset_index()
)

rfm["recency"] = (snapshot - rfm["last_order_ts"]).dt.days
rfm = rfm[["customer_id", "recency", "frequency", "monetary"]]

print(f"RFM Table Built: {len(rfm)} customers.")

# ==========================================
# Task 2 & 3: K-means Clustering & Profiling
# ==========================================
scaler_rfm = StandardScaler()
X_rfm_scaled = scaler_rfm.fit_transform(rfm[["recency", "frequency", "monetary"]])

# Choose K = 4 (optimal standard based on project brief)
CHOSEN_K = 4
kmeans = KMeans(n_clusters=CHOSEN_K, n_init=10, random_state=42)
rfm["segment_id"] = kmeans.fit_predict(X_rfm_scaled)

segment_profile = (
    rfm.groupby("segment_id")
    .agg(
        size=("customer_id", "count"),
        mean_recency=("recency", "mean"),
        mean_frequency=("frequency", "mean"),
        mean_monetary=("monetary", "mean"),
    )
    .reset_index()
)

premium_seg_id = segment_profile.loc[segment_profile["mean_monetary"].idxmax(), "segment_id"]

def assign_segment_metadata(seg_id):
    if seg_id == premium_seg_id:
        return "VIP High-Value", True
    elif seg_id == 0:
        return "Active Regulars", False
    elif seg_id == 1:
        return "At-Risk Lapsed", False
    else:
        return "New / Low Engagement", False

rfm[["segment_name", "is_premium"]] = pd.DataFrame(
    rfm["segment_id"].apply(assign_segment_metadata).tolist(), index=rfm.index
)

customer_segments_out = customers[["customer_id"]].merge(
    rfm[["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]],
    on="customer_id", how="left"
)
customer_segments_out.to_csv("outputs/customer_segments.csv", index=False)
print("Saved outputs/customer_segments.csv")

# ==========================================
# Task 4 & 5: Define Lapse Problem & Features (No Leakage)
# ==========================================
cutoff = pd.Timestamp("2025-09-01 00:00:00")

pre_cutoff_completed = completed_orders[orders["order_ts"] < cutoff]
cohort_ids = pre_cutoff_completed["customer_id"].unique()

sep_completed = orders[
    (orders["order_ts"] >= cutoff) & 
    (orders["order_ts"] < pd.Timestamp("2025-10-01")) & 
    (orders["status"] == "completed")
]["customer_id"].unique()

labels_df = pd.DataFrame({"customer_id": cohort_ids})
labels_df["lapsed"] = (~labels_df["customer_id"].isin(sep_completed)).astype(int)

orders_prod = pre_cutoff_completed.merge(products, on="product_id", how="left")

feat_recency = pre_cutoff_completed.groupby("customer_id")["order_ts"].max().apply(lambda x: (cutoff - x).days).rename("feat_recency")
feat_freq = pre_cutoff_completed.groupby("customer_id")["order_id"].nunique().rename("feat_frequency")
feat_mon = pre_cutoff_completed.groupby("customer_id")["order_amount"].sum().rename("feat_monetary")
feat_disc = pre_cutoff_completed.groupby("customer_id")["discount_pct"].mean().rename("feat_avg_discount")

all_pre_orders = orders[(orders["order_ts"] < cutoff) & (orders["customer_id"].isin(cohort_ids))]
canc_counts = all_pre_orders[all_pre_orders["status"] == "cancelled"].groupby("customer_id")["order_id"].count()
total_counts = all_pre_orders.groupby("customer_id")["order_id"].count()
feat_canc_rate = (canc_counts / total_counts).fillna(0).rename("feat_cancellation_rate")

feat_deliv = pre_cutoff_completed.groupby("customer_id")["delivery_minutes"].mean().rename("feat_avg_delivery_mins")

cat_mode = (
    orders_prod.groupby(["customer_id", "category"])
    .size()
    .reset_index(name="cnt")
    .sort_values(["customer_id", "cnt"], ascending=[True, False])
    .drop_duplicates("customer_id")
    .set_index("customer_id")["category"]
    .rename("favorite_category")
)

features_df = labels_df.set_index("customer_id").join([
    feat_recency, feat_freq, feat_mon, feat_disc, feat_canc_rate, feat_deliv, cat_mode
]).reset_index()

features_df = features_df.merge(customers, on="customer_id", how="left")
features_df["tenure_days"] = (cutoff - features_df["signup_date"]).dt.days

X = features_df[[
    "feat_recency", "feat_frequency", "feat_monetary", "feat_avg_discount",
    "feat_cancellation_rate", "feat_avg_delivery_mins", "favorite_category",
    "city", "gender", "age", "tenure_days"
]].copy()
y = features_df["lapsed"].values

X["age"] = X["age"].fillna(X["age"].median())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ==========================================
# Task 6: Preprocessing & Model Training
# ==========================================
numeric_features = [
    "feat_recency", "feat_frequency", "feat_monetary", "feat_avg_discount",
    "feat_cancellation_rate", "feat_avg_delivery_mins", "age", "tenure_days"
]
categorical_features = ["favorite_category", "city", "gender"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features)
    ]
)

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

rf_param_grid = {
    "classifier__max_depth": [3, 5, 8],
    "classifier__min_samples_leaf": [5, 10, 20]
}
rf_grid = GridSearchCV(rf_pipeline, rf_param_grid, cv=5, scoring="f1")
rf_grid.fit(X_train, y_train)

y_pred_test = rf_grid.predict(X_test)
print(f"Best Random Forest CV F1: {rf_grid.best_score_:.3f}")
print(f"Test F1 (Lapsed Class): {f1_score(y_test, y_pred_test, zero_division=0):.3f}")

# ==========================================
# Task 7 & 8: Persist Model & Retention List
# ==========================================
joblib.dump(rf_grid, "outputs/lapse_model.joblib")
print("Saved outputs/lapse_model.joblib")

test_customers_df = X_test.copy()
test_customers_df["customer_id"] = features_df.loc[X_test.index, "customer_id"]
test_customers_df["actually_lapsed"] = y_test
test_customers_df["lapse_probability"] = rf_grid.predict_proba(X_test)[:, 1]

test_customers_df = test_customers_df.merge(
    customer_segments_out[["customer_id", "is_premium"]], on="customer_id", how="left"
)

retention_list = (
    test_customers_df[test_customers_df["is_premium"] == True]
    .sort_values("lapse_probability", ascending=False)
    [["customer_id", "lapse_probability", "actually_lapsed"]]
    .head(10)
)

retention_list.to_csv("outputs/retention_list.csv", index=False)
print("Saved outputs/retention_list.csv successfully!")
