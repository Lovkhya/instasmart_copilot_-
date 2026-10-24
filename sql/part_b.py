"""part_b.py -- Customer Segments and Lapse Prediction"""
import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def run_part_b():
    os.makedirs("outputs", exist_ok=True)
    print("--- 1. BUILDING RFM TABLE ---")
    cust_df = pd.read_csv("data/clean/customers.csv")
    orders_df = pd.read_csv("data/clean/orders.csv")
    orders_df['order_ts'] = pd.to_datetime(orders_df['order_ts'])

    comp_orders = orders_df[orders_df['status'] == 'completed'].copy()
    snapshot = pd.to_datetime("2025-10-01 00:00:00")

    rfm = comp_orders.groupby('customer_id').agg(
        recency=('order_ts', lambda x: (snapshot - x.max()).days),
        frequency=('order_id', 'count'),
        monetary=('order_amount', 'sum')
    ).reset_index()

    print(f"RFM table built for {len(rfm)} customers.")

    print("\n--- 2. K-MEANS CLUSTERING ---")
    X_rfm = rfm[['recency', 'frequency', 'monetary']]
    scaler_rfm = StandardScaler()
    X_scaled = scaler_rfm.fit_transform(X_rfm)

    for k in range(2, 9):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        print(f"K={k}: Inertia={kmeans.inertia_:.1f}, Silhouette={sil:.3f}")

    # Choose K=4 based on balance and interpretability
    best_k = 4
    kmeans_final = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    rfm['cluster'] = kmeans_final.fit_predict(X_scaled)

    # Identify Premium segment (highest mean monetary)
    cluster_monetary = rfm.groupby('cluster')['monetary'].mean()
    premium_cluster = cluster_monetary.idxmax()

    rfm['segment_id'] = rfm['cluster']
    rfm['segment_name'] = rfm['cluster'].map({
        premium_cluster: 'Premium',
        0: 'Frequent',
        1: 'Occasional',
        2: 'At-Risk'
    }).fillna('Standard')
    rfm['is_premium'] = rfm['cluster'] == premium_cluster

    rfm[['customer_id', 'recency', 'frequency', 'monetary', 'segment_id', 'segment_name', 'is_premium']].to_csv(
        "outputs/customer_segments.csv", index=False
    )

    print("\n--- 3. LAPSE PREDICTION DATASET & MODELING ---")
    cutoff = pd.to_datetime("2025-09-01 00:00:00")
    pre_cutoff_orders = orders_df[orders_df['order_ts'] < cutoff]

    cohort_custs = pre_cutoff_orders[pre_cutoff_orders['status'] == 'completed']['customer_id'].unique()
    
    features = []
    for cid in cohort_custs:
        c_orders = pre_cutoff_orders[pre_cutoff_orders['customer_id'] == cid]
        c_comp = c_orders[c_orders['status'] == 'completed']
        
        recency = (cutoff - c_comp['order_ts'].max()).days if len(c_comp) > 0 else 999
        frequency = len(c_comp)
        monetary = c_comp['order_amount'].sum()
        avg_disc = c_comp['discount_pct'].mean() if len(c_comp) > 0 else 0
        
        final_statuses = c_orders[c_orders['status'].isin(['completed', 'cancelled'])]
        canc_rate = len(c_orders[c_orders['status'] == 'cancelled']) / len(final_statuses) if len(final_statuses) > 0 else 0
        avg_deliv = c_comp['delivery_minutes'].mean() if len(c_comp) > 0 else 0
        
        fav_cat = "Grocery" # Default placeholder fallback
        
        cust_row = cust_df[cust_df['customer_id'] == cid].iloc[0]
        city = cust_row['city']
        gender = cust_row['gender']
        age = cust_row['age']
        tenure = (cutoff - pd.to_datetime(cust_row['signup_date'])).days

        # Label: lapsed = 1 if no completed order between 2025-09-01 and 2025-09-30
            sept_orders = orders_df[
            (orders_df['customer_id'] == cid) & 
            (orders_df['status'] == 'completed') & 
            (orders_df['order_ts'] >= cutoff) & 
            (orders_df['order_ts'] <= pd.to_datetime("2025-09-30 23:59:59"))
        ]
        lapsed = 1 if len(sept_orders) == 0 else 0

        features.append({
            'customer_id': cid, 'recency': recency, 'frequency': frequency, 'monetary': monetary,
            'avg_discount_pct': avg_disc, 'cancellation_rate': canc_rate, 'avg_delivery_minutes': avg_deliv,
            'city': city, 'gender': gender, 'age': age, 'tenure': tenure, 'lapsed': lapsed
        })

    feat_df = pd.DataFrame(features)
    X = feat_df.drop(columns=['customer_id', 'lapsed'])
    y = feat_df['lapsed']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    numeric_features = ['recency', 'frequency', 'monetary', 'avg_discount_pct', 'cancellation_rate', 'avg_delivery_minutes', 'age', 'tenure']
    categorical_features = ['city', 'gender']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ]
    )

    clf_rf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=10, random_state=42))
    ])

    clf_rf.fit(X_train, y_train)
    joblib.dump(clf_rf, "outputs/lapse_model.joblib")

    preds = clf_rf.predict(X_test)
    print("Random Forest Test F1 (Lapsed):", f1_score(y_test, preds, zero_division=0))

    # Retention List for Premium customers
    test_customers = feat_df.loc[X_test.index].copy()
    test_customers['lapse_probability'] = clf_rf.predict_proba(X_test)[:, 1]
    test_customers['actually_lapsed'] = y_test.values

    # Merge with RFM whole-window premium flag
    premium_ids = rfm[rfm['is_premium']]['customer_id']
    ret_list = test_customers[test_customers['customer_id'].isin(premium_ids)].sort_values(by='lapse_probability', ascending=False).head(10)
    ret_list[['customer_id', 'lapse_probability', 'actually_lapsed']].to_csv("outputs/retention_list.csv", index=False)
    print("Part B completed successfully and saved outputs.")

if __name__ == "__main__":
    run_part_b()