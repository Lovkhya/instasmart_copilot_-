import os
import sqlite3
import numpy as np
import pandas as pd

# Paths
RAW_DIR = os.path.join("data", "raw")
CLEAN_DIR = os.path.join("data", "clean")
DB_PATH = os.path.join("data", "instamart.db")
os.makedirs(CLEAN_DIR, exist_ok=True)

print("--- 1. LOADING RAW DATA ---")
customers = pd.read_csv(os.path.join(RAW_DIR, "customers.csv"))
products = pd.read_csv(os.path.join(RAW_DIR, "products.csv"))
orders = pd.read_csv(os.path.join(RAW_DIR, "orders.csv"))
tickets = pd.read_csv(os.path.join(RAW_DIR, "tickets.csv"), keep_default_na=False)

# Audit shapes and missing values
print(f"Customers shape: {customers.shape}")
print(f"Products shape: {products.shape}")
print(f"Orders shape: {orders.shape}")

print("\n--- 2. CLEANING CUSTOMERS ---")
# Standardize city: remove leading/trailing spaces and apply title case
customers["city"] = customers["city"].str.strip().str.title()

# Map gender spellings to male, female, or unknown (blank -> unknown)
gender_map = {
    "male": "male", "Male": "male", "M": "male", "m": "male", "MALE": "male",
    "female": "female", "Female": "female", "F": "female", "f": "female", "FEMALE": "female"
}
customers["gender"] = customers["gender"].map(gender_map).fillna("unknown")

# Treat ages outside 10-100 as missing, then fill with median age of customer's city
customers["age"] = pd.to_numeric(customers["age"], errors="coerce")
customers.loc[(customers["age"] < 10) | (customers["age"] > 100), "age"] = np.nan
city_median_ages = customers.groupby("city")["age"].transform("median")
customers["age"] = customers["age"].fillna(city_median_ages)

# Convert signup_date to datetime
customers["signup_date"] = pd.to_datetime(customers["signup_date"])

print("Customers cleaned successfully.")

print("\n--- 3. CLEANING ORDERS ---")
# Convert order_ts to datetime
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

# Remove exact duplicate rows
orders = orders.drop_duplicates()

# For orders appearing more than once with different statuses, keep the record with latest order_ts
orders = orders.sort_values("order_ts").groupby("order_id", as_index=False).last()

# Convert order_amount to numeric, turning unparseable entries into missing values
orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")

# Merge with products to get price_inr for missing amount imputation
orders_merged = orders.merge(products, on="product_id", how="left")
missing_amt_mask = orders_merged["order_amount"].isna()
orders_merged.loc[missing_amt_mask, "order_amount"] = np.round(
    orders_merged.loc[missing_amt_mask, "price_inr"] * 
    orders_merged.loc[missing_amt_mask, "quantity"] * 
    (1 - orders_merged.loc[missing_amt_mask, "discount_pct"] / 100), 2
)
orders["order_amount"] = orders_merged["order_amount"]

# Set negative delivery_minutes to missing
orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = np.nan

# Impute missing delivery_minutes for completed orders using median delivery time of that product category
orders = orders.merge(products[["product_id", "category"]], on="product_id", how="left")
completed_mask = (orders["status"] == "completed") & (orders["delivery_minutes"].isna())
cat_median_deliv = orders[orders["status"] == "completed"].groupby("category")["delivery_minutes"].transform("median")
orders.loc[completed_mask, "delivery_minutes"] = cat_median_deliv

# Drop temporary category column used for grouping
orders = orders.drop(columns=["category"])

print("Orders cleaned successfully.")

print("\n--- 4. STORING CLEAN TABLES AND SQLITE DATABASE ---")
# Save clean csvs
customers.to_csv(os.path.join(CLEAN_DIR, "customers.csv"), index=False)
products.to_csv(os.path.join(CLEAN_DIR, "products.csv"), index=False)
orders.to_csv(os.path.join(CLEAN_DIR, "orders.csv"), index=False)

# Store in SQLite database
conn = sqlite3.connect(DB_PATH)
customers.to_sql("customers", conn, if_exists="replace", index=False)
products.to_sql("products", conn, if_exists="replace", index=False)
orders.to_sql("orders", conn, if_exists="replace", index=False)
conn.close()
print("Stored to data/clean/ and data/instamart.db.")

print("\n--- 5. VALIDATION CHECKPOINTS ---")
# Assert checkpoints from project brief
assert len(orders) == 12309, f"Expected 12309 clean order rows, got {len(orders)}"
completed_orders_sum = orders[orders["status"] == "completed"]["order_amount"].sum()
assert np.isclose(completed_orders_sum, 2998668.80, atol=1e-2), f"Revenue sum mismatch: {completed_orders_sum}"
assert customers["age"].isna().sum() == 0, "There are still missing ages"
assert orders[orders["status"] == "completed"]["delivery_minutes"].isna().sum() == 0, "Missing delivery times in completed orders"
assert orders[orders["status"] == "cancelled"]["delivery_minutes"].isna().sum() == len(orders[orders["status"] == "cancelled"]), "Cancelled orders should have missing delivery times"
assert len(customers["city"].unique()) == 6, "Expected exactly 6 cities"
assert customers["gender"].isin(["male", "female", "unknown"]).all(), "Invalid gender found"
assert "order_id" in orders.columns, "order_id missing from orders"

print("ALL CHECKPOINTS PASSED")
