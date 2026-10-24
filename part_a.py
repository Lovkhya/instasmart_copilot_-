import os
import sqlite3
import pandas as pd
import numpy as np

# Ensure clean data directory exists
os.makedirs("data/clean", exist_ok=True)

print("--- Starting Part A: Data Cleaning & Storing ---")

# 1. Load Raw Data
customers = pd.read_csv("data/raw/customers.csv")
products = pd.read_csv("data/raw/products.csv")
orders = pd.read_csv("data/raw/orders.csv")

# ----------------------------------------------------
# 2. Clean Customers Table
# ----------------------------------------------------
# (a) Standardize city: remove leading/trailing spaces, apply title case
customers["city"] = customers["city"].astype(str).str.strip().str.title()

# (b) Map gender spellings to male, female, or unknown (blank -> unknown)
def clean_gender(g):
    if pd.isna(g) or str(g).strip() == "":
        return "unknown"
    g_lower = str(g).strip().lower()
    if g_lower in ["m", "male"]:
        return "male"
    elif g_lower in ["f", "female"]:
        return "female"
    return "unknown"

customers["gender"] = customers["gender"].apply(clean_gender)

# (c) Treat ages outside 10-100 as missing, fill missing ages with city median
customers["age"] = pd.to_numeric(customers["age"], errors="coerce")
customers.loc[(customers["age"] < 10) | (customers["age"] > 100), "age"] = np.nan

city_medians = customers.groupby("city")["age"].transform("median")
customers["age"] = customers["age"].fillna(city_medians).round().astype(int)

# (d) Convert signup_date to datetime
customers["signup_date"] = pd.to_datetime(customers["signup_date"])

# ----------------------------------------------------
# 3. Clean Orders Table
# ----------------------------------------------------
# (a) Convert order_ts to datetime
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

# (b) Remove exact duplicate rows
orders = orders.drop_duplicates()

# (c) Keep only the record with the latest order_ts for each order_id
orders = orders.sort_values("order_ts").drop_duplicates(subset=["order_id"], keep="last")

# (d) Convert order_amount to numeric, fill missing from product catalog price
orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
orders = orders.merge(products[["product_id", "price_inr"]], on="product_id", how="left")

missing_amt_mask = orders["order_amount"].isna()
calculated_amt = round(orders["price_inr"] * orders["quantity"] * (1 - orders["discount_pct"] / 100), 2)
orders.loc[missing_amt_mask, "order_amount"] = calculated_amt[missing_amt_mask]
orders = orders.drop(columns=["price_inr"])

# (e) Handle delivery_minutes
orders["delivery_minutes"] = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = np.nan

# Impute completed order delivery times with category median
orders_with_cat = orders.merge(products[["product_id", "category"]], on="product_id", how="left")
cat_medians = orders_with_cat[orders_with_cat["status"] == "completed"].groupby("category")["delivery_minutes"].transform("median")
completed_missing_mask = (orders["status"] == "completed") & (orders["delivery_minutes"].isna())
orders.loc[completed_missing_mask, "delivery_minutes"] = cat_medians[completed_missing_mask]

# ----------------------------------------------------
# 4. Save Clean Tables & Store in SQLite Database
# ----------------------------------------------------
customers.to_csv("data/clean/customers.csv", index=False)
products.to_csv("data/clean/products.csv", index=False)
orders.to_csv("data/clean/orders.csv", index=False)

# Convert order_ts back to text YYYY-MM-DD HH:MM:SS for SQLite storage
orders_db = orders.copy()
orders_db["order_ts"] = orders_db["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")

db_path = "data/instamart.db"
conn = sqlite3.connect(db_path)
customers.to_sql("customers", conn, if_exists="replace", index=False)
products.to_sql("products", conn, if_exists="replace", index=False)
orders_db.to_sql("orders", conn, if_exists="replace", index=False)
conn.close()

print("Clean tables saved and stored in data/instamart.db successfully!")

# ----------------------------------------------------
# 5. Validation Checkpoints Assertions
# ----------------------------------------------------
assert len(orders) == 12309, f"Expected 12309 order rows, got {len(orders)}"
assert len(customers) == 1200, f"Expected 1200 customer rows, got {len(customers)}"
assert orders["order_amount"].isna().sum() == 0, "Missing order amounts found!"
assert len(orders[(orders["status"] == "completed") & (orders["delivery_minutes"].isna())]) == 0, "Missing delivery minutes in completed orders!"
assert len(orders[(orders["status"] == "cancelled") & (~orders["delivery_minutes"].isna())]) == 0, "Cancelled orders should have missing delivery minutes!"
assert customers["age"].between(10, 100).all(), "Ages out of bounds found!"
assert len(customers["city"].unique()) == 6, f"Expected 6 cities, got {len(customers['city'].unique())}"

print("ALL CHECKPOINTS PASSED")
