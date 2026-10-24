"""part_a.py -- Data Foundation: Clean, Store, Query"""
import os
import sqlite3
import pandas as pd
import numpy as np

def run_part_a():
    print("--- 1. LOADING RAW DATA ---")
    RAW_DIR = "data/raw"
    CLEAN_DIR = "data/clean"
    os.makedirs(CLEAN_DIR, exist_ok=True)
    os.makedirs("sql", exist_ok=True)

    cust_df = pd.read_csv(os.path.join(RAW_DIR, "customers.csv"))
    prod_df = pd.read_csv(os.path.join(RAW_DIR, "products.csv"))
    orders_df = pd.read_csv(os.path.join(RAW_DIR, "orders.csv"))

    print(f"Customers shape: {cust_df.shape}")
    print(f"Products shape: {prod_df.shape}")
    print(f"Orders shape: {orders_df.shape}")

    print("\n--- 2. CLEANING CUSTOMERS ---")
    # Standardise city: strip and title case
    cust_df['city'] = cust_df['city'].str.strip().str.title()
    
    # Map gender
    def clean_gender(g):
        if pd.isna(g) or str(g).strip() == "":
            return "unknown"
        g_lower = str(g).strip().lower()
        if g_lower in ['m', 'male']:
            return "male"
        elif g_lower in ['f', 'female']:
            return "female"
        return "unknown"
    cust_df['gender'] = cust_df['gender'].apply(clean_gender)

    # Clean age: treat ages outside 10-100 as missing, then fill with city median
    cust_df['age'] = pd.to_numeric(cust_df['age'], errors='coerce')
    cust_df.loc[(cust_df['age'] < 10) | (cust_df['age'] > 100), 'age'] = np.nan
    cust_df['age'] = cust_df['age'].fillna(
        cust_df.groupby('city')['age'].transform('median')
    ).round().astype(int)

    # Convert signup_date to datetime
    cust_df['signup_date'] = pd.to_datetime(cust_df['signup_date'])
    print("Customers cleaned successfully.")

    print("\n--- 3. CLEANING ORDERS ---")
    # Convert order_ts to datetime
    orders_df['order_ts'] = pd.to_datetime(orders_df['order_ts'])

    # Remove exact duplicate rows
    initial_orders_len = len(orders_df)
    orders_df = orders_df.drop_duplicates()
    print(f"Removed {initial_orders_len - len(orders_df)} exact duplicate rows.")

    # Keep latest record per order_id (handling pending vs final states)
    orders_df = orders_df.sort_values('order_ts').drop_duplicates(subset=['order_id'], keep='last')

    # Convert order_amount to numeric, turning unparseable into NaN
    orders_df['order_amount'] = pd.to_numeric(orders_df['order_amount'], errors='coerce')

    # Merge with product prices to fill missing amounts
    merged_orders = orders_df.merge(prod_df[['product_id', 'price_inr']], on='product_id', how='left')
    missing_amt_mask = merged_orders['order_amount'].isna()
    computed_amt = (
        merged_orders.loc[missing_amt_mask, 'price_inr'] * 
        merged_orders.loc[missing_amt_mask, 'quantity'] * 
        (1 - merged_orders.loc[missing_amt_mask, 'discount_pct'] / 100.0)
    ).round(2)
    orders_df.loc[missing_amt_mask, 'order_amount'] = computed_amt

    # Negative delivery_minutes to missing
    orders_df.loc[orders_df['delivery_minutes'] < 0, 'delivery_minutes'] = np.nan
    orders_df['delivery_minutes'] = pd.to_numeric(orders_df['delivery_minutes'], errors='coerce')

    # Impute delivery_minutes for completed orders using category median among completed orders
    orders_with_cat = orders_df.merge(prod_df[['product_id', 'category']], on='product_id', how='left')
    completed_mask = orders_with_cat['status'] == 'completed'
    cat_medians = orders_with_cat[completed_mask].groupby('category')['delivery_minutes'].transform('median')
    
    missing_deliv_completed = completed_mask & orders_df['delivery_minutes'].isna()
    orders_df.loc[missing_deliv_completed, 'delivery_minutes'] = cat_medians[missing_deliv_completed].round()
    print("Orders cleaned successfully.")

    print("\n--- 4. STORING CLEAN TABLES AND SQLITE DATABASE ---")
    cust_df.to_csv(os.path.join(CLEAN_DIR, "customers.csv"), index=False)
    prod_df.to_csv(os.path.join(CLEAN_DIR, "products.csv"), index=False)
    orders_df.to_csv(os.path.join(CLEAN_DIR, "orders.csv"), index=False)

    db_path = "data/instamart.db"
    conn = sqlite3.connect(db_path)
    cust_df.to_sql("customers", conn, if_exists="replace", index=False)
    prod_df.to_sql("products", conn, if_exists="replace", index=False)
    # Store order_ts as text YYYY-MM-DD HH:MM:SS
    orders_sql = orders_df.copy()
    orders_sql['order_ts'] = orders_sql['order_ts'].dt.strftime('%Y-%m-%d %H:%M:%S')
    orders_sql.to_sql("orders", conn, if_exists="replace", index=False)
    conn.close()
    print("Stored to data/clean/ and data/instamart.db.")

    print("\n--- 5. VALIDATION CHECKPOINTS ---")
    assert len(orders_df) == 12309, f"Expected 12309 orders, got {len(orders_df)}"
    assert orders_df['order_amount'].isna().sum() == 0, "Missing order amounts found!"
    assert len(cust_df[cust_df['age'].isna()]) == 0, "Missing ages found!"
    print("ALL CHECKPOINTS PASSED")

    print("\n--- 6. SQL ANALYSIS SETUP ---")
    queries = {
        "Q1": "SELECT customer_id FROM orders WHERE status='completed' AND order_amount > 500 GROUP BY customer_id HAVING SUM(order_amount) > 8000;",
        "Q2": "SELECT c.city, p.category, SUM(o.order_amount) as revenue FROM orders o JOIN customers c ON o.customer_id = c.customer_id JOIN products p ON o.product_id = p.product_id WHERE o.status='completed' GROUP BY c.city, p.category;",
        "Q3": "SELECT product_name FROM products WHERE price_inr > (SELECT AVG(price_inr) FROM products);",
        "Q4": "SELECT c.customer_id FROM customers c LEFT JOIN orders o ON c.customer_id = o.customer_id AND o.status='completed' WHERE o.order_id IS NULL;",
        "Q5": "WITH spend AS (SELECT customer_id, SUM(order_amount) as tot FROM orders WHERE status='completed' GROUP BY customer_id), avg_spend AS (SELECT AVG(tot) as avg_tot FROM spend) SELECT COUNT(*) FROM spend, avg_spend WHERE spend.tot > avg_spend.avg_tot;",
        "Q6": "WITH spend AS (SELECT c.customer_id, c.city, SUM(o.order_amount) as total_spend FROM orders o JOIN customers c ON o.customer_id = c.customer_id WHERE o.status='completed' GROUP BY c.customer_id, c.city) SELECT customer_id, total_spend, RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) as rnk FROM spend;",
        "Q7": "SELECT SUBSTR(order_ts, 1, 10) as day, SUM(order_amount) as daily_rev, SUM(SUM(order_amount)) OVER (ORDER BY SUBSTR(order_ts, 1, 10)) as cum_rev FROM orders WHERE status='completed' GROUP BY day;"
    }
    os.makedirs("sql", exist_ok=True)
    with open("sql/queries.sql", "w", encoding="utf-8") as f:
        for qname, qstr in queries.items():
            f.write(f"-- {qname}\n{qstr}\n\n")
    print("Saved queries to sql/queries.sql")

if __name__ == "__main__":
    run_part_a()