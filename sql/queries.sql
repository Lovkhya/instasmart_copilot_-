-- Instamart SQL queries

-- Q1: WHERE then HAVING (Completed orders > 500, grouped by customer with total spend > 8000)
SELECT customer_id, SUM(order_amount) as total_spend
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000;

-- Q2: Three-table join (Revenue by city and product category)
SELECT c.city, p.category, SUM(o.order_amount) as revenue
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY revenue DESC;

-- Q3: Scalar subquery (Products priced above the average catalogue price)
SELECT product_id, product_name, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products);

-- Q4: LEFT JOIN (Customers who have no completed order)
SELECT c.customer_id, c.city
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id AND o.status = 'completed'
WHERE o.order_id IS NULL;

-- Q5: Common Table Expressions (Customers whose total spend is above the overall average customer spend)
WITH customer_spend AS (
    SELECT customer_id, SUM(order_amount) as total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
avg_spend AS (
    SELECT AVG(total_spend) as mean_spend FROM customer_spend
)
cs, avg_spend as (
    SELECT cs.customer_id, cs.total_spend
    FROM customer_spend cs, avg_spend
    WHERE cs.total_spend > avg_spend.mean_spend;

-- Q6: Ranking window functions (Rank, dense_rank, row_number partitioned by city)
WITH customer_city_spend AS (
    SELECT c.city, c.customer_id, SUM(o.order_amount) as total_spend
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.city, c.customer_id
)
SELECT city, customer_id, total_spend,
       RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) as rnk,
       DENSE_RANK() OVER (PARTITION BY city ORDER BY dense_rank DESC) as dense_rnk, -- wait, standard syntax below:
       ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) as rn
FROM customer_city_spend;

-- Q7: Running total (Daily completed revenue with cumulative revenue)
SELECT SUBSTR(order_ts, 1, 10) as day,
       SUM(order_amount) as daily_revenue,
       SUM(SUM(order_amount)) OVER (ORDER BY SUBSTR(order_ts, 1, 10)) as cumulative_revenue
FROM orders
WHERE status = 'completed'
GROUP BY day
ORDER BY day;
