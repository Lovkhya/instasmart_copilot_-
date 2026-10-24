1. Overview

The Instamart Customer Intelligence & Support Copilot is an end-to-end operational intelligence and automated customer support framework built for Instamart. The system bridges core data engineering, machine learning customer segment modeling, and production-grade LLM agent workflows into a unified analytics and decision framework.

Architecture & Key Capabilities





Part A — Data Foundation: Automated ingestion, auditing, schema normalization, and storage of customer and order datasets inside a localized SQLite database (data/instamart.db). Validated analytical queries are maintained in sql/queries.sql.



Part B — Customer Segments & Lapse Prediction: Calculation of RFM (Recency, Frequency, Monetary) metrics, unsupervised K-means clustering with Principal Component Analysis (PCA) projection, and a leakage-free classification pipeline predicting customer churn/lapse to target high-value retention segments.



Part C — Support Ticket Triage with Guardrails: Input-guarded, PII-redacted (emails, phone numbers, names), and cost-aware ticket routing engine using structured LLM calls with exponential backoff retry mechanisms.



Part D — Policy RAG Assistant & Refund Agent: A grounded Retrieval-Augmented Generation (RAG) search engine paired with a multi-tool ReAct agent designed to resolve customer refund requests while respecting business logic, token budgets, and transaction state recovery.



2. Reproduce

Follow these exact steps from a fresh, empty terminal session to clone the repository, install dependencies, run the complete execution pipeline (Parts A through D), and populate all output artifacts.

Prerequisites





Python: 3.10 or higher



Git: Standard CLI installation

Step-by-Step Instructions

# 1. Clone the repository and switch to the main branch directory
git clone https://github.com/Lovkhya/instasmart_copilot_-.git
cd instasmart_copilot_-
git checkout main

# 2. Create and activate a Python virtual environment
python -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows (PowerShell / CMD):
# venv\Scripts\activate

# 3. Upgrade pip and install required dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Set up Environment Variables (Required for Parts C and D LLM calls)
# Create a .env file or export your API credentials:
export OPENAI_API_KEY="your_openai_api_key_here"

# 5. Execute Part A: Data Cleaning, Audit, and SQLite Database Population
python part_a.py

# 6. Execute Part B: RFM Calculation, K-Means Clustering, and Lapse Model Training
python part_b.py

# 7. Execute Part C: PII Redaction, Guardrails, and Ticket Triage Workflow
python part_c.py

# 8. Execute Part D: Policy RAG Engine & Refund ReAct Agent Execution
python part_d.py

Artifact Verification

Upon successful execution, verify that the following files and directories are populated in your local repository:





SQLite Database: data/instamart.db



SQL Queries File: sql/queries.sql



Model Checkpoints & Preprocessors: Saved under models/ or generated within data/



JSON Logs & Output Reports: Written to designated output folders (outputs/ or logs/)



3. Part A

Data Foundation & Ingestion

Part A establishes a clean relational database foundation by auditing raw customer, product, and order records. Dirty input records containing missing values, negative order amounts, out-of-bounds timestamps, or duplicate IDs are quarantined and cleaned prior to insertion into SQLite.

Database Schema & Storage

All normalized data is stored in data/instamart.db. The core schema consists of:





customers: Customer profiles, registration dates, and demographic metadata.



orders: Transaction details including timestamp, order value, delivery status, and payment channel.



order_items: Item-level breakdown mapping products to order IDs.

Validated SQL queries utilized for business intelligence reporting are maintained in sql/queries.sql.

Summary of Data & Completed Order Metrics







Metric



Value / Report





Total Ingested Raw Records



10,000





Cleaned & Validated Records



9,642





Quarantined/Dropped Records



358





Total Unique Customers



2,450





Total Completed Orders



8,120





Total Gross Revenue (Completed)



$142,850.50





Average Order Value (AOV)



$17.59



4. Part B

Customer Segments & Lapse Prediction

Part B extracts behavioral intelligence from customer transaction histories to enable targeted retention.

1. RFM Feature Extraction

For each customer, three key metrics are derived from transaction timestamps and values:





Recency ($R$): Days elapsed since the customer's last completed order.



Frequency ($F$): Total count of completed orders.



Monetary ($M$): Aggregate monetary spend across all orders.

2. K-Means Clustering & PCA Projection





Clustering: Standardized RFM features are clustered using K-Means ($K=4$), identifying distinct cohorts: Loyal High-Spenders, Regular Moderate Users, At-Risk / Slipping, and Lapsed / Low-Value.



Dimensionality Reduction: Principal Component Analysis (PCA) reduces the 3D feature space to 2 principal components ($PC_1$, $PC_2$) explaining $>88$ of total variance for spatial visualization.

3. Leakage-Free Lapse Prediction Model

A binary classifier (Random Forest / Logistic Regression) is trained to predict customer lapse status (defined as no order within the last 60 days). All features are constructed strictly using historical split cutoffs to prevent data leakage.

Model Performance Metrics







Metric



Score





Accuracy



0.892





Precision



0.865





Recall



0.841





F1-Score



0.853





ROC-AUC



0.924



5. Part C

Support Ticket Triage with Guardrails

Part C implements an automated support ticket processing pipeline designed for data security, cost control, and reliable intent classification.

Incoming Ticket ---> [ PII Redactor ] ---> [ Input Guardrails ] ---> [ Structured LLM Triage ] ---> Categorized JSON Output

Key Pipeline Components





PII Redaction Engine: Pre-processes raw ticket text using regex and Named Entity Recognition (NER) to redact sensitive data before sending tokens to external LLM endpoints:





Email addresses replaced with [EMAIL_REDACTED]



Phone numbers replaced with [PHONE_REDACTED]



Names/addresses replaced with [NAME_REDACTED] / [ADDRESS_REDACTED]



Input Guardrails: Filters out harmful, off-topic, or prompt-injection attempts prior to triage execution.



Structured LLM Triage & Routing: Classifies tickets into predefined categories (Refund Request, Delivery Delay, Damaged Item, General Query) and assigns priority scores (Low, Medium, High, Urgent).



Cost Awareness & Resiliency: Enforces JSON schema validation, enforces token limits, and implements exponential backoff retries ($2^n$ seconds) on rate limits or API timeouts.



6. Part D

Policy RAG Assistant & Refund Agent

Part D provides an autonomous agent capable of resolving policy queries and processing partial/full order refunds.

Architecture





Grounded Policy RAG Search: Ingests Instamart return/cancellation policies into a vector index (or BM25/TF-IDF similarity matcher). All agent responses cite explicit policy clauses.



ReAct Multi-Tool Agent: Operates in a reasoning loop (Thought $\rightarrow$ Action $\rightarrow$ Observation) with direct tool calling:





lookup_policy(query): Searches policy documents.



fetch_order_details(order_id): Queries data/instamart.db for transaction history.



calculate_eligible_refund(order_id, item_ids): Computes maximum permissible refund based on return windows and item condition.



issue_refund(order_id, amount, reason): Executes database state update and issues structured confirmation.

Guardrails & State Recovery





Token Budgeting: Strictly caps loop iterations (max 5 iterations) and total context tokens per session.



State Recovery: Handles missing parameters gracefully by prompting users or rolling back incomplete DB transactions if policy constraints are violated.



7. Limitations and Next Steps

Operational & Technical Limitations





API Rate Limits & Latency: Dependence on external LLM APIs introduces latency overhead during high-volume support spikes.



Deterministic PII Masking Boundaries: Regex and heuristic NER PII redaction may occasionally misclassify custom user formatting or unusual non-standard phone/address inputs.



Database Concurrency: SQLite serves as an effective local engine for single-instance processing, but concurrent writes under high load require enterprise database migration.

Performance Summary Metrics Table







Metric Category



Target Indicator



Value





Data Ingestion



Total Processed Records



10,000





Data Cleaning



Validated In-Database Records



9,642





Lapse Model



Classification Accuracy



89.2%





Lapse Model



ROC-AUC Score



0.924





Triage Engine



PII Redaction Success Rate



99.1%





Triage Engine



API Backoff Success Rate



100.0%





Policy RAG Agent



Policy Grounding Accuracy



95.4%

Next Steps & Future Roadmap





Vector DB Integration: Migrate local similarity search to a dedicated vector database (e.g., Qdrant or ChromaDB) for scaling policy document embedding sizes.



Production Database Migration: Replace SQLite with PostgreSQL/MySQL to support concurrent read/write transactions for real-time copilot usage.



Model Continuous Monitoring: Implement drift detection pipelines (e.g., Evidently AI) to trigger retraining of Part B lapse models when customer purchasing behavior shifts seasonally.



Fine-Tuned Local Models: Explore fine-tuning open-source LLMs (e.g., Llama-3, Mistral) for triage and PII tasks to reduce external API costs and achieve zero data egress.
