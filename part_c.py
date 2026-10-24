import os
import re
import json
import random
import math
import time
import pandas as pd
from mock_llm import chat, reset, est_tokens, TransientLLMError

os.makedirs("outputs", exist_ok=True)
print("--- Starting Part C: Support Ticket Triage ---")

# Load raw tickets
tickets_df = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)

# Load customer segments for routing (if available)
try:
    segments_df = pd.read_csv("outputs/customer_segments.csv")
    segment_map = segments_df.set_index("customer_id").to_dict(orient="index")
except Exception:
    segment_map = {}

# 1. Input Guardrails
MAX_INPUT_TOKENS = 400

def apply_input_guardrails(text):
    if not text or not str(text).strip():
        return "skipped_empty"
    if est_tokens(text) > MAX_INPUT_TOKENS:
        return "flagged_long"
    return "ok"

# 2. Redaction Function
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_RE = re.compile(r"\b[6-9]\d{9}\b")
ORDER_RE = re.compile(r"ORD-\d{6}")

def redact_text(text):
    t = EMAIL_RE.sub("[EMAIL]", text)
    t = PHONE_RE.sub("[PHONE]", t)
    t = ORDER_RE.sub("[ORDER_ID]", t)
    return t

# 3. Resilient Call with Retries, Backoff, and Jitter
BASE_DELAY = 0.05

def resilient_chat(messages):
    for n in range(4): # up to 3 retries
        try:
            return chat(messages)
        except TransientLLMError as e:
            if n == 3:
                raise e
            wait = BASE_DELAY * (2 ** n) * random.uniform(0.9, 1.15)
            # print(f"Retry {n+1} after error: {e}, waiting {wait:.4f}s")
            time.sleep(wait)

# 4. Prompts
ZERO_SHOT_SYSTEM = (
    "CLASSIFY support tickets into one of four categories: billing, delivery, technical, fraud. "
    "Reply with valid JSON only in the form {\"category\": \"...\", \"reason\": \"...\"}."
)

FEW_SHOT_SYSTEM = (
    "CLASSIFY support tickets into one of four categories: billing, delivery, technical, fraud. "
    "Reply with valid JSON only in the form {\"category\": \"...\", \"reason\": \"...\".\n"
    "EXAMPLES:\n"
    "Text: There is an unauthorized transaction on my order ORD-123456. -> {\"category\": \"fraud\", \"reason\": \"unauthorized transaction\"}\n"
    "Text: My order has not arrived yet. -> {\"category\": \"delivery\", \"reason\": \"late delivery\"}"
)

ALLOWED_CATS = {"billing", "delivery", "technical", "fraud"}

def validate_and_parse(response_content, raw_text):
    try:
        data = json.loads(response_content)
        cat = data.get("category")
        if cat in ALLOWED_CATS:
            return "classified_json", cat
    except json.JSONDecodeError:
        pass
    
    # Keyword fallback extraction using re.findall
    words = re.findall(r"[a-z]+", raw_text.lower())
    matched_cats = [c for c in ALLOWED_CATS if c in words]
    if len(matched_cats) == 1:
        return "classified_keyword", matched_cats[0]
        
    return "flagged_invalid_output", ""

# Run Pipeline Function
def run_triage_pipeline(prompt_type="few-shot", token_budget=100000):
    reset()
    system_prompt = FEW_SHOT_SYSTEM if prompt_type == "few-shot" else ZERO_SHOT_SYSTEM
    
    results = []
    running_tokens = 0
    
    for idx, row in tickets_df.iterrows():
        tid = row["ticket_id"]
        cid = row["customer_id"]
        text = row["text"]
        true_cat = row["true_category"]
        
        guard_status = apply_input_guardrails(text)
        if guard_status != "ok":
            results.append({
                "ticket_id": tid, "customer_id": cid, "status": guard_status,
                "category": "", "segment_name": "", "priority": "NORMAL", "route": "human_review", "tokens_used": 0
            })
            continue
            
        redacted = redact_text(text)
        
        # Token Budget Check
        if running_tokens >= token_budget:
            results.append({
                "ticket_id": tid, "customer_id": cid, "status": "budget_exhausted",
                "category": "", "segment_name": "", "priority": "NORMAL", "route": "human_review", "tokens_used": running_tokens
            })
            continue
            
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": redacted}
        ]
        
        try:
            resp = resilient_chat(messages)
            toks = resp.usage.total_tokens
            running_tokens += toks
            
            status, cat = validate_and_parse(resp.choices[0].message.content, resp.choices[0].message.content)
        except Exception:
            status, cat, toks = "flagged_invalid_output", "", 0
            
        # Determine Routing and Priority based on rules
        seg_info = segment_map.get(cid, {})
        seg_name = seg_info.get("segment_name", "")
        is_prem = seg_info.get("is_premium", False)
        
        if status in ["skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"]:
            route, priority = "human_review", "NORMAL"
        elif cat == "fraud":
            route, priority = "security_team", "HIGH"
        elif is_prem:
            route, priority = "premium_desk", "HIGH"
        else:
            route, priority = "support_queue", "NORMAL"
            
        results.append({
            "ticket_id": tid, "customer_id": cid, "status": status,
            "category": cat, "segment_name": seg_name, "priority": priority,
            "route": route, "tokens_used": toks
        })
        
    return pd.DataFrame(results), running_tokens

# Run Budget Demo (Zero-shot with budget 6000)
budget_df, _ = run_triage_pipeline(prompt_type="zero-shot", token_budget=6000)
budget_df.to_csv("outputs/triage_budget_demo.csv", index=False)
exhausted_count = (budget_df["status"] == "budget_exhausted").sum()
print(f"Budget Demo Complete: {exhausted_count} tickets ended budget_exhausted.")

# Run Final Production Triage (Few-shot)
final_df, total_toks = run_triage_pipeline(prompt_type="few-shot", token_budget=100000)
final_df.to_csv("outputs/triage_results.csv", index=False)
print(f"Final Triage Complete: Saved outputs/triage_results.csv (Total Tokens: {total_toks})")
