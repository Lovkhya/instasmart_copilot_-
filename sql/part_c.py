"""part_c.py -- Support Ticket Triage with Guardrails"""
import os
import csv
import re
import json
import random
import time
import pandas as pd
import mock_llm

def run_part_c():
    os.makedirs("outputs", exist_ok=True)
    mock_llm.reset()
    
    raw_tickets_path = "data/raw/tickets.csv"
    tickets = []
    with open(raw_tickets_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tickets.append(row)

    print(f"Loaded {len(tickets)} raw tickets.")

    # 1. Redact PI regex patterns
    RE_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
    RE_PHONE = re.compile(r"\b[6-9]\d{9}\b")
    RE_ORDER = re.compile(r"ORD-\d{6}")

    redacted_counts = {"email": 0, "phone": 0, "order_id": 0}
    processed_tickets = []

    for t in tickets:
        txt = t["text"]
        # Guardrail checks
        is_empty = (not txt or txt.strip() == "")
        est_tok = mock_llm.est_tokens(txt)
        is_long = est_tok > 400

        # Redaction counts
        emails = RE_EMAIL.findall(txt)
        phones = RE_PHONE.findall(txt)
        orders = RE_ORDER.findall(txt)
        redacted_counts["email"] += len(emails)
        redacted_counts["phone"] += len(phones)
        redacted_counts["order_id"] += len(orders)

        redacted_txt = RE_EMAIL.sub("[EMAIL]", txt)
        redacted_txt = RE_PHONE.sub("[PHONE]", redacted_txt)
        redacted_txt = RE_ORDER.sub("[ORDER_ID]", redacted_txt)

        # Assertion test: re-applying regex finds nothing
        assert len(RE_EMAIL.findall(redacted_txt)) == 0
        assert len(RE_PHONE.findall(redacted_txt)) == 0
        assert len(RE_ORDER.findall(redacted_txt)) == 0

        t["redacted_text"] = redacted_txt
        t["is_empty"] = is_empty
        t["is_long"] = is_long
        processed_tickets.append(t)

    print(f"Redaction counts: {redacted_counts}")

    # 2. Prompts definition
    sys_zero = "CLASSIFY support ticket into one of: billing, delivery, technical, fraud. Reply ONLY as JSON: {\"category\": \"...\", \"reason\": \"...\"}"
    sys_few = """CLASSIFY support ticket into one of: billing, delivery, technical, fraud. Reply ONLY as JSON: {"category": "...", "reason": "..."}
EXAMPLES:
- text: "Someone hacked my account and stole my wallet balance." category: "fraud"
- text: "I was charged twice for order ORD-123456." category: "billing"
"""

    # 3. Resilient call wrapper with backoff
    def call_llm_with_retry(messages):
        retries = 3
        base_delay = 0.05
        for n in range(retries):
            try:
                return mock_llm.chat(messages)
            except mock_llm.TransientLLMError:
                wait = base_delay * (2 ** n) * random.uniform(0.9, 1.15)
                print(f"Retry {n} due to transient error. Waiting {wait:.4f}s")
                time.sleep(wait)
        raise RuntimeError("Max retries exceeded for transient LLM error.")

    # Run Few-Shot Pipeline for final results
    results = []
    total_tokens_running = 0
    token_budget = 100000

    # Load customer segments for routing
    segments_df = pd.read_csv("outputs/customer_segments.csv") if os.path.exists("outputs/customer_segments.csv") else pd.DataFrame()
    premium_customers = set(segments_df[segments_df['is_premium'] == True]['customer_id']) if not segments_df.empty else set()
    seg_map = dict(zip(segments_df['customer_id'], segments_df['segment_name'])) if not segments_df.empty else {}

    for t in processed_tickets:
        tid = t["ticket_id"]
        cid = t["customer_id"]
        true_cat = t["true_category"]
        
        status = ""
        category = ""
        tokens_used = 0

        if t["is_empty"]:
            status = "skipped_empty"
        elif t["is_long"]:
            status = "flagged_long"
        elif total_tokens_running >= token_budget:
            status = "budget_exhausted"
        else:
            messages = [
                {"role": "system", "content": sys_few},
                {"role": "user", "content": t["redacted_text"]}
            ]
            try:
                resp = call_llm_with_retry(messages)
                content = resp.choices[0].message.content
                tokens_used = resp.usage.total_tokens
                total_tokens_running += tokens_used

                # Validate JSON or keyword fallback
                try:
                    parsed = json.loads(content)
                    cat_val = parsed.get("category", "").lower()
                    if cat_val in ["billing", "delivery", "technical", "fraud"]:
                        status = "classified_json"
                        category = cat_val
                    else:
                        status = "flagged_invalid_output"
                except json.JSONDecodeError:
                    words = re.findall(r"[a-z]+", content.lower())
                    allowed = ["billing", "delivery", "technical", "fraud"]
                    matched = [w for w in allowed if w in words]
                    if len(matched) == 1:
                        status = "classified_keyword"
                        category = matched[0]
                    else:
                        status = "flagged_invalid_output"
            except Exception:
                status = "flagged_invalid_output"

        # Routing rules
        segment_name = seg_map.get(cid, "")
        is_prem = cid in premium_customers

        if status in ["skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"]:
            route, priority = "human_review", "NORMAL"
        elif category == "fraud":
            route, priority = "security_team", "HIGH"
        elif is_prem:
            route, priority = "premium_desk", "HIGH"
        else:
            route, priority = "support_queue", "NORMAL"

        results.append({
            "ticket_id": tid,
            "customer_id": cid,
            "status": status,
            "category": category if category else "",
            "segment_name": segment_name,
            "priority": priority,
            "route": route,
            "tokens_used": tokens_used
        })

    res_df = pd.DataFrame(results)
    res_df.to_csv("outputs/triage_results.csv", index=False)

    # Budget demo run
    budget_demo_results = []
    mock_llm.reset()
    demo_budget = 6000
    demo_tokens = 0
    exhausted_count = 0

    for t in processed_tickets:
        if demo_tokens >= demo_budget or t["is_empty"] or t["is_long"]:
            status = "budget_exhausted" if demo_tokens >= demo_budget else ("skipped_empty" if t["is_empty"] else "flagged_long")
            if status == "budget_exhausted": exhausted_count += 1
            budget_demo_results.append({"ticket_id": t["ticket_id"], "status": status})
            continue

        messages = [{"role": "system", "content": sys_zero}, {"role": "user", "content": t["redacted_text"]}]
        try:
            resp = call_llm_with_retry(messages)
            demo_tokens += resp.usage.total_tokens
            budget_demo_results.append({"ticket_id": t["ticket_id"], "status": "classified_json"})
        except:
            budget_demo_results.append({"ticket_id": t["ticket_id"], "status": "error"})

    pd.DataFrame(budget_demo_results).to_csv("outputs/triage_budget_demo.csv", index=False)
    print(f"Part C completed. Budget demo exhausted count: {exhausted_count}")

if __name__ == "__main__":
    run_part_c()
    