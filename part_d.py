import os
import re
import json
import time
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from mock_llm import chat, reset, est_tokens, TransientLLMError

os.makedirs("outputs", exist_ok=True)
os.makedirs("state", exist_ok=True)

print("--- Starting Part D: Policy RAG & Refund Agent ---")

# 1. Chunk Documents (40 words, 10 overlap)
POL_DIR = "data/policies"
chunks = []

for fn in sorted(os.listdir(POL_DIR)):
    if fn.endswith(".txt"):
        path = os.path.join(POL_DIR, fn)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        words = text.split()
        start = 0
        while start < len(words):
            end = min(start + 40, len(words))
            chunk_text = " ".join(words[start:end])
            chunks.append({"source": fn, "text": chunk_text})
            if end == len(words):
                break
            start += 30 # 40 - 10 overlap

print(f"Total Policy Chunks Created: {len(chunks)}")

# 2. Vectorize and Retrieve
corpus = [c["text"] for c in chunks]
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(corpus)

def retrieve(query, k=3):
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, X_tfidf).flatten()
    top_indices = sims.argsort()[::-1][:k]
    results = []
    for idx in top_indices:
        results.append({
            "source": chunks[idx]["source"],
            "score": float(sims[idx]),
            "text": chunks[idx]["text"]
        })
    return results

# 3. Augment and Answer (RAG Assistant)
def answer_policy(question):
    retrieved = retrieve(question, k=3)
    context_str = "\n".join([f"[{r['source']}] {r['text']}" for r in retrieved])
    messages = [
        {"role": "system", "content": "POLICY ASSISTANT: Answer only from the supplied context. If you cannot find the information, say you could not find it."},
        {"role": "user", "content": f"CONTEXT:\n{context_str}\n\nQUESTION:\n{question}"}
    ]
    resp = chat(messages)
    return resp.choices[0].message.content, [r["source"] for r in retrieved]

# 4. Define Tools and Tool Map
DB_PATH = "data/instamart.db"

def get_order(order_id):
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?"
    df = pd.read_sql_query(query, conn, params=(order_id,))
    conn.close()
    if df.empty:
        return {"error": "order not found"}
    row = df.iloc[0].to_dict()
    # Handle possible NaN for delivery minutes
    if pd.isna(row["delivery_minutes"]):
        row["delivery_minutes"] = None
    else:
        row["delivery_minutes"] = int(row["delivery_minutes"])
    return row

def search_policy(query):
    res = retrieve(query, k=3)
    return {"results": res}

def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes is None or delivery_minutes > 45:
        return {"refund_inr": float(order_amount)}
    return {"refund_inr": 0.0}

def issue_refund(order_id, refund_inr):
    record = {"timestamp": datetime.now().isoformat(), "action": "ISSUE_REFUND", "order_id": order_id, "refund_inr": float(refund_inr)}
    with open("outputs/refund_ledger.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return {"refund_inr": float(refund_inr)}

def escalate_to_human(order_id, refund_inr):
    record = {"timestamp": datetime.now().isoformat(), "action": "ESCALATE", "order_id": order_id, "refund_inr": float(refund_inr)}
    with open("outputs/refund_ledger.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return {"refund_inr": float(refund_inr)}

TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}

TOOLS_SPEC = [
    {"name": "get_order", "description": "Get order details by order_id.", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}}},
    {"name": "search_policy", "description": "Search policy documents.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}},
    {"name": "calculate_refund", "description": "Calculate refund amount based on order amount and delivery minutes.", "parameters": {"type": "object", "properties": {"order_amount": {"type": "number"}, "delivery_minutes": {"type": "integer"}}}},
    {"name": "issue_refund", "description": "Issue refund for an order.", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}}},
    {"name": "escalate_to_human", "description": "Escalate refund above INR 1000 to human lead.", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}}}
]

# 5. ReAct Agent Loop with Guardrails & Checkpoints
def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10, run_id=None):
    start_time = time.monotonic()
    messages = [
        {"role": "system", "content": "AGENT: You are a helpful refund assistant. Use tools to resolve customer requests."},
        {"role": "user", "content": user_message}
    ]
    
    steps = []
    tokens_used = 0
    llm_calls = 0
    executed_tools = []
    
    for it in range(max_iterations):
        if time.monotonic() - start_time >= time_limit_seconds:
            return {"status": "FALLBACK_TIMEOUT", "final_answer": "Timeout reached before completion.", "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used}
            
        try:
            resp = chat(messages, tools=TOOLS_SPEC)
            llm_calls += 1
            toks = resp.usage.total_tokens
            tokens_used += toks
            
            if tokens_used > token_limit:
                return {"status": "FALLBACK_TOKEN_LIMIT", "final_answer": "Token limit exceeded.", "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used}
                
        except Exception as e:
            return {"status": "FALLBACK_ERROR", "final_answer": str(e), "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used}
            
        choice = resp.choices[0]
        msg = choice.message
        
        if not msg.tool_calls:
            # Final answer reached
            final_ans = msg.content
            status = final_ans.split(":")[0].strip().upper()
            if status not in ["REFUND", "ESCALATE", "DENIED", "NOT_FOUND"]:
                status = "FALLBACK"
            return {"status": status, "final_answer": final_ans, "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used}
            
        # Append assistant message with tool calls
        tc_dicts = [{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in msg.tool_calls]
        messages.append({"role": "assistant", "content": None, "tool_calls": tc_dicts})
        
        for tc in msg.tool_calls:
            t_name = tc.name
            t_args = json.loads(tc.arguments) if isinstance(tc.arguments, str) else tc.arguments
            
            # Execute tool
            func = TOOL_MAP.get(t_name)
            if func:
                obs = func(**t_args)
            else:
                obs = {"error": "unknown tool"}
                
            executed_tools.append(t_name)
            obs_str = json.dumps(obs)
            
            steps.append({
                "tool": t_name,
                "arguments": t_args,
                "observation": obs
            })
            
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": t_name,
                "content": obs_str
            })
            
            # Save Checkpoint if run_id provided
            if run_id:
                ckpt = {
                    "run_id": run_id, "scenario_id": scenario_id, "messages": messages,
                    "tokens_used": tokens_used, "step_count": len(steps), "executed_tools": executed_tools
                }
                with open(f"state/{run_id}.json", "w", encoding="utf-8") as f:
                    json.dump(ckpt, f)
                    
    return {"status": "FALLBACK_MAX_ITERATIONS", "final_answer": "Max iterations reached.", "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used}

# 6. Run Scenarios from agent_scenarios.csv
scenarios_df = pd.read_csv("data/raw/agent_scenarios.csv")
agent_runs_records = []

for idx, row in scenarios_df.iterrows():
    sid = row["scenario_id"]
    msg = row["message"]
    res = run_agent(msg, sid)
    
    run_record = {
        "scenario_id": sid,
        "status": res["status"],
        "final_answer": res["final_answer"],
        "llm_calls": res["llm_calls"],
        "tokens_used": res["tokens_used"],
        "steps": res["steps"]
    }
    agent_runs_records.append(run_record)
    with open("outputs/agent_runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(run_record) + "\n")

print("Agent Scenarios Execution Complete. Saved outputs/agent_runs.jsonl and refund ledger.")
