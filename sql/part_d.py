"""part_d.py -- Policy RAG Assistant and Refund Agent"""
import os
import csv
import json
import time
import math
import glob
import pandas as pd
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mock_llm

def run_part_d():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("state", exist_ok=True)
    mock_llm.reset()

    print("--- 1. CHUNKING POLICY DOCUMENTS ---")
    POL_DIR = "data/policies"
    chunks = []
    
    for fn in os.listdir(POL_DIR):
        if fn.endswith(".txt"):
            path = os.path.join(POL_DIR, fn)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            words = text.split()
            step = 30
            chunk_size = 40
            for i in range(0, len(words), step):
                chunk_words = words[i:i + chunk_size]
                if not chunk_words:
                    break
                chunks.append({"source": fn, "text": " ".join(chunk_words)})
                if i + chunk_size >= len(words):
                    break

    print(f"Created {len(chunks)} chunks across policy documents.")

    print("\n--- 2. VECTORISING & RETRIEVAL SETUP ---")
    corpus = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    def retrieve(query, k=3):
        q_vec = vectorizer.transform([query])
        sims = cosine_similarity(q_vec, tfidf_matrix).flatten()
        top_indices = sims.argsort()[::-1][:k]
        results = []
        for idx in top_indices:
            results.append({
                "source": chunks[idx]["source"],
                "score": float(sims[idx]),
                "text": chunks[idx]["text"]
            })
        return results

    print("Retrieval function ready.")

    print("\n--- 3. DEFINING AGENT TOOLS ---")
    def get_order(order_id):
        conn = sqlite3.connect("data/instamart.db")
        cursor = conn.cursor()
        cursor.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?", (order_id,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            return {"error": "order not found"}
        return {
            "order_id": row[0],
            "status": row[1],
            "order_amount": row[2],
            "delivery_minutes": row[3] if row[3] is not None else 0
        }

    def search_policy(query):
        res = retrieve(query, k=3)
        return {"results": res}

    def calculate_refund(order_amount, delivery_minutes):
        if delivery_minutes > 45:
            return {"refund_inr": float(order_amount)}
        return {"refund_inr": 0.0}

    def issue_refund(order_id, refund_inr):
        ledger_entry = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "tool": "issue_refund", "order_id": order_id, "refund_inr": refund_inr}
        with open("outputs/refund_ledger.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(ledger_entry) + "\n")
        return {"refund_inr": refund_inr}

    def escalate_to_human(order_id, refund_inr):
        ledger_entry = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "tool": "escalate_to_human", "order_id": order_id, "refund_inr": refund_inr}
        with open("outputs/refund_ledger.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(ledger_entry) + "\n")
        return {"refund_inr": refund_inr}

    TOOL_MAP = {
        "get_order": get_order,
        "search_policy": search_policy,
        "calculate_refund": calculate_refund,
        "issue_refund": issue_refund,
        "escalate_to_human": escalate_to_human
    }

    print("\n--- 4. RUNNING AGENT SCENARIOS ---")
    scenarios_path = "data/raw/agent_scenarios.csv"
    scenarios = []
    with open(scenarios_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scenarios.append(row)

    def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10):
        start_time = time.monotonic()
        messages = [
            {"role": "system", "content": "AGENT: You are an autonomous refund assistant that calls tools."},
            {"role": "user", "content": user_message}
        ]
        tokens_used = 0
        llm_calls = 0
        steps = []
        executed_tools = []

        for it in range(max_iterations):
            if (time.monotonic() - start_time) >= time_limit_seconds:
                return {"status": "FALLBACK_TIMEOUT", "final_answer": "Timeout reached", "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used}

            try:
                resp = mock_llm.chat(messages, tools=list(TOOL_MAP.keys()))
                llm_calls += 1
                tokens_used += resp.usage.total_tokens

                if tokens_used > token_limit:
                    return {"status": "FALLBACK_TOKEN_LIMIT", "final_answer": "Token limit exceeded", "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used}

                msg = resp.choices[0].message
                if not msg.tool_calls:
                    final_ans = msg.content
                    status = final_ans.split(":")[0].strip() if ":" in final_ans else "UNKNOWN"
                    return {"status": status, "final_answer": final_ans, "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used}

                # Handle tool calls
                tool_call = msg.tool_calls[0]
                t_name = tool_call.name
                t_args = json.loads(tool_call.arguments) if isinstance(tool_call.arguments, str) else tool_call.arguments
                
                messages.append({"role": "assistant", "content": None, "tool_calls": [tool_call]})

                # Execute tool
                func = TOOL_MAP.get(t_name)
                obs = func(**t_args)
                obs_str = json.dumps(obs)
                executed_tools.append(t_name)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": t_name,
                    "content": obs_str
                })

                steps.append({"tool": t_name, "arguments": t_args, "observation": obs})

                # Checkpoint persistence for crash/resume demo
                if scenario_id == "S1" and len(executed_tools) == 2:
                    ckpt = {
                        "run_id": "run_s1", "scenario_id": scenario_id, "messages": messages,
                        "tokens_used": tokens_used, "step_number": len(steps), "executed_tools": executed_tools
                    }
                    with open("state/run_s1.json", "w", encoding="utf-8") as ckpt_f:
                        json.dump(ckpt, ckpt_f)

            except Exception as e:
                return {"status": "ERROR", "final_answer": str(e), "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used}

        return {"status": "FALLBACK_MAX_ITERATIONS", "final_answer": "Max iterations reached", "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used}

    agent_runs = []
    for sc in scenarios:
        res = run_agent(sc["message"], sc["scenario_id"])
        agent_runs.append({
            "scenario_id": sc["scenario_id"],
            "status": res["status"],
            "final_answer": res["final_answer"],
            "llm_calls": res["llm_calls"],
            "tokens_used": res["tokens_used"],
            "steps": res["steps"]
        })

    with open("outputs/agent_runs.jsonl", "w", encoding="utf-8") as f:
        for r in agent_runs:
            f.write(json.dumps(r) + "\n")

    print("Part D completed successfully. All outputs generated.")

if __name__ == "__main__":
    run_part_d()