import os, time, json
from typing import List, Dict, Any
from supabase import create_client
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

assert SUPABASE_URL and SUPABASE_ANON_KEY, "Missing Supabase env vars"
assert OPENAI_API_KEY, "Missing OPENAI_API_KEY"

sb = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

# Tunables
BATCH = int(os.getenv("AI_BATCH", "60"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")  # 1536-dim
MAX_TXT = int(os.getenv("AI_MAX_CHARS", "8000"))

LABEL_PROMPT = """You are a CX triage assistant. Given the customer's message, return:
- issue_label: short noun phrase from this set if possible: ["Size issue","Return/Exchange","Shipping delay","Defect/Quality","Promo code","Order status","Cancellation","Availability","Wholesale","Billing","Website/Checkout","Other"].
- sentiment: positive|neutral|negative
- priority: integer 1 (low) .. 5 (urgent)
- summary: <= 2 sentences with key detail.

Respond as compact JSON with keys: issue_label, sentiment, priority, summary. Do not include backticks or extra text.
"""

def fetch_batch() -> List[Dict[str, Any]]:
    # Pick recent rows with message text; keep it simple for v1
    q = (
        sb.table("raw_gorgias")
        .select("id, subject, message_for_ai, ai_updated_at")
        .is_("message_for_ai", "not", None)
        .order("updated_datetime", desc=True)
        .limit(BATCH)
        .execute()
    )
    return q.data or []

def embed(texts: List[str]) -> List[List[float]]:
    out = []
    for t in texts:
        t = (t or "")[:MAX_TXT]
        r = client.embeddings.create(model=EMBED_MODEL, input=t)
        out.append(r.data[0].embedding)
        time.sleep(0.03)  # small pacing
    return out

def label(text: str) -> Dict[str, Any]:
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":LABEL_PROMPT},
                  {"role":"user","content":text[:6000]}],
        temperature=0.2
    )
    content = r.choices[0].message.content.strip()
    try:
        return json.loads(content)
    except Exception:
        # Fallback if the model returns prose
        return {"issue_label":"Other","sentiment":"neutral","priority":3,"summary":content[:400]}

def run():
    rows = fetch_batch()
    if not rows:
        print("No rows to enrich.")
        return

    texts = [(r.get("message_for_ai") or "") for r in rows]
    embs = embed(texts)

    updates = []
    for row, emb in zip(rows, embs):
        meta = label(row.get("message_for_ai") or "")
        updates.append({
            "id": row["id"],
            "embedding": emb,
            "ai_issue_label": meta.get("issue_label"),
            "ai_sentiment": meta.get("sentiment"),
            "ai_priority": int(meta.get("priority", 3)),
            "ai_summary": meta.get("summary"),
            "ai_updated_at": "now()"
        })
        time.sleep(0.05)

    # Upsert one-by-one (avoids oversized payloads)
    for u in updates:
        sb.table("raw_gorgias").update(u).eq("id", u["id"]).execute()

    print(f"Enriched {len(updates)} tickets.")

if __name__ == "__main__":
    run()
