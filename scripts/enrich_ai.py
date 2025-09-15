import os, time, json, sys, traceback
from typing import List, Dict, Any
from supabase import create_client
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
# Prefer service role for CI updates (bypasses RLS); fall back to anon if not set.
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def die(msg: str):
    print(msg, file=sys.stderr)
    sys.exit(1)

if not SUPABASE_URL: die("Missing SUPABASE_URL")
if not SUPABASE_KEY: die("Missing SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_ANON_KEY)")
if not OPENAI_API_KEY: die("Missing OPENAI_API_KEY")

sb = create_client(SUPABASE_URL, SUPABASE_KEY)
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
        time.sleep(0.03)
    return out

def label(text: str) -> Dict[str, Any]:
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":LABEL_PROMPT},
                  {"role":"user","content":(text or "")[:6000]}],
        temperature=0.2
    )
    content = (r.choices[0].message.content or "").strip()
    try:
        return json.loads(content)
    except Exception:
        return {"issue_label":"Other","sentiment":"neutral","priority":3,"summary":content[:400]}

def run():
    try:
        rows = fetch_batch()
        if not rows:
            print("No rows to enrich.")
            return

        texts = [(r.get("message_for_ai") or "") for r in rows]
        embs = embed(texts)

        for row, emb in zip(rows, embs):
            meta = label(row.get("message_for_ai") or "")
            update = {
                "embedding": emb,
                "ai_issue_label": meta.get("issue_label"),
                "ai_sentiment": meta.get("sentiment"),
                "ai_priority": int(meta.get("priority", 3)),
                "ai_summary": meta.get("summary"),
                "ai_updated_at": "now()"
            }
            sb.table("raw_gorgias").update(update).eq("id", row["id"]).execute()

        print(f"Enriched {len(rows)} tickets.")
    except Exception as e:
        print("ERROR during enrichment:", e)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run()
