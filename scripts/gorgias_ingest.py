import os
import base64
import time
from typing import Dict, Any, List

import requests
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

# --- Required env vars (provided via GitHub Secrets in the workflow) ---
GORGIAS_EMAIL = os.getenv("GORGIAS_EMAIL")
GORGIAS_API_KEY = os.getenv("GORGIAS_API_KEY")
GORGIAS_SUBDOMAIN = os.getenv("GORGIAS_SUBDOMAIN", "").strip()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

assert GORGIAS_EMAIL and GORGIAS_API_KEY and GORGIAS_SUBDOMAIN, "Set Gorgias env vars."
assert SUPABASE_URL and SUPABASE_ANON_KEY, "Set Supabase env vars."

# --- Optional tuning knobs ---
MAX_PAGES = int(os.getenv("MAX_PAGES", "5"))        # safety guard per run
PAGE_LIMIT = int(os.getenv("PAGE_LIMIT", "100"))    # Gorgias page size

# --- Subject CONTAINS blocklist (case-insensitive) ---
# Add phrases to exclude anywhere in the subject (not just prefix).
SUBJECT_BLOCKLIST_CONTAINS = [
    "report domain:",  # your requested filter
    # "out of office",
    # "auto-reply",
    # "dmca",
]

# --- Init Supabase client ---
sb: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

def _auth_header(email: str, key: str) -> Dict[str, str]:
    token = base64.b64encode(f"{email}:{key}".encode()).decode()
    return {"Authorization": f"Basic {token}", "accept": "application/json"}

def gorgias_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    base = f"https://{GORGIAS_SUBDOMAIN}.gorgias.com"
    url = f"{base}{path}"
    r = requests.get(url, headers=_auth_header(GORGIAS_EMAIL, GORGIAS_API_KEY), params=params, timeout=30)
    r.raise_for_status()
    return r.json()

# ----------------------- FILTERS -----------------------
def _subject_blocklisted(subject: str) -> bool:
    subj = (subject or "").lower()
    if not subj:
        return False
    for needle in SUBJECT_BLOCKLIST_CONTAINS:
        if needle and needle.lower() in subj:
            return True
    return False

def keep_ticket(t: Dict[str, Any]) -> bool:
    """
    Keep only real, useful tickets:
      - not spam
      - not trashed (trashed_datetime is null)
      - has a customer email
      - subject does NOT contain any blocklisted phrase (case-insensitive)
      - has some content (subject OR excerpt OR messages_count > 0)
    """
    if t.get("spam"):
        return False
    if t.get("trashed_datetime"):
        return False

    cust = t.get("customer") or {}
    email = (cust.get("email") or "").strip()
    if not email:
        return False

    subject = (t.get("subject") or "").strip()
    if _subject_blocklisted(subject):
        return False

    excerpt = (t.get("excerpt") or "").strip()
    if not subject and not excerpt and not (t.get("messages_count") or 0) > 0:
        return False

    return True
# -------------------------------------------------------

def normalize_ticket(t: Dict[str, Any]) -> Dict[str, Any]:
    cust = t.get("customer") or {}
    tags = t.get("tags") or []
    return {
        "id": t.get("id"),
        "status": t.get("status"),
        "channel": t.get("channel"),
        "via": t.get("via"),
        "subject": t.get("subject"),
        "customer_email": cust.get("email"),
        "customer_name": cust.get("name"),
        "created_datetime": t.get("created_datetime"),
        "updated_datetime": t.get("updated_datetime"),
        "last_message_datetime": t.get("last_message_datetime"),
        "tags": tags,
        "raw": t,  # full payload for future analysis
    }

def upsert_tickets(rows: List[Dict[str, Any]]) -> None:
    if rows:
        sb.table("raw_gorgias").upsert(rows, on_conflict="id").execute()

def fetch_latest_batch(limit: int = PAGE_LIMIT, cursor: str = None) -> Dict[str, Any]:
    params = {"limit": limit, "order_by": "created_datetime:desc"}
    if cursor:
        params["cursor"] = cursor
    return gorgias_get("/api/tickets", params)

def run():
    total_seen = 0
    total_kept = 0
    cursor = None

    for _ in range(MAX_PAGES):
        payload = fetch_latest_batch(limit=PAGE_LIMIT, cursor=cursor)
        data = payload.get("data", []) or []
        meta = payload.get("meta", {}) or {}

        total_seen += len(data)

        filtered = [t for t in data if keep_ticket(t)]
        rows = [normalize_ticket(t) for t in filtered]
        upsert_tickets(rows)
        total_kept += len(rows)

        cursor = meta.get("next_cursor")
        if not cursor or not data:
            break

        time.sleep(0.4)  # be gentle with the API

    print(f"✅ Ingest finished: kept {total_kept} / {total_seen} tickets (filters + subject contains blocklist).")

if __name__ == "__main__":
    run()
