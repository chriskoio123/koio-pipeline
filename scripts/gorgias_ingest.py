import os
import base64
import time
from typing import Dict, Any, List, Optional

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

# --- Tuning knobs ---
MAX_PAGES = int(os.getenv("MAX_PAGES", "5"))          # safety guard per run
PAGE_LIMIT = int(os.getenv("PAGE_LIMIT", "100"))      # Gorgias page size
MSG_LIMIT  = int(os.getenv("MSG_LIMIT", "50"))        # messages per ticket fetch
MSG_SLEEP  = float(os.getenv("MSG_SLEEP", "0.15"))    # throttle between message calls (seconds)

# --- Subject CONTAINS blocklist (case-insensitive) ---
SUBJECT_BLOCKLIST_CONTAINS = [
    "report domain:",   # your requested filter
    # "out of office",
    # "auto-reply",
    # "dmca",
]

# --- Init Supabase client ---
sb: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

def _auth_header(email: str, key: str) -> Dict[str, str]:
    token = base64.b64encode(f"{email}:{key}".encode()).decode()
    return {"Authorization": f"Basic {token}", "accept": "application/json"}

def _get(url_path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    base = f"https://{GORGIAS_SUBDOMAIN}.gorgias.com"
    url = f"{base}{url_path}"
    r = requests.get(url, headers=_auth_header(GORGIAS_EMAIL, GORGIAS_API_KEY), params=params, timeout=30)
    if r.status_code == 429:
        # basic backoff on rate limit
        time.sleep(1.0)
        r = requests.get(url, headers=_auth_header(GORGIAS_EMAIL, GORGIAS_API_KEY), params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def gorgias_get_tickets(limit: int, cursor: Optional[str]) -> Dict[str, Any]:
    params = {"limit": limit, "order_by": "created_datetime:desc"}
    if cursor:
        params["cursor"] = cursor
    return _get("/api/tickets", params)

def gorgias_get_messages(ticket_id: int, limit: int = MSG_LIMIT) -> List[Dict[str, Any]]:
    payload = _get("/api/messages", {"ticket_id": ticket_id, "limit": limit})
    return payload.get("data", []) or []

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

def _pick_first_and_latest_customer_messages(msgs: List[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    """
    From a list of messages, return plain text of:
    - first customer-authored message
    - latest customer-authored message
    """
    cust_msgs = [m for m in msgs if not m.get("from_agent")]
    first_text = None
    latest_text = None

    def _text(m: Dict[str, Any]) -> Optional[str]:
        # Prefer body_text, fall back to body_html if needed
        txt = (m.get("body_text") or "").strip()
        if txt:
            return txt
        html = (m.get("body_html") or "").strip()
        if html:
            # naive HTML strip (quick + dependency-free)
            import re
            return re.sub("<[^>]+>", " ", html)
        return None

    if cust_msgs:
        first_text = _text(cust_msgs[0]) or None
        latest_text = _text(cust_msgs[-1]) or first_text

    return {"first": first_text, "latest": latest_text}

def normalize_ticket(t: Dict[str, Any], first_msg: Optional[str], latest_msg: Optional[str]) -> Dict[str, Any]:
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
        "first_customer_message": first_msg,
        "latest_customer_message": latest_msg,
        "raw": t,  # full payload for future analysis
    }

def upsert_tickets(rows: List[Dict[str, Any]]) -> None:
    if rows:
        sb.table("raw_gorgias").upsert(rows, on_conflict="id").execute()

def run():
    total_seen = 0
    total_kept = 0
    cursor = None

    for _ in range(MAX_PAGES):
        payload = gorgias_get_tickets(limit=PAGE_LIMIT, cursor=cursor)
        data = payload.get("data", []) or []
        meta = payload.get("meta", {}) or {}

        total_seen += len(data)

        rows: List[Dict[str, Any]] = []
        for t in data:
            if not keep_ticket(t):
                continue

            # Fetch messages for this ticket (customer text for classification)
            msgs = gorgias_get_messages(ticket_id=t["id"], limit=MSG_LIMIT)
            picked = _pick_first_and_latest_customer_messages(msgs)
            row = normalize_ticket(t, picked["first"], picked["latest"])
            rows.append(row)

            # gentle throttle between message calls
            time.sleep(MSG_SLEEP)

        upsert_tickets(rows)
        total_kept += len(rows)

        cursor = meta.get("next_cursor")
        if not cursor or not data:
            break

        # throttle between ticket pages
        time.sleep(0.4)

    print(f"✅ Ingest finished: kept {total_kept} / {total_seen} tickets; messages fetched and stored (first/latest customer messages).")

if __name__ == "__main__":
    run()
