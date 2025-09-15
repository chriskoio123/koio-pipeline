import os
import base64
import time
import random
from typing import Dict, Any, List, Optional, Set

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

# --- Tuning knobs (can be overridden via repo/Actions env vars) ---
MAX_PAGES = int(os.getenv("MAX_PAGES", "5"))            # ticket pages per run
PAGE_LIMIT = int(os.getenv("PAGE_LIMIT", "100"))        # tickets per page
MSG_LIMIT  = int(os.getenv("MSG_LIMIT", "50"))          # messages per ticket call
MSG_SLEEP  = float(os.getenv("MSG_SLEEP", "0.10"))      # base delay between message calls
MAX_MESSAGE_FETCH_PER_RUN = int(os.getenv("MAX_MESSAGE_FETCH_PER_RUN", "120"))  # cap messages fetched/run

# --- Subject CONTAINS blocklist (case-insensitive) ---
SUBJECT_BLOCKLIST_CONTAINS = [
    "report domain:",  # your requested filter
    # "out of office",
    # "auto-reply",
    # "dmca",
]

# --- Init Supabase client ---
sb: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# ------------------------ HTTP helpers ------------------------
def _auth_header(email: str, key: str) -> Dict[str, str]:
    token = base64.b64encode(f"{email}:{key}".encode()).decode()
    return {"Authorization": f"Basic {token}", "accept": "application/json"}

def _request_with_backoff(url: str, params: Dict[str, Any], max_retries: int = 6) -> requests.Response:
    """
    Retry on 429/5xx with exponential backoff + jitter.
    Never raises inside; returns a Response or raises after final retry.
    """
    headers = _auth_header(GORGIAS_EMAIL, GORGIAS_API_KEY)
    backoff = 0.5
    for attempt in range(max_retries):
        r = requests.get(url, headers=headers, params=params, timeout=30)
        if r.status_code < 400:
            return r
        if r.status_code in (429, 500, 502, 503, 504):
            # Honor Retry-After if present
            ra = r.headers.get("Retry-After")
            if ra:
                try:
                    sleep_s = float(ra)
                except ValueError:
                    sleep_s = backoff
            else:
                # exp backoff with jitter
                sleep_s = backoff + random.uniform(0, 0.3)
            time.sleep(sleep_s)
            backoff = min(backoff * 2, 8.0)
            continue
        # Other client errors: break early
        r.raise_for_status()
    # Last attempt
    r.raise_for_status()
    return r  # unreachable, keeps type checkers happy

def _get(url_path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    base = f"https://{GORGIAS_SUBDOMAIN}.gorgias.com"
    url = f"{base}{url_path}"
    r = _request_with_backoff(url, params)
    return r.json()

def gorgias_get_tickets(limit: int, cursor: Optional[str]) -> Dict[str, Any]:
    params = {"limit": limit, "order_by": "created_datetime:desc"}
    if cursor:
        params["cursor"] = cursor
    return _get("/api/tickets", params)

def gorgias_get_messages(ticket_id: int, limit: int = MSG_LIMIT) -> List[Dict[str, Any]]:
    payload = _get("/api/messages", {"ticket_id": ticket_id, "limit": limit})
    return payload.get("data", []) or []
# ---------------------------------------------------------------

# ----------------------- FILTERS -------------------------------
def _subject_blocklisted(subject: str) -> bool:
    subj = (subject or "").lower()
    if not subj:
        return False
    return any(needle and needle.lower() in subj for needle in SUBJECT_BLOCKLIST_CONTAINS)

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
# ---------------------------------------------------------------

def _pick_first_and_latest_customer_messages(msgs: List[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    cust_msgs = [m for m in msgs if not m.get("from_agent")]
    def _txt(m: Dict[str, Any]) -> Optional[str]:
        txt = (m.get("body_text") or "").strip()
        if txt:
            return txt
        html = (m.get("body_html") or "").strip()
        if html:
            import re
            return re.sub("<[^>]+>", " ", html)
        return None
    first_text = _txt(cust_msgs[0]) if cust_msgs else None
    latest_text = _txt(cust_msgs[-1]) if cust_msgs else first_text
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
        "raw": t,
    }

def upsert_tickets(rows: List[Dict[str, Any]]) -> None:
    if rows:
        sb.table("raw_gorgias").upsert(rows, on_conflict="id").execute()

def _ids_needing_messages(ids: List[int]) -> Set[int]:
    """
    Ask Supabase which of these ids already have message text.
    We only fetch messages for tickets where both first/latest are null/empty.
    """
    if not ids:
        return set()
    resp = (
        sb.table("raw_gorgias")
        .select("id, first_customer_message, latest_customer_message")
        .in_("id", ids)
        .execute()
    )
    have = set()
    for row in resp.data or []:
        has_first = bool((row.get("first_customer_message") or "").strip())
        has_latest = bool((row.get("latest_customer_message") or "").strip())
        if has_first or has_latest:
            have.add(int(row["id"]))
    return set(ids) - have

def run():
    total_seen = 0
    total_kept = 0
    total_msg_fetch = 0
    cursor = None

    for _ in range(MAX_PAGES):
        payload = gorgias_get_tickets(limit=PAGE_LIMIT, cursor=cursor)
        data = payload.get("data", []) or []
        meta = payload.get("meta", {}) or {}

        total_seen += len(data)

        # Filter at ticket level
        kept_tickets = [t for t in data if keep_ticket(t)]
        ids = [int(t["id"]) for t in kept_tickets if t.get("id") is not None]

        # Decide which tickets actually need message fetching (skip ones we already enriched)
        need_msg_ids = list(_ids_needing_messages(ids))
        # Respect per-run cap
        remaining_budget = max(0, MAX_MESSAGE_FETCH_PER_RUN - total_msg_fetch)
        msg_targets = set(need_msg_ids[:remaining_budget])

        rows: List[Dict[str, Any]] = []
        for t in kept_tickets:
            tid = int(t["id"])
            first_msg: Optional[str] = None
            latest_msg: Optional[str] = None

            if tid in msg_targets:
                try:
                    msgs = gorgias_get_messages(ticket_id=tid, limit=MSG_LIMIT)
                    picked = _pick_first_and_latest_customer_messages(msgs)
                    first_msg, latest_msg = picked["first"], picked["latest"]
                except Exception as e:
                    # Don't fail the whole run if one ticket's messages error out
                    first_msg = latest_msg = None
                finally:
                    total_msg_fetch += 1
                    # gentle jitter to avoid bursts
                    time.sleep(MSG_SLEEP + random.uniform(0, 0.1))

            # Upsert the ticket row (with or without messages)
            rows.append(normalize_ticket(t, first_msg, latest_msg))

        upsert_tickets(rows)
        total_kept += len(rows)

        cursor = meta.get("next_cursor")
        if not cursor or not data:
            break

        # throttle between ticket pages
        time.sleep(0.3)

    print(
        f"✅ Ingest finished: kept {total_kept} / {total_seen} tickets; "
        f"messages fetched this run: {total_msg_fetch} (cap {MAX_MESSAGE_FETCH_PER_RUN})."
    )

if __name__ == "__main__":
    run()
