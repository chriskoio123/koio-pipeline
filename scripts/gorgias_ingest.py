import os
import base64
import time
import json
from datetime import datetime, timezone
from typing import Dict, Any, List
import requests
from dotenv import load_dotenv

# Supabase client (v2)
from supabase import create_client, Client

load_dotenv()

GORGIAS_EMAIL = os.getenv("GORGIAS_EMAIL")
GORGIAS_API_KEY = os.getenv("GORGIAS_API_KEY")
GORGIAS_SUBDOMAIN = os.getenv("GORGIAS_SUBDOMAIN", "").strip()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")

assert GORGIAS_EMAIL and GORGIAS_API_KEY and GORGIAS_SUBDOMAIN, "Set Gorgias env vars."
assert SUPABASE_URL and SUPABASE_ANON_KEY, "Set Supabase env vars."

sb: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

def auth_header(email: str, key: str) -> Dict[str, str]:
    token = base64.b64encode(f"{email}:{key}".encode()).decode()
    return {
        "Authorization": f"Basic {token}",
        "accept": "application/json",
    }

def gorgias_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    base = f"https://{GORGIAS_SUBDOMAIN}.gorgias.com"
    url = f"{base}{path}"
    r = requests.get(url, headers=auth_header(GORGIAS_EMAIL, GORGIAS_API_KEY), params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def upsert_tickets(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    # Upsert on primary key 'id'
    sb.table("raw_gorgias").upsert(rows, on_conflict="id").execute()

def slack_post(text: str):
    if not SLACK_WEBHOOK_URL:
        return
    try:
        requests.post(SLACK_WEBHOOK_URL, json={"text": text}, timeout=10)
    except Exception:
        pass

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
        "raw": t,
    }

def fetch_latest_batch(limit: int = 100, cursor: str = None) -> Dict[str, Any]:
    params = {"limit": limit, "order_by": "created_datetime:desc"}
    if cursor:
        params["cursor"] = cursor
    return gorgias_get("/api/tickets", params)

def run():
    total = 0
    cursor = None
    max_pages = 5  # safety guard for first run
    for _ in range(max_pages):
        payload = fetch_latest_batch(limit=100, cursor=cursor)
        data = payload.get("data", [])
        meta = payload.get("meta", {})
        rows = [normalize_ticket(t) for t in data]
        upsert_tickets(rows)
        total += len(rows)
        cursor = meta.get("next_cursor")
        if not cursor or len(data) == 0:
            break
        time.sleep(0.5)  # be nice to the API
    slack_post(f"✅ Gorgias ingest complete: {total} tickets upserted.")

if __name__ == "__main__":
    run()
