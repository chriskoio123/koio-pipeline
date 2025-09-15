# Koio Listening Pipeline — Starter (Gorgias → Supabase → Slack)

## Quick Start (10 minutes)

1) **Create Supabase project**
   - Copy `supabase_schema.sql` into Supabase SQL editor → Run.

2) **Create a new private GitHub repo** and upload these files.

3) **Add GitHub Secrets (Settings → Secrets → Actions):**
   - `GORGIAS_EMAIL` = your Gorgias login email
   - `GORGIAS_API_KEY` = your Gorgias API key (Settings → Your profile → REST API)
   - `GORGIAS_SUBDOMAIN` = `koio` (or your subdomain)
   - `SUPABASE_URL` = from Supabase settings
   - `SUPABASE_ANON_KEY` = from Supabase settings
   - `SLACK_WEBHOOK_URL` = Slack Incoming Webhook (optional)

4) **Run workflow now**
   - GitHub → Actions → *Nightly Ingest* → **Run workflow**.
   - It upserts latest tickets to `raw_gorgias` and posts a Slack summary.

## Local run (optional)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.sample .env  # fill values
python scripts/gorgias_ingest.py
```

## Next
- Add Trustpilot, Meta, Reddit, X in `scripts/` similarly.
- Add embeddings + clustering job after ingest (next milestone).
