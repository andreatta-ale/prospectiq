# ProspectIQ – AI-Integrated Demo Dataset Generator

Generate a **realistic, relational** B2B sales dataset (ProspectIQ-style) with LLM-written email threads and meeting notes. Outputs are easy-to-consume CSVs plus a ZIP bundle for demos or analytics.

---

## Contents
- `prospectiq_dataset_ai_full.py` — main generator (entities, deals, LLM activities, trackers)
- `requirements.txt` — Python deps (`pandas`, `openai`, `tenacity`)
- `Dockerfile` — container image for reproducible runs
- `Makefile` — shortcuts for build/run/shell
- `prospectiq_out/` — default output folder (created on run)

---

## Quick Start

### Option A — Python (local)
```bash
pip install -r requirements.txt
export OPENAI_API_KEY="YOUR_KEY"
python prospectiq_dataset_ai_full.py   --companies 8 --reps 10 --seed 42   --model gpt-4o-mini --lang en --outdir ./prospectiq_out
```

### Option B — Docker
```bash
docker build -t prospectiq-generator .
docker run --rm   -e OPENAI_API_KEY=$OPENAI_API_KEY   -v "$PWD":/app   prospectiq-generator   --companies 8 --reps 10 --seed 42 --model gpt-4o-mini --lang en --outdir ./prospectiq_out
```

### Option C — Makefile
```bash
make build
OPENAI_API_KEY=$OPENAI_API_KEY make run
# Optional interactive shell for debugging:
OPENAI_API_KEY=$OPENAI_API_KEY make shell
```

> **Security:** never commit or paste real API keys in code or version control.

---

## What You Get (Outputs)

All files are written to `--outdir` (default: `./prospectiq_out`):

- `companies.csv` — company_id, name, industry, size, employee_count, growth_stage, deal_value_potential  
- `contacts.csv` — contact_id, company_id, name, job_title, seniority, relationship, email  
- `sales_reps.csv` — rep_id, name, tier, expected_deals_per_quarter_min/max  
- `deals.csv` — deal_id, company_id, rep_id, stage, health, timeline_start/end, value, probability  
- `activities.csv` — activity_id, deal_id, type (Email-Outbound/Email-Inbound/Meeting), timestamp, participants, subject, content, outcome  
- `trackers.csv` — deal_id, activity_id, keyword, source (email/meeting)  
- `README.md` — short run snapshot (model/lang/QA)  
- `prospectiq_ai_full_bundle.zip` — all CSVs + README compacted

---

## Configuration (CLI Flags)

```text
--companies  int   Number of companies (default: 12)
--reps       int   Number of sales reps (default: 10)
--seed       int   Random seed for structure reproducibility (default: 42)
--model      str   OpenAI model (e.g., gpt-4o-mini)
--lang       str   Content language for emails/notes (e.g., en, pt)
--outdir     str   Output directory (default: ./prospectiq_ai_full_out)
```

Environment:
- `OPENAI_API_KEY` (required)

---

## How It Works (short)
- Generates entities: reps, companies, contacts, deals (with business rules).
- Uses OpenAI to draft **email threads** (outbound + simulated reply) and **meeting notes** per deal.
- Extracts **industry keywords** from generated text into `trackers.csv`.
- Performs a small QA snapshot and exports everything to CSVs + ZIP.

---

## Troubleshooting
- **Auth error / 401**: ensure `OPENAI_API_KEY` is set in your environment/shell.
- **Rate/timeout**: the script retries LLM calls with exponential backoff (`tenacity`); rerun or reduce `--companies`.
- **Cost/latency too high**: lower `--companies` and/or pick a cheaper/faster `--model`.

---

## License