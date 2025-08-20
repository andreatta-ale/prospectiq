#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ProspectIQ Demo Dataset Generator (AI-integrated, improved)
-----------------------------------------------------------
- Generates relational, realistic demo data for a ProspectIQ-style environment.
- Uses OpenAI (gpt-4o-mini by default) to generate *all* email threads and meeting notes.
- Business rules ensure consistency (company size -> deal value; timeline ~30-90 days; health drives engagement).
- Exports CSVs and a ZIP bundle.
- Reproducible via RNG seeding (Python + NumPy).
- Optional offline mode (no OpenAI calls) with stubbed content for CI/demos.

Usage:
  pip install pandas openai tenacity tqdm
  export OPENAI_API_KEY="your_key_here"
  python prospectiq_dataset_ai_full.py --companies 8 --seed 42 --outdir ./prospectiq_out
  # or offline:
  python prospectiq_dataset_ai_full.py --companies 4 --offline --outdir ./prospectiq_out_offline
"""

import os
import re
import json
import math
import time
import random
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# tqdm (optional): if not installed, fall back to identity iterator
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

# OpenAI availability check
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False

# =============================
# Defaults / Constants
# =============================
DEFAULT_MODEL = "gpt-4o-mini"   # per user request
DEFAULT_LANG = "en"             # email/notes language
DEAL_STAGES = ["Prospecting", "Qualified", "Demo/Presentation", "Proposal", "Negotiation", "Closed-Won", "Closed-Lost"]
DEAL_HEALTH = ["Positive", "Neutral", "Negative"]
GROWTH_STAGES = ["Startup", "Growth", "Established"]
COMPANY_SIZES = [
    ("Small", (50, 200), (5000, 20000)),
    ("Medium", (200, 1000), (20000, 80000)),
    ("Large", (1000, 5000), (80000, 200000))
]
REP_TIERS = [
    ("Top", 0.15, (8, 12)),
    ("Good", 0.35, (5, 8)),
    ("Average", 0.35, (3, 5)),
    ("Underperformer", 0.15, (1, 3)),
]
INDUSTRIES = {
    "SaaS": {
        "vocab": [
            "scalability", "multi-tenant", "SOC 2", "API rate limits",
            "webhooks", "single sign-on", "data residency", "uptime SLA",
            "integration backlog", "churn risk"
        ],
        "roles": ["CEO", "CTO", "VP Sales", "Sales Manager", "Sales Ops", "Security Lead"],
        "style": "concise, technical, goal-oriented"
    },
    "Healthcare": {
        "vocab": [
            "HIPAA", "PHI", "EHR/EMR", "patient privacy", "BAA contract",
            "audit trail", "clinical workflows", "FHIR", "interoperability", "risk assessment"
        ],
        "roles": ["CIO", "Compliance Officer", "Clinical Ops Manager", "IT Director", "Security Lead", "Procurement"],
        "style": "formal, risk-aware, compliance-driven"
    },
    "Financial Services": {
        "vocab": [
            "SEC reporting", "SOX controls", "KYC/AML", "data lineage",
            "model risk", "governance", "encryption at rest", "latency", "trading desk", "audit readiness"
        ],
        "roles": ["CFO", "Head of Compliance", "Risk Manager", "Data Architect", "IT Director", "VP Finance"],
        "style": "precise, controls-oriented, conservative tone"
    },
    "Manufacturing": {
        "vocab": [
            "ERP integration", "supply chain", "downtime", "OEE", "shop floor",
            "MES", "parts traceability", "quality defects", "cycle time", "predictive maintenance"
        ],
        "roles": ["Plant Manager", "Operations Director", "QA Lead", "IT Manager", "Procurement", "COO"],
        "style": "practical, operational, efficiency-focused"
    }
}

# LLM tuning (bounded for predictability)
DEFAULT_MAX_TOKENS = 600
DEFAULT_TIMEOUT = 60  # seconds

# Required keys for email thread payload
REQUIRED_EMAIL_KEYS = {"subject", "outbound", "inbound", "summary"}

# =============================
# Utility helpers
# =============================
def seed_everything(seed: int):
    """Seed Python and NumPy RNGs for reproducible sampling."""
    random.seed(seed)
    np.random.seed(seed)

def rnd_range(a:int, b:int) -> int:
    return random.randint(a, b)

def pick_weighted(items: List, weights: List[float]):
    return random.choices(items, weights=weights, k=1)[0]

def name_company() -> str:
    prefixes = ["Blue", "Quantum", "Apex", "Pioneer", "Vertex", "Nova", "Atlas", "Aurora", "Horizon", "Summit"]
    suffixes = ["Systems", "Analytics", "Solutions", "Dynamics", "Health", "Finance", "Manufacturing", "Labs", "Networks", "Holdings"]
    return f"{random.choice(prefixes)} {random.choice(suffixes)}"

def random_person_name() -> str:
    first = random.choice(["Alex", "Jordan", "Taylor", "Casey", "Morgan", "Riley", "Sam", "Avery", "Quinn", "Drew", "Jamie", "Skyler"])
    last = random.choice(["Silva", "Souza", "Oliveira", "Santos", "Pereira", "Almeida", "Gomes", "Costa", "Ribeiro", "Carvalho", "Moraes", "Barbosa"])
    return f"{first} {last}"

def seniority_for_role(role: str) -> str:
    executive_roles = ["CEO", "CTO", "CFO", "CIO", "COO", "Head of Compliance", "VP Sales"]
    manager_roles = ["Sales Manager", "Operations Director", "Plant Manager", "Clinical Ops Manager", "IT Director", "Security Lead", "QA Lead", "Procurement", "Sales Ops", "Data Architect"]
    if role in executive_roles:
        return "Executive"
    if role in manager_roles:
        return "Manager"
    return "Staff"

def relationship_for_role(role: str) -> str:
    decision_makers = ["CEO", "CTO", "CFO", "CIO", "COO", "Head of Compliance", "VP Sales", "Operations Director", "Plant Manager"]
    influencers = ["IT Director", "Security Lead", "Clinical Ops Manager", "Sales Manager", "QA Lead", "Data Architect"]
    if role in decision_makers:
        return "Decision Maker"
    if role in influencers:
        return "Influencer"
    return "Gatekeeper"

# =============================
# OpenAI client & prompts
# =============================
def build_openai_client() -> OpenAI:
    """Instantiate the OpenAI client or raise helpful errors."""
    if not _OPENAI_AVAILABLE:
        raise RuntimeError("openai package not available. Run: pip install openai")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in environment.")
    return OpenAI(api_key=api_key)

def ensure_email_payload(d: Dict) -> Dict:
    """Ensure required keys exist in the email thread payload."""
    return {k: d.get(k, "") for k in REQUIRED_EMAIL_KEYS}

@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=1, max=20),
       retry=retry_if_exception_type(Exception))
def chat_json(client: OpenAI, model:str, messages:List[Dict], temperature:float=0.8,
              max_tokens:int=DEFAULT_MAX_TOKENS, timeout:int=DEFAULT_TIMEOUT) -> Dict:
    """
    Request a JSON object from Chat Completions. Use JSON mode if available,
    with a non-greedy fallback parser.
    """
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        response_format={"type": "json_object"},
        max_tokens=max_tokens,
        timeout=timeout
    )
    content = resp.choices[0].message.content
    try:
        return json.loads(content)
    except Exception:
        # Non-greedy JSON block extraction
        m = re.search(r"\{[\s\S]*?\}", content)
        if m:
            return json.loads(m.group(0))
        raise RuntimeError(f"Model did not return valid JSON. Raw: {content[:200]}...")

@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=1, max=20),
       retry=retry_if_exception_type(Exception))
def chat_text(client: OpenAI, model:str, messages:List[Dict], temperature:float=0.7,
              max_tokens:int=DEFAULT_MAX_TOKENS, timeout:int=DEFAULT_TIMEOUT) -> str:
    """Request plain text from Chat Completions."""
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout
    )
    return resp.choices[0].message.content

def prompt_email_thread(industry:str, stage:str, health:str, company_name:str, contact_name:str,
                        tracker_terms:List[str], prior_summary:str, lang:str=DEFAULT_LANG) -> List[Dict]:
    style = INDUSTRIES[industry]["style"]
    if lang == "en":
        sys = f"You are a senior B2B sales rep at ProspectIQ. Write {style} emails."
        user = f"""
Company: {company_name}
Industry: {industry}
Deal stage: {stage}
Deal health: {health}
Primary contact: {contact_name}
Sector keywords to weave naturally (2-3): {', '.join(tracker_terms)}
Context from previous messages (carryover): "{prior_summary or 'N/A'}"

Return a compact JSON with keys:
- subject (string)
- outbound (string)  # the rep's email
- inbound (string)   # the contact's reply
- summary (string)   # 1-2 sentence summary of the thread
Write in English. Keep it concise yet realistic.
"""
    else:
        sys = f"Você é um representante B2B sênior da ProspectIQ. Escreva e-mails {style}."
        user = f"""
Empresa: {company_name}
Indústria: {industry}
Estágio do negócio: {stage}
Saúde do deal: {health}
Contato principal: {contact_name}
Palavras-chave do setor (2-3): {', '.join(tracker_terms)}
Contexto de mensagens anteriores: "{prior_summary or 'N/A'}"

Responda em JSON com chaves: subject, outbound, inbound, summary. Escreva em {lang}.
"""
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": user.strip()}
    ]

def prompt_meeting_notes(industry:str, stage:str, health:str, company_name:str, attendees:List[str],
                         tracker_terms:List[str], prior_emails_summary:str, lang:str=DEFAULT_LANG) -> List[Dict]:
    style = INDUSTRIES[industry]["style"]
    if lang == "en":
        sys = f"You are a meticulous sales rep. Produce {style} meeting notes with realistic details."
        user = f"""
Company: {company_name}
Industry: {industry}
Deal stage: {stage}
Deal health: {health}
Attendees: {', '.join(attendees)}
Recent email summary (context carryover): "{prior_emails_summary or 'N/A'}"
Weave in sector keywords naturally: {', '.join(tracker_terms)}

Write concise meeting notes in English (6-10 sentences), ending with next steps and owners.
"""
    else:
        sys = f"Você é um vendedor meticuloso. Produza atas {style}, com detalhes realistas."
        user = f"""
Empresa: {company_name}
Indústria: {industry}
Estágio: {stage}
Saúde: {health}
Participantes: {', '.join(attendees)}
Resumo recente de e-mails: "{prior_emails_summary or 'N/A'}"
Inclua termos do setor: {', '.join(tracker_terms)}
Escreva em {lang}. 6-10 frases. Termine com próximos passos e responsáveis.
"""
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": user.strip()}
    ]

# =============================
# Entity generators
# =============================
def choose_performance_tier() -> str:
    tiers, probs = zip(*[(t[0], t[1]) for t in REP_TIERS])
    return random.choices(tiers, probs, k=1)[0]

def tier_to_close_range(tier:str) -> Tuple[int,int]:
    for t, _, r in REP_TIERS:
        if t == tier:
            return r
    return (1,3)

def gen_sales_reps(n:int=8) -> pd.DataFrame:
    """Generate sales reps with tiers and expected closes per quarter."""
    reps = []
    for i in range(1, n+1):
        tier = choose_performance_tier()
        close_min, close_max = tier_to_close_range(tier)
        reps.append({
            "rep_id": i,
            "name": random_person_name(),
            "tier": tier,
            "expected_deals_per_quarter_min": close_min,
            "expected_deals_per_quarter_max": close_max
        })
    return pd.DataFrame(reps)

def gen_companies(n:int=12) -> pd.DataFrame:
    """Generate companies with industry, size, and value potential."""
    companies = []
    for i in range(1, n+1):
        industry = random.choice(list(INDUSTRIES.keys()))
        size_name, headcount_range, value_range = random.choice(COMPANY_SIZES)
        headcount = rnd_range(*headcount_range)
        value = rnd_range(*value_range)
        companies.append({
            "company_id": i,
            "name": name_company(),
            "industry": industry,
            "employee_count": headcount,
            "size": size_name,
            "growth_stage": random.choice(GROWTH_STAGES),
            "deal_value_potential": value
        })
    return pd.DataFrame(companies)

def gen_contacts(companies: pd.DataFrame) -> pd.DataFrame:
    """Generate multiple contacts per company with roles and relationships."""
    contacts = []
    cid = 1
    for _, c in companies.iterrows():
        roles = INDUSTRIES[c["industry"]]["roles"]
        k = rnd_range(4, 6)
        picked = [random.choice(roles) for _ in range(k)]
        for role in picked:
            name = random_person_name()
            email = f"{name.split()[0].lower()}.{name.split()[1].lower()}@{c['name'].split()[0].lower()}.com"
            contacts.append({
                "contact_id": cid,
                "company_id": c["company_id"],
                "name": name,
                "job_title": role,
                "seniority": seniority_for_role(role),
                "relationship": relationship_for_role(role),
                "email": email
            })
            cid += 1
    return pd.DataFrame(contacts)

def gen_deals(companies: pd.DataFrame, reps: pd.DataFrame) -> pd.DataFrame:
    """Create deals with stage/health/timeline consistent with business rules."""
    deals = []
    did = 1
    now = datetime.now()
    for _, c in companies.iterrows():
        num_deals = rnd_range(1, 3)
        for _ in range(num_deals):
            stage = random.choice(DEAL_STAGES)
            health_weights = {
                "Prospecting":[0.4,0.45,0.15],"Qualified":[0.35,0.45,0.2],"Demo/Presentation":[0.4,0.45,0.15],
                "Proposal":[0.45,0.4,0.15],"Negotiation":[0.5,0.35,0.15],"Closed-Won":[1,0,0],"Closed-Lost":[0,0,1]
            }
            health = pick_weighted(DEAL_HEALTH, health_weights.get(stage, [0.33,0.34,0.33]))
            cycle_days = rnd_range(30, 90)
            start = now - timedelta(days=rnd_range(40, 110))
            end = start + timedelta(days=cycle_days)
            # force end to be <= now for closed stages
            if stage in ["Closed-Won", "Closed-Lost"] and end > now:
                end = now - timedelta(days=rnd_range(1,5))
            # deterministic random_state for sampling
            rep = reps.sample(1, random_state=np.random.randint(0, 1_000_000)).iloc[0]
            value = int(c["deal_value_potential"] * random.uniform(0.8, 1.2))
            deals.append({
                "deal_id": did,
                "company_id": c["company_id"],
                "rep_id": rep["rep_id"],
                "stage": stage,
                "health": health,
                "timeline_start": start.date().isoformat(),
                "timeline_end": end.date().isoformat(),
                "value": value,
                "probability": {
                    "Prospecting": 0.05, "Qualified": 0.2, "Demo/Presentation": 0.35,
                    "Proposal": 0.6, "Negotiation": 0.75, "Closed-Won": 1.0, "Closed-Lost": 0.0
                }[stage]
            })
            did += 1
    return pd.DataFrame(deals)

# =============================
# Activities & trackers (LLM)
# =============================
def llm_generate_email_thread(client: Optional[OpenAI], model, industry, stage, health, company_name, contact_name, tracker_terms, thread_index, prior_summary, lang=DEFAULT_LANG):
    """Generate an email thread (outbound+inbound) or provide stub if offline."""
    if client is None:
        # offline/stub payload
        return {
            "subject": f"{company_name} • {stage} next steps",
            "outbound": f"[stub {industry}/{health}] Following up on {', '.join(tracker_terms[:2])}. Prior: {prior_summary or 'N/A'}",
            "inbound": "[stub reply] Thanks for the details—let’s review next week.",
            "summary": f"{stage} thread referencing {', '.join(tracker_terms[:2])}"
        }
    messages = prompt_email_thread(industry, stage, health, company_name, contact_name, tracker_terms, prior_summary, lang)
    payload = chat_json(client, model, messages, temperature=0.8)
    return ensure_email_payload(payload)

def llm_generate_meeting_notes(client: Optional[OpenAI], model, industry, stage, health, company_name, attendees, tracker_terms, prior_emails_summary, lang=DEFAULT_LANG):
    """Generate meeting notes or provide stub if offline."""
    if client is None:
        return (f"[stub notes {industry}/{stage}/{health}] Discussed {', '.join(tracker_terms[:3])}. "
                f"Attendees: {', '.join(attendees)}. Next steps assigned.")
    messages = prompt_meeting_notes(industry, stage, health, company_name, attendees, tracker_terms, prior_emails_summary, lang)
    return chat_text(client, model, messages, temperature=0.7)

def gen_activities_and_trackers(client: Optional[OpenAI], model, deals: pd.DataFrame, contacts: pd.DataFrame, companies: pd.DataFrame,
                                lang: str = DEFAULT_LANG) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each deal, synthesize email threads (outbound+inbound) and meeting notes,
    then extract keyword trackers from contents. Trackers are attached to both outbound and inbound messages.
    """
    activities = []
    trackers_rows = []
    aid = 1

    for _, d in tqdm(deals.iterrows(), total=len(deals), desc="Generating activities"):
        comp = companies.loc[companies["company_id"] == d["company_id"]].iloc[0]
        industry = comp["industry"]
        vocab = INDUSTRIES[industry]["vocab"]

        # Prefer Decision Maker / Influencer
        cands = contacts[(contacts["company_id"] == d["company_id"]) & (contacts["relationship"].isin(["Decision Maker", "Influencer"]))]

        if cands.empty:
            cands = contacts[contacts["company_id"] == d["company_id"]]

        # deterministic sampling for contact
        contact = cands.sample(1, random_state=np.random.randint(0, 1_000_000)).iloc[0]

        # Sequence sizing by health
        if d["health"] == "Positive":
            emails_n = rnd_range(3,5)
            meetings_n = rnd_range(1,2)
            response_lag = rnd_range(0,2)
        elif d["health"] == "Neutral":
            emails_n = rnd_range(2,4)
            meetings_n = rnd_range(1,2)
            response_lag = rnd_range(1,3)
        else:
            emails_n = rnd_range(2,3)
            meetings_n = rnd_range(0,1)
            response_lag = rnd_range(3,7)

        start = datetime.fromisoformat(d["timeline_start"])
        end = datetime.fromisoformat(d["timeline_end"])
        spacing_days = max(1, (end - start).days // max(1, (emails_n + meetings_n + 1)))
        prior_summary = ""
        email_threads_summaries = []

        # Emails
        for i in range(emails_n):
            t = start + timedelta(days=i * spacing_days)
            content = llm_generate_email_thread(
                client=client, model=model,
                industry=industry, stage=d["stage"], health=d["health"],
                company_name=comp["name"], contact_name=contact["name"],
                tracker_terms=random.sample(vocab, k=min(3, len(vocab))),
                thread_index=i, prior_summary=prior_summary, lang=lang
            )
            # Outbound
            outbound_id = aid
            activities.append({
                "activity_id": outbound_id, "deal_id": d["deal_id"], "type": "Email-Outbound",
                "timestamp": t.isoformat(), "participants": f"Rep->{contact['name']}",
                "subject": content.get("subject", f"{comp['name']} • {d['stage']} next steps"),
                "content": content.get("outbound", ""),
                "outcome": "Sent"
            })
            aid += 1

            # Inbound (reply)
            reply_time = t + timedelta(days=response_lag)
            inbound_outcome = "Reply" if (d["health"] != "Negative" or random.random() > 0.4) else "No reply"
            inbound_id = aid
            activities.append({
                "activity_id": inbound_id, "deal_id": d["deal_id"], "type": "Email-Inbound",
                "timestamp": reply_time.isoformat(), "participants": f"{contact['name']}->Rep",
                "subject": "Re: " + content.get("subject",""),
                "content": content.get("inbound",""),
                "outcome": inbound_outcome
            })
            aid += 1

            # Trackers from both emails
            combined_low = (content.get("outbound","") + " " + content.get("inbound","")).lower()
            for term in vocab:
                if term.lower() in combined_low:
                    trackers_rows.append({"deal_id": d["deal_id"], "activity_id": outbound_id, "keyword": term, "source": "email-out"})
                    trackers_rows.append({"deal_id": d["deal_id"], "activity_id": inbound_id,  "keyword": term, "source": "email-in"})

            prior_summary = content.get("summary", prior_summary)
            email_threads_summaries.append(content.get("summary",""))

        # Meetings
        for j in range(meetings_n):
            t = start + timedelta(days=(emails_n + j + 1) * spacing_days)
            pool = contacts[(contacts["company_id"] == d["company_id"])]
            take = min(3, len(pool))
            attendees_df = pool.sample(take, random_state=np.random.randint(0, 1_000_000)) if take > 0 else pool
            attendees = attendees_df["name"].tolist()
            notes = llm_generate_meeting_notes(
                client=client, model=model, industry=industry, stage=d["stage"], health=d["health"],
                company_name=comp["name"], attendees=attendees,
                tracker_terms=random.sample(vocab, k=min(4, len(vocab))),
                prior_emails_summary="; ".join([s for s in email_threads_summaries if s][-2:]),
                lang=lang
            )
            activities.append({
                "activity_id": aid, "deal_id": d["deal_id"], "type": "Meeting",
                "timestamp": t.isoformat(), "participants": ", ".join(attendees),
                "subject": f"{d['stage']} discussion",
                "content": notes,
                "outcome": (
                    "Agreed next steps" if d["health"] == "Positive" else
                    "Pending decision" if d["health"] == "Neutral" else
                    "No commitment"
                )
            })
            # trackers in notes
            low_notes = (notes or "").lower()
            for term in vocab:
                if term.lower() in low_notes:
                    trackers_rows.append({"deal_id": d["deal_id"], "activity_id": aid, "keyword": term, "source": "meeting"})
            aid += 1

    activities_df = pd.DataFrame(activities)
    trackers_df = pd.DataFrame(trackers_rows)
    return activities_df, trackers_df

# =============================
# QA / sanity checks
# =============================
def attach_cycle_days(deals: pd.DataFrame) -> pd.DataFrame:
    """Compute and attach cycle_days = timeline_end - timeline_start."""
    def _cycle(row):
        s = datetime.fromisoformat(row["timeline_start"])
        e = datetime.fromisoformat(row["timeline_end"])
        return (e - s).days
    deals = deals.copy()
    deals["cycle_days"] = deals.apply(_cycle, axis=1)
    return deals

def qa_snapshot(deals: pd.DataFrame, companies: pd.DataFrame) -> Dict:
    """Return quick quality metrics for the generated dataset."""
    merged = deals.merge(companies[["company_id", "size"]], on="company_id")
    avg_by_size = merged.groupby("size")["value"].mean().to_dict()
    cycle_ok = deals["cycle_days"].between(30, 120).all()
    return {"avg_value_by_size": avg_by_size, "cycle_range_ok": cycle_ok}

# =============================
# Orchestration and CLI
# =============================
def main():
    parser = argparse.ArgumentParser(description="ProspectIQ AI-integrated dataset generator")
    parser.add_argument("--companies", type=int, default=12, help="Number of companies")
    parser.add_argument("--reps", type=int, default=10, help="Number of sales reps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="OpenAI model (e.g., gpt-4o-mini)")
    parser.add_argument("--lang", type=str, default=DEFAULT_LANG, help="Language for content (en/pt/...)")
    parser.add_argument("--outdir", type=str, default="./prospectiq_ai_full_out", help="Output directory")
    parser.add_argument("--offline", action="store_true", help="Do not call OpenAI; synthesize stub text.")
    args = parser.parse_args()

    # Seed RNGs
    seed_everything(args.seed)

    # Build OpenAI client (unless offline)
    client = None
    if not args.offline:
        client = build_openai_client()

    # Generate entities
    reps_df = gen_sales_reps(n=args.reps)
    companies_df = gen_companies(n=args.companies)
    contacts_df = gen_contacts(companies_df)
    deals_df = gen_deals(companies_df, reps_df)
    deals_df = attach_cycle_days(deals_df)

    # Activities + trackers via LLM or offline stub
    activities_df, trackers_df = gen_activities_and_trackers(
        client=client, model=args.model, deals=deals_df, contacts=contacts_df, companies=companies_df, lang=args.lang
    )

    # Sanity checks
    snap = qa_snapshot(deals_df, companies_df)

    # Save outputs
    os.makedirs(args.outdir, exist_ok=True)

    def save_csv(df, name):
        path = os.path.join(args.outdir, f"{name}.csv")
        df.to_csv(path, index=False, encoding="utf-8")
        return path

    files = {}
    files["sales_reps"] = save_csv(reps_df, "sales_reps")
    files["companies"]  = save_csv(companies_df, "companies")
    files["contacts"]   = save_csv(contacts_df, "contacts")
    files["deals"]      = save_csv(deals_df.drop(columns=["cycle_days"]), "deals")
    files["activities"] = save_csv(activities_df, "activities")
    files["trackers"]   = save_csv(trackers_df, "trackers")

    readme = f"""# ProspectIQ Demo Dataset (AI-integrated)

## Entities
- companies.csv
- contacts.csv
- sales_reps.csv
- deals.csv
- activities.csv
- trackers.csv

## Relationships
companies (1) -> (N) contacts
companies (1) -> (N) deals
deals (1) -> (N) activities

## Business Rules
- Larger company sizes tend to have higher deal values.
- Deal cycles ~30–90 days; closed stages end before 'today'.
- Health guides engagement: Positive → more replies/meetings; Negative → fewer.

## LLM
- Model: {args.model}
- Language: {args.lang}
- Mode: {"offline (stub)" if args.offline else "online (OpenAI)"}

## QA Snapshot
- Avg value by size: {json.dumps(snap['avg_value_by_size'], indent=2)}
- Cycle range OK: {snap['cycle_range_ok']}
"""
    with open(os.path.join(args.outdir, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)

    # Zip bundle
    import zipfile
    zip_path = os.path.join(args.outdir, "prospectiq_ai_full_bundle.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        for label, path in files.items():
            z.write(path, arcname=os.path.basename(path))
        z.write(os.path.join(args.outdir, "README.md"), arcname="README.md")

    print("Done! Bundle:", zip_path)

if __name__ == "__main__":
    main()

