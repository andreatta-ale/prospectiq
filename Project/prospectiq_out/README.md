# ProspectIQ Demo Dataset (AI-integrated)

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
- Model: gpt-4o-mini
- Language: en
- Mode: offline (stub)

## QA Snapshot
- Avg value by size: {
  "Large": 155274.0,
  "Medium": 34265.6,
  "Small": 18360.0
}
- Cycle range OK: True
