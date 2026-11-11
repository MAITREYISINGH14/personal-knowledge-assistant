# Version change log (ground truth for temporal questions)

Kept **outside** `data/corpus/` on purpose, so the system can't retrieve the answers from here.
Every change below was applied automatically and checked, so this list is exact.

## rental_agreement (v1: 15 Dec 2025 → v2: 1 Nov 2026, renewal)

| Section | v1 | v2 |
|---|---|---|
| 3. Term | 1 Jan 2026 – 30 Nov 2026 | 1 Dec 2026 – 31 Oct 2027 |
| 4. Licence Fee | ₹28,000 per month | ₹30,240 per month (+8%) |
| 4. Licence Fee | Late fee ₹500 per day after the 10th | Late fee ₹750 per day after the 10th |
| 7. Parking → Pets | Parking slot B-12 included | **Parking section deleted.** New **Pets** section: one small pet (cat, or dog under 15 kg), ₹10,000 refundable pet deposit |
| 9. Repairs | **Rewritten.** Minor repairs by Licensee, major (plumbing, wiring, structural, appliances) by Licensor | Repairs under ₹2,000 by Licensee; ₹2,000 or more by Licensor unless misuse; quote needed first |
| 10. Termination | One month's written notice | Two months' written notice |
| 11. Renewal | Increase up to 5% | Increase up to 8% |
| Unchanged | Security deposit ₹84,000, refund within 30 days; lock-in 3 months; quiet hours | — |

## pka_project_proposal (v1: 6 Oct 2026, draft → v2: 10 Nov 2026, approved)

| Section | v1 | v2 |
|---|---|---|
| Status | Draft for review | Approved |
| Technology | all-MiniLM-L6-v2; Chroma | bge-base-en (MiniLM as fallback); Qdrant in Docker |
| Timeline | 10 weeks | 12 weeks (adds automation week and hardening weeks 11–12) |
| Budget | ₹3,000 per month | ₹5,000 per month (judge grades every run) |
| Evaluation | **Rewritten.** Target hit@5 ≥ 0.80 | 30–50 questions, hit@5 + MRR + LLM judge; target hit@5 ≥ 0.85, no type below 0.75 |
| Stretch Goals → Risks | Mobile app stretch goal | **Mobile app deleted.** New **Risks** section: chunking gain may be small, API cost of re-summarising, scanned PDFs |

## remote_work_policy (v1: effective 1 Apr 2026 → v2: effective 1 Oct 2026)

| Section | v1 | v2 |
|---|---|---|
| 3. Allowance | Up to 2 remote days per week | Up to 3 remote days per week |
| 5. Office Days → Core Hours | Fridays in office mandatory | **Friday rule deleted.** New **Core Hours**: 11:00–16:00 IST |
| 6. Equipment | **Rewritten.** Laptop; monitor on request, subject to stock | Laptop + 24-inch monitor for everyone; one-time ₹8,000 ergonomic stipend (receipt within 90 days) |
| 7. Internet | ₹1,000 per month | ₹1,500 per month |
| Unchanged | 3-month probation; 48-hour advance request; no working abroad; VPN rules | — |

## q3_marketing_plan (v1: 20 Jun 2026 → v2: 25 Aug 2026, mid-quarter revision)

| Section | v1 | v2 |
|---|---|---|
| Objective | Target 1,200 new subscribers | Target 1,000 (after a slower July) |
| Budget | ₹4,00,000: paid social 60%, search 25%, influencers 15% | ₹3,20,000: paid social 55%, search 30%, webinars and email 15% |
| Launch | 15 July 2026 | Launched 1 August 2026 (packaging delay) |
| Channels | Influencer programme: 15 creators, discount codes | **Influencers deleted.** New **webinars and email**: fortnightly Saturday brewing class, 20% first-box discount |
| Unchanged | CPA target under ₹330; 70% retention target; Monday reviews | — |

## Ready-made temporal question ideas

- current: "What is my current monthly rent?" → ₹30,240 (rental_agreement v2)
- as_of: "What was the rent under the agreement as of June 2026?" → ₹28,000 (v1, as_of 2026-06-30)
- change: "What changed in the repairs rule?" → cost-based ₹2,000 threshold replaced the minor/major split (v1 + v2)
- current: "How many WFH days does Northwind allow?" → 3 (v2)
- as_of: "What was the Q3 marketing budget in July?" → ₹4,00,000 (v1, as_of 2026-07-31)
- change: "Why and how did the Q3 plan change?" → budget cut, launch delay, influencers replaced by webinars (v1 + v2)
- change: "What changed in the PKA proposal's tech choices?" → MiniLM/Chroma to bge-base/Qdrant (v1 + v2)
