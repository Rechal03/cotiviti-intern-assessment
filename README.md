# Clinical Note Structured Data Extractor

**Cotiviti Intern Assessment — Hackathon Proof of Concept**
**Topic 1: Clinical Natural Language Technology for Health Care**

## What This Demonstrates

This proof of concept shows the first-principle pipeline behind clinical NLP:

```
Raw, unstructured clinical text  -->  Structured, machine-readable data
```

Healthcare payment integrity and analytics — Cotiviti's core business — depends
on being able to turn free-text clinical documentation (physician notes,
discharge summaries) into structured fields that can be searched, audited,
and analyzed at scale. This demo shows that conversion happening on a
realistic (fictional) clinical note.

## How It Works

The script combines two classic NLP techniques:

1. **Regex-based pattern extraction** — identifies structured patterns like
   medication dosages (`Metformin 500mg BID`), vital signs (`BP 140/90`),
   and section markers (`Dx:`, `F/u:`).
2. **A clinical abbreviation dictionary** — maps medical shorthand (`BID`,
   `T2DM`, `c/o`) to plain-language equivalents, since general-purpose NLP
   tools have no built-in medical vocabulary.

This is a **rule-based** approach rather than a trained machine-learning
model — a deliberate choice for this POC. It requires no API key, no GPU,
and no internet connection, so it runs anywhere and cannot fail during a
live demo. It also mirrors how real clinical NLP systems are built in
practice: rule-based extraction layers are commonly used alongside ML/LLM
models, since they're fast, deterministic, and fully auditable.

## Running the Demo

Requires only Python 3 — no installation, no dependencies.

**Run with the built-in sample note:**
```bash
python3 clinical_extractor.py
```

**Run in interactive mode** (type or paste your own note):
```bash
python3 clinical_extractor.py --interactive
```

## Example

**Input** (raw clinical note):
```
Pt is a 58 y/o male. C/o CP and SOB x3 days, worsening w/ exertion.
Dx: Unstable angina. Hx of HTN and T2DM.
BP 148/92, HR 96, T 98.6 F.
Started ASA 81mg qd, Metoprolol 25mg bid, Atorvastatin 40mg qhs.
F/u in 1 week, repeat EKG, cardiology referral.
```

**Output** (structured record):
```json
{
  "diagnosis": ["Unstable angina"],
  "medications": [
    {"drug": "ASA", "dose": "81mg", "frequency": "once daily"},
    {"drug": "Metoprolol", "dose": "25mg", "frequency": "twice daily"},
    {"drug": "Atorvastatin", "dose": "40mg", "frequency": "at bedtime"}
  ],
  "vitals": {
    "blood_pressure": "148/92",
    "heart_rate": "96 bpm",
    "temperature": "98.6 F"
  },
  "follow_up": ["in 1 week, repeat EKG, cardiology referral"]
}
```

## Future Direction

This POC intentionally scopes to rule-based extraction for speed and
reliability. A production system would extend this with a trained
biomedical NER model (e.g., a clinical BERT variant) or an LLM to catch
novel phrasing the hand-written rules don't anticipate — discussed further
in the accompanying written report.

## Files

- `clinical_extractor.py` — the full extraction pipeline and demo
- `report.docx` — written report on Clinical NLP for Health Care
- `presentation.pptx` — slide presentation
- `demo_video.mp4` — recorded presentation + live demo walkthrough
