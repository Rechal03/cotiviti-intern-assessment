# Clinical Natural Language Technology for Health Care
**Cotiviti Intern Assessment — Topic 1**

---

## Overview

This repository contains all deliverables for the Cotiviti Intern Assessment. The topic chosen is Clinical Natural Language Technology for Health Care — covering NLP, OCR, Computer Vision, LLM, and LMM approaches.

The proof of concept demonstrates a full hybrid clinical NLP pipeline:

```
Raw unstructured clinical text  →  Rule-Based Extraction  →  LLM Layer  →  Confidence Score  →  Structured JSON
```

---

## Files

| File | Description |
|------|-------------|
| `APP.py` | Streamlit hybrid pipeline — rule-based + Ollama LLM + confidence scoring |
| `clinical_extractor.py` | Pure rule-based NLP extractor — zero dependencies |
| `report.docx` | Written report — definition, trends, opportunities, threats, recommendations |
| `presentation.pptx` | Slide deck with speaker notes |
| `demo_video.mp4` | Recorded presentation and live demo |

---

## Demo Video

▶️ [Watch on YouTube](YOUR_YOUTUBE_LINK_HERE)

---

## How to Run the Streamlit App

**Requirements:**
- Python 3
- Ollama installed and running locally with a model pulled (e.g. mistral)

**Install dependencies:**
```
pip install streamlit requests
```

**Start Ollama in one terminal:**
```
ollama serve
```

**Run the app in a second terminal:**
```
python -m streamlit run APP.py
```

App opens at `http://localhost:8501`

In the sidebar set the model name to whichever model you have — e.g. `mistral`.

**Sample input to test:**
```
Pt is a 58 y/o male. Dx: Unstable angina. Hx of HTN.
BP 148/92 HR 96 T 98.6 F.
Aspirin 81mg qd, Metoprolol 25mg bid.
F/u in 1 week, cardiology referral.
```

---

## How to Run the Rule-Based Extractor

No dependencies needed — just Python 3.

```
python clinical_extractor.py
```

Interactive mode:
```
python clinical_extractor.py --interactive
```

---

## What the POC Demonstrates

- **Past:** Rule-based NLP using regex patterns and a clinical abbreviation dictionary to extract diagnosis, vitals, medications, and follow-up instructions
- **Present:** LLM-based extraction via Ollama running locally — catches things rules miss like follow-up instructions and medication frequencies
- **Future:** Confidence scoring that evaluates extraction completeness and would route low-confidence cases to human reviewers in production

---

## Strategic Recommendations

Two investments proposed for Cotiviti:

1. **Hybrid Clinical Record Review (CRR) Layer** — combine deterministic rule-based extraction with an LLM layer for novel or ambiguous phrasing, limiting hallucination exposure while retaining flexibility
2. **Human-in-the-Loop Confidence Scoring** — auto-route lower-confidence or higher-dollar-impact claims to human reviewers instead of fully automating decisions

---

## References

See `report.docx` for full bibliography in APA format.
