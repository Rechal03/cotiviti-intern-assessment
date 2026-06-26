"""
Clinical Note Structured Data Extractor
-----------------------------------------
A proof-of-concept demonstrating rule-based Natural Language Processing (NLP)
applied to unstructured clinical documentation.

PURPOSE:
Healthcare organizations like Cotiviti process massive volumes of unstructured
clinical text (physician notes, discharge summaries, claims narratives).
Before that text can be analyzed, audited, or used for payment integrity
decisions, it must first be converted into structured, machine-readable data.

This script demonstrates the FIRST PRINCIPLE behind that pipeline:
    raw unstructured text  -->  structured fields (diagnosis, meds, vitals, follow-up)

APPROACH:
This POC uses rule-based pattern matching (regular expressions + a clinical
abbreviation dictionary) rather than a trained ML model. This is intentional:
  1. It requires zero external dependencies or API keys, so it runs anywhere,
     reliably, with no risk of failure during a live demo.
  2. It mirrors how real clinical NLP pipelines actually work in production --
     rule-based extraction layers are commonly used alongside ML/LLM models
     because they are fast, deterministic, auditable, and don't require
     a network call or GPU.
  3. It cleanly demonstrates the "first principles" of clinical NLP: tokenizing
     text, recognizing domain-specific abbreviations, and mapping informal
     shorthand into structured, standardized output.

In production, this rule-based layer would typically be paired with a
trained clinical NER model (e.g., a biomedical BERT variant) or an LLM to
catch novel phrasing the rules don't anticipate -- that hybrid approach is
discussed further in the accompanying written report.
"""

import re
import json


# ---------------------------------------------------------------------------
# STEP 1: Domain knowledge -- clinical abbreviation dictionary
# ---------------------------------------------------------------------------
# Real clinical notes are full of shorthand. A general-purpose NLP tool with
# no medical training data will not know that "BID" means "twice daily."
# This dictionary encodes that domain-specific knowledge.

FREQUENCY_ABBREVIATIONS = {
    "qd": "once daily",
    "od": "once daily",
    "bid": "twice daily",
    "tid": "three times daily",
    "qid": "four times daily",
    "qhs": "at bedtime",
    "prn": "as needed",
    "q4h": "every 4 hours",
    "q6h": "every 6 hours",
    "q8h": "every 8 hours",
    "q12h": "every 12 hours",
}

GENERAL_ABBREVIATIONS = {
    "pt": "patient",
    "c/o": "complains of",
    "dx": "diagnosis",
    "hx": "history",
    "tx": "treatment",
    "f/u": "follow-up",
    "rx": "prescription",
    "htn": "hypertension",
    "t2dm": "type 2 diabetes mellitus",
    "sob": "shortness of breath",
    "cp": "chest pain",
    "bp": "blood pressure",
    "hr": "heart rate",
    "wt": "weight",
    "y/o": "years old",
    "w/": "with",
}


# ---------------------------------------------------------------------------
# STEP 2: Extraction functions -- regex-based pattern matching
# ---------------------------------------------------------------------------

def extract_diagnosis(text: str) -> list:
    """Find text following 'Dx:' or 'Diagnosis:' markers."""
    pattern = r"(?:Dx|Diagnosis)\s*:\s*([^.\n]+)"
    matches = re.findall(pattern, text, re.IGNORECASE)
    return [m.strip() for m in matches]


def extract_medications(text: str) -> list:
    """
    Find medication mentions in the form:
        DrugName <dose><unit> <frequency_abbreviation>
    e.g. 'Metformin 500mg BID' or 'ASA 81mg daily'
    """
    pattern = r"\b([A-Z][a-zA-Z]+)\s+(\d+\.?\d*)\s*(mg|mcg|g|ml|units?)\b\s*([A-Za-z0-9/]*)"
    matches = re.findall(pattern, text)

    meds = []
    for drug, dose, unit, freq_raw in matches:
        freq_key = freq_raw.lower().strip(".,")
        frequency = FREQUENCY_ABBREVIATIONS.get(freq_key, freq_raw if freq_raw else "not specified")
        meds.append({
            "drug": drug,
            "dose": f"{dose}{unit}",
            "frequency": frequency
        })
    return meds


def extract_vitals(text: str) -> dict:
    """Pull out common vital sign patterns."""
    vitals = {}

    bp_match = re.search(r"\bBP\s*(\d{2,3}/\d{2,3})", text, re.IGNORECASE)
    if bp_match:
        vitals["blood_pressure"] = bp_match.group(1)

    hr_match = re.search(r"\bHR\s*(\d{2,3})", text, re.IGNORECASE)
    if hr_match:
        vitals["heart_rate"] = f"{hr_match.group(1)} bpm"

    temp_match = re.search(r"\bT(?:emp)?\s*(\d{2,3}\.?\d*)\s*F?\b", text, re.IGNORECASE)
    if temp_match:
        vitals["temperature"] = f"{temp_match.group(1)} F"

    return vitals


def extract_followup(text: str) -> list:
    """Find follow-up instructions following 'F/u' markers."""
    pattern = r"(?:F/u|Follow[- ]?up)\s*:?\s*([^.\n]+)"
    matches = re.findall(pattern, text, re.IGNORECASE)
    return [m.strip() for m in matches]


def expand_abbreviations(text: str) -> str:
    """
    Produce a human-readable expansion of the note by replacing known
    clinical shorthand with full terms. This demonstrates how domain
    knowledge bridges the gap between clinical shorthand and plain language.
    """
    expanded = text
    for abbr, full in sorted(GENERAL_ABBREVIATIONS.items(), key=lambda x: -len(x[0])):
        pattern = r"\b" + re.escape(abbr) + r"\b"
        expanded = re.sub(pattern, full, expanded, flags=re.IGNORECASE)
    return expanded


# ---------------------------------------------------------------------------
# STEP 3: Pipeline orchestration
# ---------------------------------------------------------------------------

def process_clinical_note(raw_text: str) -> dict:
    """
    Run the full extraction pipeline on a raw clinical note and return
    a structured dictionary -- the kind of standardized record that could
    feed downstream systems (claims review, analytics, payment integrity checks).
    """
    structured_record = {
        "diagnosis": extract_diagnosis(raw_text),
        "medications": extract_medications(raw_text),
        "vitals": extract_vitals(raw_text),
        "follow_up": extract_followup(raw_text),
        "plain_language_expansion": expand_abbreviations(raw_text),
    }
    return structured_record


def print_structured_record(record: dict):
    """Pretty-print the structured output for a clean demo presentation."""
    print("=" * 70)
    print("STRUCTURED CLINICAL RECORD")
    print("=" * 70)

    print("\nDIAGNOSIS:")
    for d in record["diagnosis"] or ["(none detected)"]:
        print(f"  - {d}")

    print("\nMEDICATIONS:")
    if record["medications"]:
        for m in record["medications"]:
            print(f"  - {m['drug']}: {m['dose']}, {m['frequency']}")
    else:
        print("  (none detected)")

    print("\nVITALS:")
    if record["vitals"]:
        for k, v in record["vitals"].items():
            print(f"  - {k.replace('_', ' ').title()}: {v}")
    else:
        print("  (none detected)")

    print("\nFOLLOW-UP:")
    for f in record["follow_up"] or ["(none detected)"]:
        print(f"  - {f}")

    print("\nPLAIN-LANGUAGE EXPANSION:")
    print(f"  {record['plain_language_expansion']}")

    print("\n" + "=" * 70)
    print("JSON OUTPUT (for downstream systems)")
    print("=" * 70)
    print(json.dumps(record, indent=2))


# ---------------------------------------------------------------------------
# DEMO
# ---------------------------------------------------------------------------

SAMPLE_CLINICAL_NOTE = """
Pt is a 58 y/o male. C/o CP and SOB x3 days, worsening w/ exertion.
Dx: Unstable angina. Hx of HTN and T2DM.
BP 148/92, HR 96, T 98.6 F.
Started ASA 81mg qd, Metoprolol 25mg bid, Atorvastatin 40mg qhs.
F/u in 1 week, repeat EKG, cardiology referral.
"""

def run_demo(note_text: str):
    print("\nINPUT: Raw, unstructured clinical note\n")
    print(note_text)
    record = process_clinical_note(note_text)
    print_structured_record(record)


if __name__ == "__main__":
    import sys

    if "--interactive" in sys.argv:
        print("=" * 70)
        print("CLINICAL NOTE EXTRACTOR -- Interactive Mode")
        print("=" * 70)
        print("Paste or type a clinical note below.")
        print("Type 'END' on its own line when finished, or press Enter")
        print("on an empty input to use the built-in sample note.\n")

        lines = []
        first_line = input("> ")
        if first_line.strip() == "":
            run_demo(SAMPLE_CLINICAL_NOTE)
        else:
            lines.append(first_line)
            while True:
                line = input("> ")
                if line.strip().upper() == "END":
                    break
                lines.append(line)
            custom_note = "\n".join(lines)
            run_demo(custom_note)
    else:
        # Default: run the built-in sample note
        run_demo(SAMPLE_CLINICAL_NOTE)
