import streamlit as st
import re
import json
import requests

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Clinical NLP Hybrid System", layout="wide")

st.title("🏥 Clinical NLP Hybrid System")
st.write("Rule-Based (Past) + LLM (Present) + Confidence Scoring (Future)")

# =========================
# RULE-BASED (PAST)
# =========================

def extract_diagnosis(text):
    pattern = r"(?:Dx|Diagnosis)\s*:\s*([^.\n]+)"
    return [m.strip() for m in re.findall(pattern, text, re.IGNORECASE)]


def extract_vitals(text):
    vitals = {}
    bp = re.search(r"BP\s*(\d{2,3}/\d{2,3})", text)
    if bp:
        vitals["blood_pressure"] = bp.group(1)
    hr = re.search(r"HR\s*(\d{2,3})", text)
    if hr:
        vitals["heart_rate"] = f"{hr.group(1)} bpm"
    temp = re.search(r"T\s*(\d{2,3}\.?\d*)", text)
    if temp:
        vitals["temperature"] = f"{temp.group(1)} F"
    return vitals


def extract_medications(text):
    pattern = r"\b([A-Z][a-zA-Z]+)\s+(\d+)\s*(mg|g|mcg)\b"
    matches = re.findall(pattern, text)
    return [{"drug": drug, "dose": f"{dose}{unit}"} for drug, dose, unit in matches]


# =========================
# LLM via OLLAMA (PRESENT)
# =========================

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"  # change to whichever model you have pulled

def llm_extract(text, model):
    """Call local Ollama to extract structured clinical data."""

    prompt = f"""You are a clinical NLP system. Extract structured data from the clinical note below.

Return ONLY valid JSON with these exact keys:
- "diagnosis": list of strings
- "medications": list of objects with "drug" and "dose" keys
- "vitals": object with any of: blood_pressure, heart_rate, temperature
- "follow_up": list of strings

If a field is not found, return an empty list or object for it.
Do not include any explanation, only the JSON.

Clinical note:
{text}"""

    try:
        response = requests.post(OLLAMA_URL, json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }, timeout=60)

        raw = response.json()["response"].strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw), None

    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to Ollama. Make sure it's running: `ollama serve`"
    except json.JSONDecodeError:
        return None, "Ollama responded but returned invalid JSON. Try a different model."
    except Exception as e:
        return None, f"Ollama error: {str(e)}"


# =========================
# CONFIDENCE SCORE (FUTURE)
# =========================

def confidence_score(result):
    score = 0.5
    if result["diagnosis"]:
        score += 0.2
    if result["medications"]:
        score += 0.2
    if result["vitals"]:
        score += 0.1
    return round(min(score, 0.92), 2)


# =========================
# HYBRID PIPELINE
# =========================

def process_note(text, model):
    rule_output = {
        "diagnosis": extract_diagnosis(text),
        "medications": extract_medications(text),
        "vitals": extract_vitals(text)
    }

    llm_output, error = llm_extract(text, model)

    if error:
        st.warning(f"⚠️ LLM layer skipped: {error}")
        llm_output = {"diagnosis": [], "medications": [], "vitals": {}, "follow_up": []}

    seen = set()
    unique_meds = []
    for m in rule_output["medications"] + llm_output["medications"]:
        key = (m["drug"].lower(), m["dose"].lower())
        if key not in seen:
            seen.add(key)
            unique_meds.append(m)

    merged = {
        "diagnosis": list(set(rule_output["diagnosis"] + llm_output["diagnosis"])),
        "medications": unique_meds,
        "vitals": {**rule_output["vitals"], **llm_output["vitals"]},
        "follow_up": llm_output.get("follow_up", [])
    }
    merged["confidence"] = confidence_score(merged)
    return merged, rule_output, llm_output


# =========================
# UI
# =========================

with st.sidebar:
    st.markdown("### ⚙️ Ollama Settings")
    selected_model = st.text_input("Model name", value=OLLAMA_MODEL)
    st.caption("Run `ollama list` to see your available models.")
    st.caption("Run `ollama serve` to start the server.")

input_text = st.text_area(
    "Enter Clinical Note",
    height=200,
    placeholder="Example: Dx: Chest pain. BP 148/92 HR 96. Aspirin 81mg"
)

if st.button("Process Note"):
    if not input_text.strip():
        st.warning("Please enter clinical text")
    else:
        with st.spinner("Running hybrid pipeline..."):
            result, rule_output, llm_output = process_note(input_text, selected_model)

        st.subheader("📊 Results")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Diagnosis")
            st.write(result["diagnosis"])
            st.markdown("### Vitals")
            st.json(result["vitals"])
        with col2:
            st.markdown("### Medications")
            st.write(result["medications"])
            st.markdown("### Follow-up")
            st.write(result["follow_up"])

        st.markdown("### 🔐 Confidence Score")
        st.metric("System Confidence", result["confidence"])

        with st.expander("🔍 Show rule-based vs LLM breakdown"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Rule-Based Output**")
                st.json(rule_output)
            with c2:
                st.markdown(f"**LLM Output ({selected_model})**")
                st.json(llm_output)

        st.markdown("### 📦 Full JSON Output")
        st.json(result)

st.markdown("---")
st.info("""
This system demonstrates:
- Past: Rule-based NLP (regex extraction)
- Present: LLM-based understanding (Ollama — runs locally)
- Future: Confidence scoring + hybrid fusion
""")
