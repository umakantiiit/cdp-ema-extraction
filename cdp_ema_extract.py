import streamlit as st
import json
import os
from google import genai
from google.genai import types

# Page configuration
st.set_page_config(
    page_title="EMA Extraction Tool",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        display: flex;
        align-items: center;
        gap: 15px;
        padding: 20px 0;
        border-bottom: 2px solid #e0e0e0;
        margin-bottom: 20px;
    }
    .stethoscope-icon {
        font-size: 40px;
    }
    .header-title {
        font-size: 32px;
        font-weight: bold;
        color: #1f4e79;
    }
    .json-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #dee2e6;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Header with stethoscope icon
st.markdown("""
<div class="main-header">
    <span class="stethoscope-icon">🩺</span>
    <span class="header-title">EMA EXTRACTION TOOL</span>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'extracted_data' not in st.session_state:
    st.session_state.extracted_data = None
if 'credentials_loaded' not in st.session_state:
    st.session_state.credentials_loaded = False

# File uploader for JSON credentials
st.subheader("📁 Upload Credentials")
uploaded_file = st.file_uploader(
    "Upload your Google Cloud credentials JSON file",
    type=['json'],
    help="Upload the service account JSON file for authentication"
)

if uploaded_file is not None:
    try:
        # Save the uploaded credentials to a temporary file
        credentials_content = uploaded_file.read()
        credentials_path = "/tmp/credentials.json"
        
        with open(credentials_path, 'wb') as f:
            f.write(credentials_content)
        
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        st.session_state.credentials_loaded = True
        st.success("✅ Credentials file uploaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading credentials: {str(e)}")

# Divider
st.divider()

# Text input area
st.subheader("📝 Input Clinical Text")
data_input = st.text_area(
    "Paste the plain text for extraction:",
    height=200,
    placeholder="Paste your clinical text here (e.g., therapeutic indications, clinical particulars, etc.)"
)

# Define your prompt (replace with your actual prompt)
cdp_ema_prompt = """
# Role and Persona
You are an expert Clinical Data Analyst and Regulatory Affairs Specialist specializing in Pharmacovigilance. Your expertise lies in parsing complex medical texts from the European Medicines Agency (EMA) and extracting highly structured data with zero error. You do not summarize; you extract exactly what is stated.

# Objective
Your task is to analyze the provided EMA clinical text and convert it into a single valid JSON array. The text contains various "Primary Disease Categories," and within those categories, multiple specific "Indications."

# Extraction Rules (Strict Compliance Required)

## 1. High-Level Logic
- The input text is divided into sections based on disease types (e.g., "Melanoma", "Non-small cell lung cancer").
- For each section, capture the `Disease_level_full_text`. This is the raw text for that entire disease section.
- Within the `Disease_level_full_text`, identify every distinct "Indication".
- Create a separate JSON object for *each* indication found.

## 2. Field-Specific Definitions & Extraction Logic

### **Primary Disease_category**
- **Source:** The bolded or capitalized header starting the section.
- FALLBACK APPROACH - "If no explicit header exists, use the disease name found in the indication text as the Primary Category."
- IF the Indication_text lists multiple distinct tumor types (e.g., "gastric, small intestine, or biliary cancer") that share the same treatment conditions, you MUST create a separate JSON object for each tumor type.
- Bad Example: {"Disease": "gastric, small intestine, or biliary cancer"}
- Good Example: [{"Disease": "gastric cancer"}, {"Disease": "small intestine cancer"}, {"Disease": "biliary cancer"}]
- **Example:** "Melanoma", "Non-small cell lung cancer (NSCLC)".
- DONOT SPLIT THE PRIMARY DISEASE IF ONLY Treatment modality DIFFERS and everything else is same. 

### **Disease_level_full_text**
- **Source:** The entire text block belonging to that Primary Disease Category.
- **Rule:** This text will repeat identically for every indication object that belongs to this category.

### **Indication #**
- **Logic:** An integer counter (1, 2, 3...) representing the specific indication sequence within that Primary Disease Category. Reset to 1 for a new Disease Category.

### **Indication_text**
- **Source:** The specific sentence(s) defining who and what is being treated.
- **Constraint:** Stop extracting when the text moves to a new patient population or a different drug combination.

### **Treatment line**
- **Source:** EXTRACT ONLY FROM "Indication_text".
- **Logic (Evaluate in this specific order):**

  1. **Rule (The "At Least" Range):**
     - IF text contains "at least one" (or "≥ 1") prior therapy/treatment:
       - OUTPUT: "Second line and later"
     - IF text contains "at least two" (or "≥ 2") prior therapies:
       - OUTPUT: "Third line and later"

  2. **Rule (First Line):**
     - IF text says "first-line", "previously untreated", "treatment naïve", OR "no prior systemic therapy".
     - OUTPUT: "First line"

  3. **Rule (Calculated Line - The "+1" Logic):**
     - IF text says "after [Number] prior therapies" or "after [Number] lines" (e.g., "after 3 lines"):
     - ACTION: Add 1 to the number found. (e.g., 3 + 1 = 4).
     - OUTPUT: "[Result] line" (e.g., "Fourth line")

  4. **Rule (Explicit Second/Third Label):**
     - If text explicitly says "second-line" -> OUTPUT: "Second line".
     - If text explicitly says "third-line" -> OUTPUT: "Third line".

  5. **Rule (General Second Line / Relapsed / Refractory):**
     - IF text contains any of the following:
       - "considered inappropriate (example- metformin is considered inappropriate means the patient fails with that therapy.)
       - "after prior therapy", "after prior chemotherapy"
       - "after failure of...", "progressing on or after..."
       - "relapsed", "refractory"
       - "previously treated with"
     - OUTPUT: "Second line"

  6. **Rule (Adjuvant/Neoadjuvant Exception):**
     - IF text mentions "Adjuvant" or "Neoadjuvant" AND does not specify a line number.
     - OUTPUT: "_"

  7. **Rule (Default):**
     - If none of the above match.
     - OUTPUT: "_"


### **Treatment modality**
- **Source:** EXTRACT ONLY FROM "Indication_text".
- **Logic:** Look for these keywords and combine them with commas if multiple exist:
  - "Monotherapy" (or implied if used alone).
  - "Combination" (if the text contains “in combination with”; if used with ipilimumab, chemotherapy, etc.).
  - If multiple modalities apply, combine them using commas in a single string.
- **Example:** "Combination, Neoadjuvant"
- Adjunct detection rules (FOR LEQVIO AND SIMILAR DRUGS):
  - If the indication text contains any of the following phrases, include “Adjunct”:
  - “as an adjunct to diet”
  - “as an adjunct to therapy”
  - “adjunctive therapy”
  - “used as an adjunct”
  - Adjunct refers to add-on supportive use, not treatment sequencing.
  - Adjunct must be included independently of Monotherapy or Combination when applicable.
- Neoadjuvant detection rules:
  - If the indication text contains “neoadjuvant treatment” or “as neoadjuvant”, include “Neoadjuvant”.
- Adjuvant detection rules:
  - If the indication text contains “adjuvant treatment” or “as adjuvant”, include “Adjuvant”.
- Multiple modality combination rules:
  - If Adjunct + Combination → output “Adjunct, Combination”.
  - If Adjunct + Monotherapy → output “Adjunct, Monotherapy”.
  - If Adjunct + Monotherapy + Combination → output “Adjunct, Monotherapy, Combination”.
  - If Neoadjuvant followed by Adjuvant with different modalities → output: “Neoadjuvant, Adjuvant, Combination, Monotherapy”.


### **Population**
- **Source:** EXTRACT ONLY FROM "Indication_text".
- **Logic:** Identify the target demographic.
- Text-based population rules:
  - If the text contains infant, neonate, or newborn, output “Infant”.
  - If the text contains pediatric, paediatric, or children, output “Paediatric”.
  - If the text contains adolescents, output “Adolescent”.
  - If the text contains adults or adult patients, output “Adult”.
  - If the text contains elderly, geriatric, or ≥ 65 years, output “Elderly”.
- Numeric age-based population rules (mandatory):
  - Age 0 to 1 years maps to “Infant”.
  - Age greater than 1 and up to 12 years maps to “Pediatric”.
  - Age greater than 12 and up to 18 years maps to “Adolescent”.
  - Age greater than 18 and up to 60 years maps to “Adult”.
  - Age greater than 60 years maps to “Elderly”.
- Range overlap rules:
  - If an age range spans multiple groups, include all applicable populations.
  - **Example:** Age 10 to 14 outputs “Pediatric, Adolescent”.
  - **Example:** Age 58 to 70 outputs “Adult, Elderly”.
  - **Example:** ≥12 years outputs “Adolescent, Adult, Elderly”.
- Population formatting rules:
  - Only these exact values are allowed:
    - Infant
    - Paediatric
    - Adolescent
    - Adult
    - Elderly
  - If no text age and no numeric age is present, output “_”.
  - Never default to Adult.
  - Never guess the population.


**Final Formatting:**
- Join multiple matches with a comma (e.g., "Adult, Adolescent").
- If no population is mentioned, valid output is null or inferred from context only if highly obvious, otherwise "_".

### **Disease + sybtypes**
- **Source:** EXTRACT ONLY FROM "Indication_text".
- **Logic:** Extract the specific condition description, stage, or mutation status mentioned.
- **Example:** "unresectable or metastatic melanoma" or "tumours have PD-L1 expression >= 1%".

Field: Disease + subtypes (Strict Cleaning Rules)
Objective: Extract only the specific medical condition or patient state. Critical Rule: This field describes the PATIENT'S BODY, not the DRUG'S ACTION.

Instructions to extract text for Disease+subtypes:

Start by identifying the core disease name (e.g., "Type 2 diabetes mellitus", "Melanoma").

Keep specific disease modifiers found immediately around the disease name:

"insufficiently controlled"

"metastatic", "advanced", "resectable", "unresectable"

Specific genetic mutations (e.g., "PD-L1 positive", "BRAF V600 mutation")

Risk levels (e.g., "at high risk of recurrence")

REMOVE all text related to:

Treatment Context: "as an adjunct to diet and exercise", "in combination with...", "as monotherapy".

Rationale/Reasoning: "when metformin is considered inappropriate", "due to intolerance".

Treatment History (unless part of the patient definition): "after failure of...", "progressing on...". ( Note: Only keep these if they define the patient group, like 'relapsed/refractory'. If it just describes the timing, leave it out.)

Examples for Training:

Text: "treatment of adults with insufficiently controlled type 2 diabetes mellitus as an adjunct to diet and exercise"

Bad Extraction: "insufficiently controlled type 2 diabetes mellitus as an adjunct to diet and exercise"

Good Extraction: "insufficiently controlled type 2 diabetes mellitus"

Text: "treatment of advanced melanoma in adults progressing on platinum-based therapy"

Good Extraction: "advanced melanoma" (Note: "progressing on..." is captured in Treatment Line, not Disease).

Text: "treatment of adults with MSI-H colorectal cancer"

Good Extraction: "MSI-H colorectal cancer"

# Negative Constraints (To prevent Hallucination)
1. DO NOT infer information. If the `Indication_text` does not state the Population, do not guess "Adult".
2. DO NOT include text from the "Disease_level_full_text" into the "Disease + sybtypes" field unless it is explicitly present in the "Indication_text".
3. DO NOT alter the terminology used in the text (e.g., if it says "unresectable", do not change it to "non-operable").

# One-Shot Example (Use this structure exactly)

**Input Text Segment:**
4. 4.1 CLINICAL PARTICULARS Therapeutic indications Melanoma OPDIVO as monotherapy or in combination with ipilimumab is indicated for the treatment of advanced (unresectable or metastatic) melanoma in adults and adolescents 12 years of age and older. Relative to nivolumab monotherapy, an increase in progression-free survival (PFS) and overall survival (OS) for the combination of nivolumab with ipilimumab is established only in patients with low tumour PD-L1 expression (see sections 4.4 and 5.1). Adjuvant treatment of melanoma OPDIVO as monotherapy is indicated for the adjuvant treatment of adults and adolescents 12 years of age and older with Stage IIB or IIC melanoma, or melanoma with involvement of lymph nodes or metastatic disease who have undergone complete resection (see section 5.1). Non-small cell lung cancer (NSCLC) OPDIVO in combination with ipilimumab and 2 cycles of platinum-based chemotherapy is indicated for the first-line treatment of metastatic non-small cell lung cancer in adults whose tumours have no sensitising EGFR mutation or ALK translocation. OPDIVO as monotherapy is indicated for the treatment of locally advanced or metastatic non-small cell lung cancer after prior chemotherapy in adults. 2 Neoadjuvant treatment of NSCLC OPDIVO in combination with platinum-based chemotherapy is indicated for the neoadjuvant treatment of resectable non-small cell lung cancer at high risk of recurrence in adult patients whose tumours have PD-L1 expression â‰¥ 1% (see section 5.1 for selection criteria). Neoadjuvant and adjuvant treatment of NSCLC OPDIVO, in combination with platinum-based chemotherapy as neoadjuvant treatment, followed by OPDIVO as monotherapy as adjuvant treatment, is indicated for the treatment of resectable non-small cell lung cancer at high risk of recurrence in adult patients whose tumours have PD-L1 expression â‰¥ 1% (see section 5.1 for selection criteria). Malignant pleural mesothelioma (MPM) OPDIVO in combination with ipilimumab is indicated for the first-line treatment of adult patients with unresectable malignant pleural mesothelioma. Renal cell carcinoma (RCC) OPDIVO as monotherapy is indicated for the treatment of advanced renal cell carcinoma after prior therapy in adults. OPDIVO in combination with ipilimumab is indicated for the first-line treatment of adult patients with intermediate/poor-risk advanced renal cell carcinoma (see section 5.1). OPDIVO in combination with cabozantinib is indicated for the first-line treatment of adult patients with advanced renal cell carcinoma (see section 5.1). Classical Hodgkin lymphoma (cHL) OPDIVO as monotherapy is indicated for the treatment of adult patients with relapsed or refractory classical Hodgkin lymphoma after autologous stem cell transplant (ASCT) and treatment with brentuximab vedotin. Squamous cell cancer of the head and neck (SCCHN) OPDIVO as monotherapy is indicated for the treatment of recurrent or metastatic squamous cell cancer of the head and neck in adults progressing on or after platinum-based therapy (see section 5.1). Urothelial carcinoma OPDIVO in combination with cisplatin and gemcitabine is indicated for the first-line treatment of adult patients with unresectable or metastatic urothelial carcinoma. OPDIVO as monotherapy is indicated for the treatment of locally advanced unresectable or metastatic urothelial carcinoma in adults after failure of prior platinum-containing therapy. Adjuvant treatment of urothelial carcinoma OPDIVO as monotherapy is indicated for the adjuvant treatment of adults with muscle invasive urothelial carcinoma (MIUC) with tumour cell PD-L1 expression â‰¥ 1%, who are at high risk of recurrence after undergoing radical resection of MIUC (see section 5.1). 3 Mismatch repair deficient (dMMR) or microsatellite instability-high (MSI-H) colorectal cancer (CRC) OPDIVO in combination with ipilimumab is indicated for the treatment of adult patients with mismatch repair deficient or microsatellite instability-high colorectal cancer in the following settings: - - first-line treatment of unresectable or metastatic colorectal cancer; treatment of metastatic colorectal cancer after prior fluoropyrimidine-based combination chemotherapy (see section 5.1). Oesophageal squamous cell carcinoma (OSCC) OPDIVO in combination with ipilimumab is indicated for the first-line treatment of adult patients with unresectable advanced, recurrent or metastatic oesophageal squamous cell carcinoma with tumour cell PD-L1 expression â‰¥ 1%. OPDIVO in combination with fluoropyrimidine- and platinum-based combination chemotherapy is indicated for the first-line treatment of adult patients with unresectable advanced, recurrent or metastatic oesophageal squamous cell carcinoma with tumour cell PD-L1 expression â‰¥ 1%. OPDIVO as monotherapy is indicated for the treatment of adult patients with unresectable advanced, recurrent or metastatic oesophageal squamous cell carcinoma after prior fluoropyrimidine- and platinum-based combination chemotherapy. Adjuvant treatment of oesophageal or gastro-oesophageal junction cancer (OC or GEJC) OPDIVO as monotherapy is indicated for the adjuvant treatment of adult patients with oesophageal or gastro-oesophageal junction cancer who have residual pathologic disease following prior neoadjuvant chemoradiotherapy (see section 5.1). Gastric, gastro-oesophageal junction (GEJ) or oesophageal adenocarcinoma OPDIVO in combination with fluoropyrimidine- and platinum-based combination chemotherapy is indicated for the first-line treatment of adult patients with HER2-negative advanced or metastatic gastric, gastro-oesophageal junction or oesophageal adenocarcinoma whose tumours express PD-L1 with a combined positive score (CPS) â‰¥ 5. Hepatocellular carcinoma (HCC) OPDIVO in combination with ipilimumab is indicated for the first-line treatment of adult patients with unresectable or advanced hepatocellular carcinoma.

output json format:

[
    {
        "Primary Disease_category": "Melanoma",
        "Disease_level_full_text": "Melanoma OPDIVO as monotherapy or in combination with ipilimumab is indicated for the treatment of advanced (unresectable or metastatic) melanoma in adults and adolescents 12 years of age and older. Relative to nivolumab monotherapy, an increase in progression-free survival (PFS) and overall survival (OS) for the combination of nivolumab with ipilimumab is established only in patients with low tumour PD-L1 expression (see sections 4.4 and 5.1). Adjuvant treatment of melanoma OPDIVO as monotherapy is indicated for the adjuvant treatment of adults and adolescents 12 years of age and older with Stage IIB or IIC melanoma, or melanoma with involvement of lymph nodes or metastatic disease who have undergone complete resection (see section 5.1)",
        "Indication #": 1,
        "Indication_text": "OPDIVO as monotherapy or in combination with ipilimumab is indicated for the treatment of advanced (unresectable or metastatic) melanoma in adults and adolescents 12 years of age and older.",
        "Treatment line": "_",
        "Treatment modality": "Monotherapy,Combination",
        "Population": "Adult, Adolescent",
        "Disease + sybtypes": "advanced (unresectable or metastatic)"
    },
    {
        "Primary Disease_category": "Melanoma",
        "Disease_level_full_text": "Melanoma OPDIVO as monotherapy or in combination with ipilimumab is indicated for the treatment of advanced (unresectable or metastatic) melanoma in adults and adolescents 12 years of age and older. Relative to nivolumab monotherapy, an increase in progression-free survival (PFS) and overall survival (OS) for the combination of nivolumab with ipilimumab is established only in patients with low tumour PD-L1 expression (see sections 4.4 and 5.1). Adjuvant treatment of melanoma OPDIVO as monotherapy is indicated for the adjuvant treatment of adults and adolescents 12 years of age and older with Stage IIB or IIC melanoma, or melanoma with involvement of lymph nodes or metastatic disease who have undergone complete resection",
        "Indication #": 2,
        "Indication_text": "OPDIVO as monotherapy is indicated for the adjuvant treatment of adults and adolescents 12 years of age and older with Stage IIB or IIC melanoma, or melanoma with involvement of lymph nodes or metastatic disease who have undergone complete resection.",
        "Treatment line": "_",
        "Treatment modality": "Adjuvant, Monotherapy",
        "Population": "Adult, Adolescent",
        "Disease + sybtypes": "Stage IIB or IIC melanoma, or melanoma with involvement of lymph nodes or metastatic disease"
    },
    {
        "Primary Disease_category": "Non-small cell lung cancer (NSCLC)",
        "Disease_level_full_text": "Non-small cell lung cancer (NSCLC) OPDIVO in combination with ipilimumab and 2 cycles of platinum-based chemotherapy is indicated for the first-line treatment of metastatic non-small cell lung cancer in adults whose tumours have no sensitising EGFR mutation or ALK translocation. OPDIVO as monotherapy is indicated for the treatment of locally advanced or metastatic non-small cell lung cancer after prior chemotherapy in adults. Neoadjuvant treatment of NSCLC OPDIVO in combination with platinum-based chemotherapy is indicated for the neoadjuvant treatment of resectable non-small cell lung cancer at high risk of recurrence in adult patients whose tumours have PD-L1 expression \u2265 1% (see section 5.1 for selection criteria). Neoadjuvant and adjuvant treatment of NSCLC OPDIVO, in combination with platinum-based chemotherapy as neoadjuvant treatment, followed by OPDIVO as monotherapy as adjuvant treatment, is indicated for the treatment of resectable non-small cell lung cancer at high risk of recurrence in adult patients whose tumours have PD-L1 expression \u2265 1% (see section 5.1 for selection criteria).",
        "Indication #": 1,
        "Indication_text": "OPDIVO in combination with ipilimumab and 2 cycles of platinum-based chemotherapy is indicated \nfor the first-line treatment of metastatic non-small cell lung cancer in adults whose tumours have no \nsensitising EGFR mutation or ALK translocation.",
        "Treatment line": "First line",
        "Treatment modality": "Combination",
        "Population": "Adult",
        "Disease + sybtypes": "non-small cell lung cancer in adults whose tumours have no sensitising EGFR mutation or ALK translocation"
    },
    {
        "Primary Disease_category": "Non-small cell lung cancer (NSCLC)",
        "Disease_level_full_text": "Non-small cell lung cancer (NSCLC) OPDIVO in combination with ipilimumab and 2 cycles of platinum-based chemotherapy is indicated for the first-line treatment of metastatic non-small cell lung cancer in adults whose tumours have no sensitising EGFR mutation or ALK translocation. OPDIVO as monotherapy is indicated for the treatment of locally advanced or metastatic non-small cell lung cancer after prior chemotherapy in adults. Neoadjuvant treatment of NSCLC OPDIVO in combination with platinum-based chemotherapy is indicated for the neoadjuvant treatment of resectable non-small cell lung cancer at high risk of recurrence in adult patients whose tumours have PD-L1 expression \u2265 1% (see section 5.1 for selection criteria). Neoadjuvant and adjuvant treatment of NSCLC OPDIVO, in combination with platinum-based chemotherapy as neoadjuvant treatment, followed by OPDIVO as monotherapy as adjuvant treatment, is indicated for the treatment of resectable non-small cell lung cancer at high risk of recurrence in adult patients whose tumours have PD-L1 expression \u2265 1% (see section 5.1 for selection criteria).",
        "Indication #": 2,
        "Indication_text": "OPDIVO as monotherapy is indicated for the treatment of locally advanced or metastatic non-small \ncell lung cancer after prior chemotherapy in adults.",
        "Treatment line": "Second line",
        "Treatment modality": "Monotherapy",
        "Population": "Adult",
        "Disease + sybtypes": "locally advanced or metastatic non-small \ncell lung cancer"
    },
    {
        "Primary Disease_category": "Non-small cell lung cancer (NSCLC)",
        "Disease_level_full_text": "Non-small cell lung cancer (NSCLC) OPDIVO in combination with ipilimumab and 2 cycles of platinum-based chemotherapy is indicated for the first-line treatment of metastatic non-small cell lung cancer in adults whose tumours have no sensitising EGFR mutation or ALK translocation. OPDIVO as monotherapy is indicated for the treatment of locally advanced or metastatic non-small cell lung cancer after prior chemotherapy in adults. Neoadjuvant treatment of NSCLC OPDIVO in combination with platinum-based chemotherapy is indicated for the neoadjuvant treatment of resectable non-small cell lung cancer at high risk of recurrence in adult patients whose tumours have PD-L1 expression \u2265 1% (see section 5.1 for selection criteria). Neoadjuvant and adjuvant treatment of NSCLC OPDIVO, in combination with platinum-based chemotherapy as neoadjuvant treatment, followed by OPDIVO as monotherapy as adjuvant treatment, is indicated for the treatment of resectable non-small cell lung cancer at high risk of recurrence in adult patients whose tumours have PD-L1 expression \u2265 1% (see section 5.1 for selection criteria).",
        "Indication #": 3,
        "Indication_text": "OPDIVO in combination with platinum-based chemotherapy is indicated for the neoadjuvant treatment of resectable non-small cell lung cancer at high risk of recurrence in adult patients whose tumours have PD-L1 expression \u2265 1% (see section 5.1 for selection criteria).",
        "Treatment line": "_",
        "Treatment modality": "Combination, Neoadjuvant",
        "Population": "Adult",
        "Disease + sybtypes": "resectable non-small cell lung cancer at high risk of recurrence"
    },
    {
        "Primary Disease_category": "Non-small cell lung cancer (NSCLC)",
        "Disease_level_full_text": "Non-small cell lung cancer (NSCLC) OPDIVO in combination with ipilimumab and 2 cycles of platinum-based chemotherapy is indicated for the first-line treatment of metastatic non-small cell lung cancer in adults whose tumours have no sensitising EGFR mutation or ALK translocation. OPDIVO as monotherapy is indicated for the treatment of locally advanced or metastatic non-small cell lung cancer after prior chemotherapy in adults. Neoadjuvant treatment of NSCLC OPDIVO in combination with platinum-based chemotherapy is indicated for the neoadjuvant treatment of resectable non-small cell lung cancer at high risk of recurrence in adult patients whose tumours have PD-L1 expression \u2265 1% (see section 5.1 for selection criteria). Neoadjuvant and adjuvant treatment of NSCLC OPDIVO, in combination with platinum-based chemotherapy as neoadjuvant treatment, followed by OPDIVO as monotherapy as adjuvant treatment, is indicated for the treatment of resectable non-small cell lung cancer at high risk of recurrence in adult patients whose tumours have PD-L1 expression \u2265 1% (see section 5.1 for selection criteria).",
        "Indication #": 4,
        "Indication_text": "OPDIVO, in combination with platinum-based chemotherapy as neoadjuvant treatment, followed by OPDIVO as monotherapy as adjuvant treatment, is indicated for the treatment of resectable non-small cell lung cancer at high risk of recurrence in adult patients whose tumours have PD-L1 expression \u2265 1% (see section 5.1 for selection criteria).",
        "Treatment line": "_",
        "Treatment modality": "Combination, Neoadjuvant, Adjuvant, Monotherapy",
        "Population": "Adult",
        "Disease + sybtypes": "resectable non-small \ncell lung cancer at high risk of recurrence"
    },
    {
        "Primary Disease_category": "Malignant pleural mesothelioma (MPM)",
        "Disease_level_full_text": "Malignant pleural mesothelioma (MPM) OPDIVO in combination with ipilimumab is indicated for the first-line treatment of adult patients with unresectable malignant pleural mesothelioma.",
        "Indication #": 1,
        "Indication_text": "OPDIVO in combination with ipilimumab is indicated for the first-line treatment of adult patients with unresectable malignant pleural mesothelioma.",
        "Treatment line": "First line",
        "Treatment modality": "Combination",
        "Population": "Adult",
        "Disease + sybtypes": "unresectable malignant pleural mesothelioma"
    },
    {
        "Primary Disease_category": "Renal cell carcinoma (RCC)",
        "Disease_level_full_text": "Renal cell carcinoma (RCC) OPDIVO as monotherapy is indicated for the treatment of advanced renal cell carcinoma after prior therapy in adults. OPDIVO in combination with ipilimumab is indicated for the first-line treatment of adult patients with intermediate/poor-risk advanced renal cell carcinoma (see section 5.1). OPDIVO in combination with cabozantinib is indicated for the first-line treatment of adult patients with advanced renal cell carcinoma (see section 5.1).",
        "Indication #": 1,
        "Indication_text": "OPDIVO as monotherapy is indicated for the treatment of advanced renal cell carcinoma after prior therapy in adults",
        "Treatment line": "Second line",
        "Treatment modality": "Monotherapy",
        "Population": "Adult",
        "Disease + sybtypes": "advanced renal cell carcinoma"
    },
    {
        "Primary Disease_category": "Renal cell carcinoma (RCC)",
        "Disease_level_full_text": "Renal cell carcinoma (RCC) OPDIVO as monotherapy is indicated for the treatment of advanced renal cell carcinoma after prior therapy in adults. OPDIVO in combination with ipilimumab is indicated for the first-line treatment of adult patients with intermediate/poor-risk advanced renal cell carcinoma (see section 5.1). OPDIVO in combination with cabozantinib is indicated for the first-line treatment of adult patients with advanced renal cell carcinoma (see section 5.1).",
        "Indication #": 2,
        "Indication_text": "OPDIVO in combination with ipilimumab is indicated for the first-line treatment of adult patients with intermediate/poor-risk advanced renal cell carcinoma",
        "Treatment line": "First line",
        "Treatment modality": "Combination",
        "Population": "Adult",
        "Disease + sybtypes": "intermediate/poor-risk advanced renal cell carcinoma"
    },
    {
        "Primary Disease_category": "Renal cell carcinoma (RCC)",
        "Disease_level_full_text": "Renal cell carcinoma (RCC) OPDIVO as monotherapy is indicated for the treatment of advanced renal cell carcinoma after prior therapy in adults. OPDIVO in combination with ipilimumab is indicated for the first-line treatment of adult patients with intermediate/poor-risk advanced renal cell carcinoma (see section 5.1). OPDIVO in combination with cabozantinib is indicated for the first-line treatment of adult patients with advanced renal cell carcinoma (see section 5.1).",
        "Indication #": 3,
        "Indication_text": "OPDIVO in combination with cabozantinib is indicated for the first-line treatment of adult patients with advanced renal cell carcinoma",
        "Treatment line": "First line",
        "Treatment modality": "Combination",
        "Population": "Adult",
        "Disease + sybtypes": "advanced renal cell carcinoma"
    },
    {
        "Primary Disease_category": "Classical Hodgkin lymphoma (cHL)",
        "Disease_level_full_text": "Classical Hodgkin lymphoma (cHL) OPDIVO as monotherapy is indicated for the treatment of adult patients with relapsed or refractory classical Hodgkin lymphoma after autologous stem cell transplant (ASCT) and treatment with brentuximab vedotin.",
        "Indication #": 1,
        "Indication_text": "OPDIVO as monotherapy is indicated for the treatment of adult patients with relapsed or refractory classical Hodgkin lymphoma after autologous stem cell transplant (ASCT) and treatment with brentuximab vedotin",
        "Treatment line": "Third line",
        "Treatment modality": "Monotherapy",
        "Population": "Adult",
        "Disease + sybtypes": "relapsed or refractory classical Hodgkin lymphoma"
    },
    {
        "Primary Disease_category": "Squamous cell cancer of the head and neck (SCCHN)",
        "Disease_level_full_text": "Squamous cell cancer of the head and neck (SCCHN) OPDIVO as monotherapy is indicated for the treatment of recurrent or metastatic squamous cell cancer of the head and neck in adults progressing on or after platinum-based therapy",
        "Indication #": 1,
        "Indication_text": "OPDIVO as monotherapy is indicated for the treatment of recurrent or metastatic squamous cell cancer of the head and neck in adults progressing on or after platinum-based therapy",
        "Treatment line": "Second line",
        "Treatment modality": "Monotherapy",
        "Population": "Adult",
        "Disease + sybtypes": "recurrent or metastatic squamous cell cancer of the head and neck"
    },
    {
        "Primary Disease_category": "Urothelial carcinoma",
        "Disease_level_full_text": "Urothelial carcinoma OPDIVO in combination with cisplatin and gemcitabine is indicated for the first-line treatment of adult patients with unresectable or metastatic urothelial carcinoma. OPDIVO as monotherapy is indicated for the treatment of locally advanced unresectable or metastatic urothelial carcinoma in adults after failure of prior platinum-containing therapy. Adjuvant treatment of urothelial carcinoma OPDIVO as monotherapy is indicated for the adjuvant treatment of adults with muscle invasive urothelial carcinoma (MIUC) with tumour cell PD-L1 expression \u2265 1%, who are at high risk of recurrence after undergoing radical resection of MIUC (see section 5.1).",
        "Indication #": 1,
        "Indication_text": "OPDIVO in combination with cisplatin and gemcitabine is indicated for the first-line treatment of adult patients with unresectable or metastatic urothelial carcinoma.",
        "Treatment line": "First line",
        "Treatment modality": "Combination",
        "Population": "Adult",
        "Disease + sybtypes": "unresectable or metastatic urothelial carcinoma"
    },
    {
        "Primary Disease_category": "Urothelial carcinoma",
        "Disease_level_full_text": "Urothelial carcinoma OPDIVO in combination with cisplatin and gemcitabine is indicated for the first-line treatment of adult patients with unresectable or metastatic urothelial carcinoma. OPDIVO as monotherapy is indicated for the treatment of locally advanced unresectable or metastatic urothelial carcinoma in adults after failure of prior platinum-containing therapy. Adjuvant treatment of urothelial carcinoma OPDIVO as monotherapy is indicated for the adjuvant treatment of adults with muscle invasive urothelial carcinoma (MIUC) with tumour cell PD-L1 expression \u2265 1%, who are at high risk of recurrence after undergoing radical resection of MIUC (see section 5.1).",
        "Indication #": 2,
        "Indication_text": "OPDIVO as monotherapy is indicated for the treatment of locally advanced unresectable or metastatic urothelial carcinoma in adults after failure of prior platinum-containing therapy",
        "Treatment line": "Second line",
        "Treatment modality": "Monotherapy",
        "Population": "Adult",
        "Disease + sybtypes": "locally advanced unresectable or metastatic urothelial carcinoma"
    },
    {
        "Primary Disease_category": "Urothelial carcinoma",
        "Disease_level_full_text": "Urothelial carcinoma OPDIVO in combination with cisplatin and gemcitabine is indicated for the first-line treatment of adult patients with unresectable or metastatic urothelial carcinoma. OPDIVO as monotherapy is indicated for the treatment of locally advanced unresectable or metastatic urothelial carcinoma in adults after failure of prior platinum-containing therapy. Adjuvant treatment of urothelial carcinoma OPDIVO as monotherapy is indicated for the adjuvant treatment of adults with muscle invasive urothelial carcinoma (MIUC) with tumour cell PD-L1 expression \u2265 1%, who are at high risk of recurrence after undergoing radical resection of MIUC (see section 5.1)",
        "Indication #": 3,
        "Indication_text": "OPDIVO as monotherapy is indicated for the adjuvant treatment of adults with muscle invasive urothelial carcinoma (MIUC) with tumour cell PD-L1 expression \u2265 1%, who are at high risk of recurrence after undergoing radical resection of MIUC",
        "Treatment line": "Second line",
        "Treatment modality": "Monotherapy,Adjuvant",
        "Population": "Adult",
        "Disease + sybtypes": "muscle invasive urothelial carcinoma (MIUC) with tumour cell PD-L1 expression \u2265 1%, who are at high risk of recurrence"
    },
    {
        "Primary Disease_category": "Mismatch repair deficient (dMMR) or microsatellite instability-high (MSI-H) colorectal cancer (CRC)",
        "Disease_level_full_text": "Mismatch repair deficient (dMMR) or microsatellite instability-high (MSI-H) colorectal cancer (CRC) OPDIVO in combination with ipilimumab is indicated for the treatment of adult patients with mismatch repair deficient or microsatellite instability-high colorectal cancer in the following settings: - - first-line treatment of unresectable or metastatic colorectal cancer; treatment of metastatic colorectal cancer after prior fluoropyrimidine-based combination chemotherapy",
        "Indication #": 1,
        "Indication_text": "OPDIVO in combination with ipilimumab is indicated for the treatment of adult patients with mismatch repair deficient or microsatellite instability-high colorectal cancer in the following settings: - - first-line treatment of unresectable or metastatic colorectal cancer",
        "Treatment line": "First line",
        "Treatment modality": "Combination",
        "Population": "Adult",
        "Disease + sybtypes": "unresectable or metastatic colorectal cancer"
    },
    {
        "Primary Disease_category": "Mismatch repair deficient (dMMR) or microsatellite instability-high (MSI-H) colorectal cancer (CRC)",
        "Disease_level_full_text": "Mismatch repair deficient (dMMR) or microsatellite instability-high (MSI-H) colorectal cancer (CRC) OPDIVO in combination with ipilimumab is indicated for the treatment of adult patients with mismatch repair deficient or microsatellite instability-high colorectal cancer in the following settings: - - first-line treatment of unresectable or metastatic colorectal cancer; treatment of metastatic colorectal cancer after prior fluoropyrimidine-based combination chemotherapy",
        "Indication #": 2,
        "Indication_text": "OPDIVO in combination with ipilimumab is indicated for the treatment of adult patients with mismatch repair deficient or microsatellite instability-high colorectal cancer in the following settings-treatment of metastatic colorectal cancer after prior fluoropyrimidine-based combination chemotherapy",
        "Treatment line": "Second line",
        "Treatment modality": "Combination",
        "Population": "Adult",
        "Disease + sybtypes": "metastatic colorectal cancer"
    },
    {
        "Primary Disease_category": "Oesophageal squamous cell carcinoma (OSCC)",
        "Disease_level_full_text": "Oesophageal squamous cell carcinoma (OSCC) OPDIVO in combination with ipilimumab is indicated for the first-line treatment of adult patients with unresectable advanced, recurrent or metastatic oesophageal squamous cell carcinoma with tumour cell PD-L1 expression \u2265 1%. OPDIVO in combination with fluoropyrimidine- and platinum-based combination chemotherapy is indicated for the first-line treatment of adult patients with unresectable advanced, recurrent or metastatic oesophageal squamous cell carcinoma with tumour cell PD-L1 expression \u2265 1%. OPDIVO as monotherapy is indicated for the treatment of adult patients with unresectable advanced, recurrent or metastatic oesophageal squamous cell carcinoma after prior fluoropyrimidine- and platinum-based combination chemotherapy.",
        "Indication #": 1,
        "Indication_text": "OPDIVO in combination with ipilimumab is indicated for the first-line treatment of adult patients with unresectable advanced, recurrent or metastatic oesophageal squamous cell carcinoma with tumour cell PD-L1 expression \u2265 1%",
        "Treatment line": "First line",
        "Treatment modality": "Combination",
        "Population": "Adult",
        "Disease + sybtypes": "unresectable advanced, recurrent or metastatic oesophageal squamous cell carcinoma"
    },
    {
        "Primary Disease_category": "Oesophageal squamous cell carcinoma (OSCC)",
        "Disease_level_full_text": "Oesophageal squamous cell carcinoma (OSCC) OPDIVO in combination with ipilimumab is indicated for the first-line treatment of adult patients with unresectable advanced, recurrent or metastatic oesophageal squamous cell carcinoma with tumour cell PD-L1 expression \u2265 1%. OPDIVO in combination with fluoropyrimidine- and platinum-based combination chemotherapy is indicated for the first-line treatment of adult patients with unresectable advanced, recurrent or metastatic oesophageal squamous cell carcinoma with tumour cell PD-L1 expression \u2265 1%. OPDIVO as monotherapy is indicated for the treatment of adult patients with unresectable advanced, recurrent or metastatic oesophageal squamous cell carcinoma after prior fluoropyrimidine- and platinum-based combination chemotherapy.",
        "Indication #": 2,
        "Indication_text": "OPDIVO in combination with fluoropyrimidine- and platinum-based combination chemotherapy is indicated for the first-line treatment of adult patients with unresectable advanced, recurrent or metastatic oesophageal squamous cell carcinoma with tumour cell PD-L1 expression \u2265 1%",
        "Treatment line": "First line",
        "Treatment modality": "Combination",
        "Population": "Adult",
        "Disease + sybtypes": "unresectable advanced, recurrent or metastatic oesophageal squamous cell carcinoma"
    },
    {
        "Primary Disease_category": "Oesophageal squamous cell carcinoma (OSCC)",
        "Disease_level_full_text": "Oesophageal squamous cell carcinoma (OSCC) OPDIVO in combination with ipilimumab is indicated for the first-line treatment of adult patients with unresectable advanced, recurrent or metastatic oesophageal squamous cell carcinoma with tumour cell PD-L1 expression \u2265 1%. OPDIVO in combination with fluoropyrimidine- and platinum-based combination chemotherapy is indicated for the first-line treatment of adult patients with unresectable advanced, recurrent or metastatic oesophageal squamous cell carcinoma with tumour cell PD-L1 expression \u2265 1%. OPDIVO as monotherapy is indicated for the treatment of adult patients with unresectable advanced, recurrent or metastatic oesophageal squamous cell carcinoma after prior fluoropyrimidine- and platinum-based combination chemotherapy.",
        "Indication #": 3,
        "Indication_text": "OPDIVO as monotherapy is indicated for the treatment of adult patients with unresectable advanced, recurrent or metastatic oesophageal squamous cell carcinoma after prior fluoropyrimidine- and platinum-based combination chemotherapy",
        "Treatment line": "Second line",
        "Treatment modality": "Monotherapy",
        "Population": "Adult",
        "Disease + sybtypes": "unresectable advanced, recurrent or metastatic oesophageal squamous cell carcinoma"
    },
    {
        "Primary Disease_category": "Oesophageal or gastro-oesophageal junction cancer (OC or GEJC)",
        "Disease_level_full_text": "Adjuvant treatment of oesophageal or gastro-oesophageal junction cancer (OC or GEJC) OPDIVO as monotherapy is indicated for the adjuvant treatment of adult patients with oesophageal or gastro-oesophageal junction cancer who have residual pathologic disease following prior neoadjuvant chemoradiotherapy",
        "Indication #": 1,
        "Indication_text": "OPDIVO as monotherapy is indicated for the adjuvant treatment of adult patients with oesophageal or gastro-oesophageal junction cancer who have residual pathologic disease following prior neoadjuvant chemoradiotherapy",
        "Treatment line": "Second line",
        "Treatment modality": "Monotherapy,Adjuvant",
        "Population": "Adult",
        "Disease + sybtypes": "oesophageal or gastro-oesophageal junction cancer who have residual pathologic disease"
    },
    {
        "Primary Disease_category": "Gastric, gastro-oesophageal junction (GEJ) or oesophageal adenocarcinoma",
        "Disease_level_full_text": "Gastric, gastro-oesophageal junction (GEJ) or oesophageal adenocarcinoma OPDIVO in combination with fluoropyrimidine- and platinum-based combination chemotherapy is indicated for the first-line treatment of adult patients with HER2-negative advanced or metastatic gastric, gastro-oesophageal junction or oesophageal adenocarcinoma whose tumours express PD-L1 with a combined positive score (CPS) \u2265 5",
        "Indication #": 1,
        "Indication_text": "OPDIVO in combination with fluoropyrimidine- and platinum-based combination chemotherapy is indicated for the first-line treatment of adult patients with HER2-negative advanced or metastatic gastric, gastro-oesophageal junction or oesophageal adenocarcinoma whose tumours express PD-L1 with a combined positive score (CPS) \u2265 5",
        "Treatment line": "First line",
        "Treatment modality": "Combination",
        "Population": "Adult",
        "Disease + sybtypes": "HER2-negative advanced or metastatic gastric, gastro-oesophageal junction or oesophageal adenocarcinoma"
    },
    {
        "Primary Disease_category": "Hepatocellular carcinoma (HCC)",
        "Disease_level_full_text": "OPDIVO in combination with ipilimumab is indicated for the first-line treatment of adult patients with unresectable or advanced hepatocellular carcinoma",
        "Indication #": 1,
        "Indication_text": "OPDIVO in combination with ipilimumab is indicated for the first-line treatment of adult patients with unresectable or advanced hepatocellular carcinoma",
        "Treatment line": "First line",
        "Treatment modality": "Combination",
        "Population": "Adult",
        "Disease + sybtypes": "unresectable or advanced hepatocellular carcinoma"
    },
    {
        "Primary Disease_category": "Neovascular (wet) age-related macular degeneration (AMD)",
        "Disease_level_full_text": "Lucentis is indicated in adults for: The treatment of neovascular (wet) age-related macular degeneration (AMD)",
        "Indication #": 1,
        "Indication_text": "Lucentis is indicated in adults for: The treatment of neovascular (wet) age-related macular degeneration (AMD)",
        "Treatment line": "_",
        "Treatment modality": "_",
        "Population": "Adult",
        "Disease + sybtypes": "neovascular (wet) age-related macular degeneration (AMD)"
    }
]
"""  # Replace with your actual prompt

# Function to clean JSON response
# Function to clean JSON response
def clean_json_response(response_text):
    """
    Clean the response text by removing ```
    """
    cleaned = response_text.strip()
    
    # Define markers using string concatenation to avoid syntax issues
    json_marker = "`" + "`" + "`" + "json"
    code_marker = "`" + "`" + "`"
    
    # Remove ```json or ```
    if cleaned.startswith(json_marker):
        cleaned = cleaned[len(json_marker):]
    elif cleaned.startswith(code_marker):
        cleaned = cleaned[len(code_marker):]
    
    # Remove ``` at the end
    if cleaned.endswith(code_marker):
        cleaned = cleaned[:-len(code_marker)]
    
    return cleaned.strip()



# Function to call Gemini API
def call_gemini_api(text_data, prompt):
    """
    Call the Gemini API with the provided text and prompt
    """
    PROJECT_ID = "ybrant-gemini-vertexai"  # Replace with your project ID
    LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
    
    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION
    )
    
    contents = [text_data, prompt]
    
    generate_config = types.GenerateContentConfig(
        temperature=0,
        thinking_config=types.ThinkingConfig(
            thinking_budget=2500
        )
    )
    
    model_name = "gemini-2.5-flash"
    
    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=generate_config
    )
    
    return response.text

# Extract Info button
if st.button("🔍 Extract Info", type="primary", use_container_width=True):
    if not st.session_state.credentials_loaded:
        st.warning("⚠️ Please upload your credentials JSON file first.")
    elif not data_input.strip():
        st.warning("⚠️ Please paste some text in the input box.")
    else:
        with st.spinner("🔄 Extracting information using Gemini AI..."):
            try:
                # Call Gemini API
                raw_response = call_gemini_api(data_input, cdp_ema_prompt)
                
                # Clean the response (remove ```
                cleaned_response = clean_json_response(raw_response)
                
                # Parse JSON
                parsed_json = json.loads(cleaned_response)
                st.session_state.extracted_data = parsed_json
                
            except json.JSONDecodeError as e:
                st.error(f"❌ Error parsing JSON response: {str(e)}")
                st.text("Raw response:")
                st.code(raw_response)
            except Exception as e:
                st.error(f"❌ Error during extraction: {str(e)}")

# Display extracted data
# Display extracted data
if st.session_state.extracted_data is not None:
    st.divider()
    st.subheader("📊 Extracted Information")
    
    # Check if data is a list or dict and handle accordingly
    extracted_data = st.session_state.extracted_data
    
    # If it's a list (array of indications)
    if isinstance(extracted_data, list):
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 🗂️ Interactive JSON View")
            st.json(extracted_data, expanded=True)
        
        with col2:
            st.markdown("#### 📋 Formatted Details")
            
            # Loop through each indication in the list
            for idx, item in enumerate(extracted_data):
                if isinstance(item, dict):
                    # Create expander title from Primary Disease_category and Indication #
                    disease_cat = item.get("Primary Disease_category", f"Item {idx + 1}")
                    indication_num = item.get("Indication #", "")
                    expander_title = f"{disease_cat} - Indication #{indication_num}" if indication_num else disease_cat
                    
                    with st.expander(f"**{expander_title}**", expanded=False):
                        for key, value in item.items():
                            if key not in ["Primary Disease_category", "Indication #"]:  # Already in title
                                st.markdown(f"**{key.replace('_', ' ').title()}:**")
                                if isinstance(value, list):
                                    for v in value:
                                        st.markdown(f"  - {v}")
                                else:
                                    st.write(value)
                else:
                    st.write(item)
    
    # If it's a dictionary (single object)
    elif isinstance(extracted_data, dict):
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 🗂️ Interactive JSON View")
            st.json(extracted_data, expanded=True)
        
        with col2:
            st.markdown("#### 📋 Formatted Details")
            
            for key, value in extracted_data.items():
                with st.expander(f"**{key.replace('_', ' ').title()}**", expanded=True):
                    if isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                for k, v in item.items():
                                    st.markdown(f"- **{k}**: {v}")
                            else:
                                st.markdown(f"- {item}")
                    elif isinstance(value, dict):
                        for k, v in value.items():
                            st.markdown(f"- **{k}**: {v}")
                    else:
                        st.write(value)
    
    # Fallback for other types
    else:
        st.json(extracted_data)
    
    # Download button for the extracted JSON
    st.divider()
    st.download_button(
        label="📥 Download Extracted JSON",
        data=json.dumps(st.session_state.extracted_data, indent=2),
        file_name="extracted_ema_data.json",
        mime="application/json",
        use_container_width=True
    )


# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 10px;">
    <small>EMA Extraction Tool | Powered by Google Gemini AI</small>
</div>
""", unsafe_allow_html=True)
