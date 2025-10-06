import streamlit as st
import os
import tempfile
import time
import pandas as pd
import json
import re
from collections import defaultdict
import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import easyocr
import spacy
import difflib
from fastapi import FastAPI, UploadFile, File, HTTPException

# To run the UI: streamlit run this_file.py
# To run the API: uvicorn this_file:app --reload --port 8000

# Medical terms
MEDICAL_TERMS = [
    'Hemoglobin', 'Hemoglobin A1c', 'HbA1c', 'WBC', 'RBC', 'Platelets',
    'Hematocrit', 'MCV', 'MCH', 'MCHC', 'RDW', 'Neutrophils', 'Lymphocytes',
    'Monocytes', 'Eosinophils', 'Basophils', 'Total Protein', 'Albumin',
    'Globulin', 'A/G Ratio', 'Bilirubin', 'ALT', 'AST', 'ALP', 'GGT',
    'Total Cholesterol', 'HDL', 'LDL', 'Triglycerides', 'Glucose', 'Fasting Glucose',
    'Random Glucose', 'Urea', 'Creatinine', 'Uric Acid', 'Sodium', 'Potassium',
    'Chloride', 'Calcium', 'Phosphate', 'Magnesium', 'TSH', 'T3', 'T4',
    'CRP', 'ESR', 'Vitamin D', 'Vitamin B12', 'Ferritin', 'Iron', 'Transferrin',
    'LDH', 'CPK', 'Amylase', 'Lipase', 'Urine Protein', 'Urine Glucose',
    'Hb', 'PT', 'INR', 'APTT', 'Creatine Kinase', 'Blood Urea Nitrogen', 'BUN'
]

SYNONYMS = {
    'hb': 'Hemoglobin', 'hemoglobin a1c': 'Hemoglobin A1c', 'hba1c': 'Hemoglobin A1c',
    'blood glucose': 'Glucose', 'fasting blood sugar': 'Fasting Glucose', 'fbs': 'Fasting Glucose',
    'creatinine serum': 'Creatinine', 'urea nitrogen': 'Blood Urea Nitrogen', 'bun': 'Blood Urea Nitrogen',
    'hdl cholesterol': 'HDL', 'ldl cholesterol': 'LDL', 'triglyceride': 'Triglycerides',
    'total bilirubin': 'Bilirubin', 'alt (sgpt)': 'ALT', 'ast (sgot)': 'AST'
}

all_terms_lower = [term.lower() for term in MEDICAL_TERMS] + list(SYNONYMS.keys())

COMMON_LAB_TESTS = [
    'Hemoglobin', 'Hb', 'RBC', 'WBC', 'Platelets', 'PCV', 'Hematocrit',
    'MCV', 'MCH', 'MCHC', 'RDW', 'Glucose', 'Random Glucose', 'Fasting Glucose',
    'Plasma Glucose', 'RBS', 'Urea', 'Creatinine', 'BUN', 'Uric Acid',
    'Sodium', 'Potassium', 'Chloride', 'Calcium', 'Magnesium', 'Phosphorus',
    'Total Protein', 'Albumin', 'Globulin', 'A/G Ratio', 'Bilirubin', 'AST',
    'ALT', 'ALP', 'GGT', 'Cholesterol', 'HDL', 'LDL', 'Triglycerides', 'TSH',
    'T3', 'T4', 'CRP', 'ESR', 'HbA1c', 'HBA1C', 'Vitamin D', '25-OH Vitamin D',
    'Sodium', 'Potassium', 'Creatinine', 'PT', 'INR', 'APTT', 'B12', 'Folic Acid',
    'Amylase', 'Lipase', 'Total Cholesterol', 'HDL Cholesterol', 'LDL Cholesterol',
    'Serum Iron', 'TIBC', 'Ferritin',
]

COMMON_LAB_TESTS_SET = set(t.lower() for t in COMMON_LAB_TESTS)

# Utils
def parse_number(token_str):
    if token_str is None or not isinstance(token_str, str):
        return None
    s = token_str.strip()
    if not s:
        return None

    if s in ['.', '-', '—', 'ND', 'nd', 'Nil', 'nil', 'Trace', 'trace', 'NA', 'N/A']:
        return None

    m = re.match(r'^([<>])\s*([0-9]*\.?[0-9]+)$', s)
    if m:
        qualifier = m.group(1)
        try:
            val = float(m.group(2))
            return {'value': val, 'qualifier': qualifier}
        except:
            return None

    m = re.match(r'^([0-9]*\.?[0-9]+)\s*[-–—]\s*([0-9]*\.?[0-9]+)$', s)
    if m:
        try:
            v1 = float(m.group(1))
            v2 = float(m.group(2))
            return {'value': (v1 + v2) / 2.0, 'qualifier': '-'}
        except:
            return None

    m = re.match(r'^([0-9]*\.?[0-9]+)$', s)
    if m:
        try:
            return float(m.group(1))
        except:
            return None

    s2 = s.replace(',', '')
    m = re.match(r'^([0-9]*\.?[0-9]+)$', s2)
    if m:
        try:
            return float(m.group(1))
        except:
            return None

    return None

def normalize_unit(unit_str):
    if not unit_str or not isinstance(unit_str, str):
        return ''
    s = unit_str.strip()
    s = s.replace('l', 'L')
    s = s.replace('mgldL', 'mg/dL')
    s = s.replace('mgldl', 'mg/dL')
    s = s.replace('mg/dl', 'mg/dL')
    s = s.replace('mgdl', 'mg/dL')
    s = s.replace('pIUImL', 'pIU/mL')
    s = s.replace('pIUImL', 'pIU/mL')
    s = s.replace('uL', 'µL')
    s = s.replace('ul', 'µL')
    return s

# Preprocess
def expand_canvas(img, padding=50):
    h, w = img.shape[:2]
    top, bottom = padding, padding
    left, right = padding, padding
    new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return new_img

def deskew_image(img, enabled=True):
    if not enabled:
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return deskewed

def preprocess(file_path, deskew_enabled=False):
    output_dir = "temp_images"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        if file_path.endswith('.pdf'):
            images = convert_from_path(file_path, dpi=300, output_folder=output_dir, fmt='png', paths_only=True)
        else:
            images = [file_path]
        
        if not images:
            return []
        
        processed_paths = []
        for img_path in images:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Failed to read image {img_path}")
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            denoised = cv2.medianBlur(gray, 3)
            _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened = cv2.filter2D(binary, -1, kernel)
            expanded = expand_canvas(cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR))
            deskewed = deskew_image(expanded, deskew_enabled)
            h, w = expanded.shape[:2]
            cropped = deskewed[50: h-50, 50: w-50] if deskew_enabled else deskewed
            processed_img = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            processed_pil = Image.fromarray(processed_img)
            processed_pil.save(img_path)
            processed_paths.append(img_path)
        
        return processed_paths
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return []

# OCR
reader = easyocr.Reader(['en'], gpu=False)

def ocr_extract(img_path):
    try:
        result = reader.readtext(img_path, detail=1)  # Ensure detail=1 for bounding box data
        tokens = []
        text = ""
        for detection in result:
            if len(detection) == 3:  # Ensure proper detection format (bbox, text, conf)
                bbox, txt, conf = detection
                left = min(point[0] for point in bbox)
                top = min(point[1] for point in bbox)
                width = max(point[0] for point in bbox) - left
                height = max(point[1] for point in bbox) - top
                tokens.append({
                    'left': left,
                    'top': top,
                    'width': width,
                    'height': height,
                    'text': txt,
                    'conf': conf
                })
                text += txt + " "
        return tokens, text.strip() if text else ""
    except Exception as e:
        print(f"Warning: OCR failed for {img_path}: {e}")
        return [], ""  # Return empty on failure

# Extraction
nlp = spacy.load('en_core_web_sm')

def group_tokens_by_lines(tokens):
    if not isinstance(tokens, list):
        print(f"Warning: tokens is not a list, got {type(tokens)}")
        return []
    lines = defaultdict(list)
    for token in tokens:
        if (isinstance(token, dict) and all(key in token for key in ['left', 'top', 'text', 'conf']) and
            isinstance(token['left'], (int, float)) and isinstance(token['top'], (int, float))):
            top = token['top']
            key = int(round(top / 5.0)) * 5
            lines[key].append(token)
    sorted_lines = sorted(lines.items(), key=lambda x: x[0])
    return [sorted(line[1], key=lambda t: t['left']) for _, line in sorted_lines] if sorted_lines else []

def rule_extract(text, tokens):
    if not isinstance(text, str) or not tokens:
        return {'patient': {}, 'tests': []}
    patient = {}
    tests = []
    lines = group_tokens_by_lines(tokens)
    
    patterns = {
        'name': r'(?:Name|Pt Name|Patient)[:\s]*([A-Za-z\s]+?)(?=\n|Gender|Age|Lab ID|ID|$)',
        'age': r'(?:Age|Age/Sex)[:\s]*(\d+)',
        'gender': r'(?:Gender|Sex)[:\s]*([MF]+)',
        'id': r'(?:ID|Lab ID|Pt\. ID)[:\s]*(\S+)'
    }
    for field, pat in patterns.items():
        match = re.search(pat, text, re.I | re.M)
        if match:
            patient[field] = {'value': match.group(1).strip()}

    patient_end = 0
    for pat in patterns.values():
        m = re.search(pat, text, re.I | re.M)
        if m:
            patient_end = max(patient_end, m.end())
    
    test_pat = r'([A-Za-z\s]+?)(?:\s*[:=]\s*)?([0-9<>\-–—,.]+)\s+([A-Za-z/µL%mgpIU]+)'
    for line in lines:
        line_text = " ".join(t['text'] for t in line if isinstance(t, dict) and 'text' in t)
        match = re.search(test_pat, line_text, re.I)
        if match:
            idx_in_text = text.find(line_text)
            if idx_in_text != -1 and idx_in_text < patient_end:
                continue
            raw_val = match.group(2).strip()
            parsed = parse_number(raw_val)
            if parsed is None:
                continue
            if isinstance(parsed, dict):
                value = parsed.get('value')
                qualifier = parsed.get('qualifier')
            else:
                value = parsed
                qualifier = None
            name = match.group(1).strip()
            if name:
                tests.append({
                    'name': name,
                    'value': value,
                    'unit': normalize_unit(match.group(3).strip()),
                    'qualifier': qualifier
                })
    
    medical_pat = r'([A-Za-z\s]+?)\s+F?\s*([0-9<>\-–—,.]+)\s+(mgldL|pIUImL|%|\S+)'
    for match in re.finditer(medical_pat, text, re.I):
        if match.start() < patient_end:
            continue
        name = match.group(1).strip()
        raw_val = match.group(2).strip()
        parsed = parse_number(raw_val)
        if parsed is None:
            continue
        if isinstance(parsed, dict):
            value = parsed.get('value')
            qualifier = parsed.get('qualifier')
        else:
            value = parsed
            qualifier = None
        if name and name not in [t['name'] for t in tests]:
            tests.append({
                'name': name,
                'value': value,
                'unit': normalize_unit(match.group(3).strip()),
                'qualifier': qualifier
            })
    
    return {'patient': patient, 'tests': tests}

def ml_extract(text):
    if not isinstance(text, str):
        return {'patient': {}, 'tests': []}
    entities = {}
    try:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                entities['name'] = {'value': ent.text}
            elif ent.label_ == 'CARDINAL' and ent.text.isdigit():
                entities['age'] = {'value': int(ent.text)}
    except Exception as e:
        print(f"Warning: ML extraction failed: {e}")
    return {'patient': entities, 'tests': []}

def inference(tokens, text):
    if not isinstance(tokens, list) or not isinstance(text, str):
        return {'patient': {}, 'tests': [], 'patient_table': [], 'tests_table': []}
    rule_ext = rule_extract(text, tokens)
    ml_ext = ml_extract(text)
    
    combined = {'patient': {}, 'tests': rule_ext['tests']}
    for field in ['name', 'age', 'gender', 'id']:
        rule_value = rule_ext['patient'].get(field, {}).get('value')
        ml_value = ml_ext['patient'].get(field, {}).get('value')
        if ml_value is not None:
            combined['patient'][field] = {'value': ml_value}
        elif rule_value is not None:
            combined['patient'][field] = {'value': rule_value}
        if field in combined['patient'] and combined['patient'][field].get('value') is None:
            combined['patient'][field]['needs_review'] = True
    
    canonical_list_lower = [ct.lower() for ct in MEDICAL_TERMS]
    canonical_set = set(canonical_list_lower)
    synonym_map = {k.lower(): v for k, v in SYNONYMS.items()}

    FUZZY_THRESH = 0.86
    MAX_NGRAM = 3
    LOOKAHEAD = 4

    STOPWORDS = set(["page", "sample", "report", "final", "collected", "lab", "id", "name", "gender", "age", "remarks", "test", "results"])

    whitelist_tests = []
    seen = set()

    for line in group_tokens_by_lines(tokens):
        line_texts = [t['text'].strip() for t in line if isinstance(t, dict) and 'text' in t]
        line_lower = [s.lower() for s in line_texts]
        if not line_texts or any(w.lower() in STOPWORDS for w in line_lower) and len(line_lower) < 6:
            continue

        L = len(line_lower)
        i = 0
        while i < L:
            matched_name = None
            matched_span = 1
            for n in range(MAX_NGRAM, 0, -1):
                if i + n > L:
                    continue
                phrase = " ".join(line_lower[i:i+n]).strip()
                if not phrase or len(phrase) <= 2:
                    continue
                if phrase in canonical_set:
                    matched_name = next(ct for ct in MEDICAL_TERMS if ct.lower() == phrase)
                    matched_span = n
                    break
                if phrase in synonym_map:
                    matched_name = synonym_map[phrase]
                    matched_span = n
                    break
                if len(phrase) > 3:
                    close = difflib.get_close_matches(phrase, canonical_list_lower, n=1, cutoff=FUZZY_THRESH)
                    if close:
                        matched_name = next(ct for ct in MEDICAL_TERMS if ct.lower() == close[0])
                        matched_span = n
                        break

            if matched_name:
                value = None
                qualifier = None
                unit = ''
                start_j = i + matched_span
                for j in range(start_j, min(start_j + LOOKAHEAD, L)):
                    parsed = parse_number(line_texts[j])
                    if parsed is None:
                        continue
                    if isinstance(parsed, dict):
                        value = parsed.get('value')
                        qualifier = parsed.get('qualifier')
                    else:
                        value = parsed
                    if j + 1 < L:
                        maybe_unit = line_texts[j+1]
                        if any(ch.isalpha() for ch in maybe_unit) or '%' in maybe_unit or '/' in maybe_unit or 'µ' in maybe_unit:
                            unit = normalize_unit(maybe_unit)
                    break

                if value is not None:
                    key = (matched_name.lower(), unit.lower(), round(float(value), 3) if value is not None else None)
                    if key not in seen:
                        seen.add(key)
                        whitelist_tests.append({
                            'name': matched_name,
                            'value': value,
                            'unit': unit,
                            'qualifier': qualifier
                        })
                i += matched_span
            else:
                i += 1

    combined['tests'] = whitelist_tests or combined['tests']

    if combined['patient']:
        patient_df = pd.DataFrame([(k, v['value']) for k, v in combined['patient'].items()], columns=['Field', 'Value'])
        combined['patient_table'] = patient_df.to_dict('records')
    else:
        combined['patient_table'] = []
    if combined['tests']:
        tests_df = pd.DataFrame(combined['tests'])
        combined['tests_table'] = tests_df.to_dict('records')
    else:
        combined['tests_table'] = []

    return combined

# Streamlit App
st.set_page_config(page_title="Lab Report Digitizer", layout="wide")

st.title("Lab Report Digitization App")
st.markdown("Upload a PDF, JPG, or PNG lab report to extract patient details and test results.")

deskew_enabled = st.checkbox("Enable Deskew (for tilted scans)", value=False)

uploaded_file = st.file_uploader("Choose a file", type=["pdf", "jpg", "png", "jpeg"])

if uploaded_file is not None:
    with st.spinner("Processing the report..."):
        suffix = ".pdf" if uploaded_file.type == "application/pdf" else ".png" if uploaded_file.type == "image/png" else ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            file_path = tmp.name

        start_time = time.time()
        all_images = preprocess(file_path, deskew_enabled=deskew_enabled)
        preprocess_time = time.time() - start_time

        start_time = time.time()
        all_tokens = []
        all_text_parts = []
        for img_path in all_images:
            page_tokens, page_text = ocr_extract(img_path)
            all_tokens.extend(page_tokens)
            all_text_parts.append(page_text)
        formatted_text = " ".join(all_text_parts) if all_text_parts else ""
        ocr_time = time.time() - start_time

        os.unlink(file_path)

    st.write(f"Preprocessing time: {preprocess_time:.2f} seconds")
    st.write(f"OCR time: {ocr_time:.2f} seconds")
    st.success("Processing complete!")

    st.subheader("Preprocessed Images")
    for i, img_path in enumerate(all_images):
        img = Image.open(img_path)
        st.image(img, caption=f"Page {i+1}", use_column_width=True)
        os.unlink(img_path)  # Clean up immediately after display

    st.subheader("Raw OCR Text (All Pages)")
    if formatted_text.strip():
        st.text_area("Full Extracted Text", formatted_text, height=300)
    else:
        st.warning("No text extracted.")

    st.subheader("Tabulated Values")
    extracted = inference(all_tokens, formatted_text)

    st.subheader("Patient Details Table")
    if extracted.get('patient_table'):
        patient_df = pd.DataFrame(extracted['patient_table'])
        edited_patient_df = st.data_editor(patient_df, num_rows="dynamic", use_container_width=True)
    else:
        st.info("No patient details found.")
        edited_patient_df = None

    st.subheader("Test Results Table")
    if extracted.get('tests_table'):
        tests_df = pd.DataFrame(extracted['tests_table'])
        edited_tests_df = st.data_editor(tests_df, num_rows="dynamic", use_container_width=True)
    else:
        st.info("No test results found.")
        edited_tests_df = None

    # Update extracted with edits
    if edited_patient_df is not None:
        edited_patient = {}
        for _, row in edited_patient_df.iterrows():
            edited_patient[row['Field']] = {'value': row['Value']}
        extracted['patient'] = edited_patient
        extracted['patient_table'] = edited_patient_df.to_dict('records')

    if edited_tests_df is not None:
        extracted['tests'] = edited_tests_df.to_dict('records')
        extracted['tests_table'] = edited_tests_df.to_dict('records')

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Confirm and Save"):
            report_id = len(os.listdir('final_reports')) + 1 if os.path.exists('final_reports') else 1
            os.makedirs('final_reports', exist_ok=True)
            os.makedirs('corrections', exist_ok=True)
            correction_path = os.path.join('corrections', f'correction_{report_id:03d}.json')
            final_path = os.path.join('final_reports', f'confirmed_{report_id:03d}.json')
            with open(correction_path, 'w') as f:
                json.dump({'raw_text': formatted_text[:1000], 'corrected': extracted}, f)
            with open(final_path, 'w') as f:
                json.dump(extracted, f, indent=4)
            st.success(f"Saved to {final_path} and {correction_path}!")

    with col2:
        st.download_button(
            label="Download Extracted JSON",
            data=json.dumps(extracted, indent=4),
            file_name="extracted_report.json",
            mime="application/json"
        )

    st.subheader("Train Model (Placeholder)")
    st.write("Training from corrections is not fully implemented in this version. Corrections are saved for future training.")

st.markdown("---")
st.caption("Lab Report Digitization System - Ready for submission.")

# FastAPI Integration
app = FastAPI(title="Lab Report Digitization API")

@app.post("/upload", response_model=dict)
async def upload_report(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.pdf', '.jpg', '.jpeg', '.png')):
        raise HTTPException(status_code=400, detail="Invalid file type. Supported: PDF, JPG, PNG")

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        tmp.write(await file.read())
        file_path = tmp.name

    try:
        all_images = preprocess(file_path, deskew_enabled=False)
        all_tokens = []
        all_text_parts = []
        for img_path in all_images:
            page_tokens, page_text = ocr_extract(img_path)
            all_tokens.extend(page_tokens)
            all_text_parts.append(page_text)
        formatted_text = " ".join(all_text_parts) if all_text_parts else ""
        extracted = inference(all_tokens, formatted_text)
    finally:
        os.unlink(file_path)
        for img_path in all_images:
            if os.path.exists(img_path):
                os.unlink(img_path)

    return extracted