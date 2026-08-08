import os
import sys
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import gdown
from flask import Flask, render_template, request, jsonify

# Import your generated files
from symptoms import SYMPTOM_MAP, CHAT_KEYWORDS
from diseases_data import DISEASE_INFO

app = Flask(__name__)

# ==================== GOOGLE DRIVE CONFIG ====================
# REPLACE WITH YOUR FILE IDs (from shareable link: https://drive.google.com/file/d/FILE_ID/view)
MODEL_FILE_ID = "10IiRx52OXA36UJ6yYbMw8OTdN0E4MGcv"      # xgb_top200_model.pkl
BUNDLE_FILE_ID = "1jcY9npYVjtY4Vtc2PfJz7drSMWJHNgLJ"    # xgb_top200_bundle.pkl

MODEL_PATH = "/tmp/xgb_top200_model.pkl"
BUNDLE_PATH = "/tmp/xgb_top200_bundle.pkl"

# ==================== LAZY LOAD MODEL ====================
_model = None
_bundle = None

def download_if_needed(file_id, output_path):
    if not os.path.exists(output_path):
        print(f"Downloading {os.path.basename(output_path)}...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False, fuzzy=True)
        print(f"Saved: {output_path}")

def get_model():
    global _model, _bundle
    if _model is None or _bundle is None:
        download_if_needed(MODEL_FILE_ID, MODEL_PATH)
        download_if_needed(BUNDLE_FILE_ID, BUNDLE_PATH)
        _model = joblib.load(MODEL_PATH)
        _bundle = joblib.load(BUNDLE_PATH)
    return _model, _bundle

# ==================== HELPERS ====================
def symptoms_to_df(selected_symptoms, bundle):
    cols = bundle['columns']
    cat_cols = set(bundle['cat_cols'])
    medians = bundle['num_medians']
    
    data = {}
    for col in cols:
        if col in cat_cols:
            data[col] = [0]
        else:
            data[col] = [medians.get(col, 0.0)]
    
    for sym_key in selected_symptoms:
        if sym_key in data:
            data[sym_key][0] = 1.0
    
    return pd.DataFrame(data)

def parse_chat_input(text):
    text = text.lower()
    found = []
    for keyword, col_key in CHAT_KEYWORDS.items():
        if keyword in text:
            found.append(col_key)
    return list(set(found))

def get_disease_info(disease_name):
    if disease_name in DISEASE_INFO:
        return DISEASE_INFO[disease_name]
    return {
        "title": f"Based on your symptoms, you may have {disease_name.replace('_', ' ').title()}.",
        "lines": [
            f"{disease_name.replace('_', ' ').title()} is a medical condition that may require professional evaluation.",
            "We recommend consulting a healthcare provider for proper testing and diagnosis."
        ]
    }

# ==================== ROUTES ====================
@app.route('/')
def index():
    return render_template('index.html', symptoms=list(SYMPTOM_MAP.keys()))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        method = request.form.get('input_method', 'checkbox')
        chat_text = request.form.get('chat_text', '').strip()
        selected = request.form.getlist('symptoms')
        
        if method == 'chat' and chat_text:
            selected = parse_chat_input(chat_text)
        else:
            selected = [SYMPTOM_MAP.get(s, s) for s in selected if s in SYMPTOM_MAP]
        
        if not selected:
            return render_template('index.html', symptoms=list(SYMPTOM_MAP.keys()), 
                                   error="Please select or describe at least one symptom.")
        
        model, bundle = get_model()
        X_input = symptoms_to_df(selected, bundle)
        dmatrix = xgb.DMatrix(X_input)
        proba = model.predict(dmatrix)[0]
        
        model_le = bundle['model_le']
        name_le = bundle['label_encoder']
        
        top5_idx = np.argsort(proba)[-5:][::-1]
        results = []
        
        for rank, idx in enumerate(top5_idx, 1):
            orig_label = model_le.inverse_transform([idx])[0]
            disease_name = name_le.inverse_transform([orig_label])[0]
            info = get_disease_info(disease_name)
            
            results.append({
                'rank': rank,
                'disease': disease_name.replace('_', ' ').title(),
                'title': info['title'],
                'lines': info['lines'],
                'search_url': f"https://www.google.com/search?q={disease_name.replace(' ', '+')}+symptoms+treatment"
            })
        
        selected_readable = [k for k, v in SYMPTOM_MAP.items() if v in selected]
        return render_template('results.html', results=results, 
                               selected_symptoms=selected_readable)
    
    except Exception as e:
        print("ERROR:", str(e))
        import traceback
        traceback.print_exc()
        return render_template('index.html', symptoms=list(SYMPTOM_MAP.keys()), 
                               error="Something went wrong. Please try again or contact support.")

@app.route('/health')
def health():
    return jsonify({"status": "ok"})

# Vercel entry point
if __name__ == '__main__':
    app.run(debug=True)
