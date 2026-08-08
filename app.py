import os
import sys
import traceback
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import gdown
from flask import Flask, render_template, request

from symptoms import SYMPTOM_MAP, CHAT_KEYWORDS
from diseases_data import DISEASE_INFO

app = Flask(__name__)

# ==================== CONFIG ====================
# REPLACE THESE WITH YOUR ACTUAL GOOGLE DRIVE FILE IDs
MODEL_FILE_ID = "10IiRx52OXA36UJ6yYbMw8OTdN0E4MGcv"      # <-- YOUR MODEL FILE ID
BUNDLE_FILE_ID = "1jcY9npYVjtY4Vtc2PfJz7drSMWJHNgLJ"     # <-- YOUR BUNDLE FILE ID

MODEL_PATH = "/tmp/xgb_top200_model.pkl"
BUNDLE_PATH = "/tmp/xgb_top200_bundle.pkl"

# ==================== STARTUP: DOWNLOAD & LOAD MODEL ====================
_model = None
_bundle = None
_startup_error = None

def download_file(file_id, output_path):
    if os.path.exists(output_path):
        print(f"✓ Already exists: {output_path}")
        return True
    print(f"⬇️ Downloading from Drive: {file_id}")
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False, fuzzy=True)
        print(f"✓ Saved: {output_path} ({os.path.getsize(output_path)/1024/1024:.1f} MB)")
        return True
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False

print("=" * 60)
print("STARTING MediDiagnose AI")
print("=" * 60)

try:
    # Download both files
    ok1 = download_file(MODEL_FILE_ID, MODEL_PATH)
    ok2 = download_file(BUNDLE_FILE_ID, BUNDLE_PATH)
    
    if not ok1 or not ok2:
        raise Exception("Model files failed to download")
    
    # Load into memory
    print("⏳ Loading model into RAM...")
    _model = joblib.load(MODEL_PATH)
    _bundle = joblib.load(BUNDLE_PATH)
    print(f"✓ Model ready. Classes: {len(_bundle['label_encoder'].classes_)}")
    
except Exception as e:
    _startup_error = str(e)
    print("✗ STARTUP FAILED:")
    traceback.print_exc()

# ==================== HELPERS ====================
def symptoms_to_df(selected_symptoms):
    cols = _bundle['columns']
    cat_cols = set(_bundle['cat_cols'])
    medians = _bundle['num_medians']
    
    data = {}
    for col in cols:
        data[col] = [0 if col in cat_cols else medians.get(col, 0.0)]
    
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

# ==================== ROUTES ====================
@app.route('/')
def index():
    if _startup_error:
        return f"<h1>Server Error</h1><p>Startup failed: {_startup_error}</p><p>Check Render Logs.</p>", 500
    return render_template('index.html', symptoms=list(SYMPTOM_MAP.keys()))

@app.route('/predict', methods=['POST'])
def predict():
    if _startup_error:
        return render_template('index.html', symptoms=list(SYMPTOM_MAP.keys()), 
                               error="Server startup failed. Check logs.")
    
    if _model is None or _bundle is None:
        return render_template('index.html', symptoms=list(SYMPTOM_MAP.keys()), 
                               error="Model is still loading. Please wait 30 seconds and refresh.")
    
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
        
        print(f"🩺 Predicting for {len(selected)} symptoms...")
        X_input = symptoms_to_df(selected)
        dmatrix = xgb.DMatrix(X_input)
        proba = _model.predict(dmatrix)[0]
        
        model_le = _bundle['model_le']
        name_le = _bundle['label_encoder']
        
        top5_idx = np.argsort(proba)[-5:][::-1]
        results = []
        
        for rank, idx in enumerate(top5_idx, 1):
            orig_label = model_le.inverse_transform([idx])[0]
            disease_name = name_le.inverse_transform([orig_label])[0]
            clean_name = disease_name.replace('_', ' ').title()
            
            info = DISEASE_INFO.get(disease_name, {
                "title": f"Based on your symptoms, you may have {clean_name}.",
                "lines": [
                    f"{clean_name} is a medical condition that requires professional evaluation.",
                    "Please consult a licensed healthcare provider for accurate diagnosis."
                ]
            })
            
            results.append({
                'rank': rank,
                'disease': clean_name,
                'title': info['title'],
                'lines': info['lines'],
                'search_url': f"https://www.google.com/search?q={disease_name.replace(' ', '+')}+symptoms+treatment"
            })
        
        selected_readable = [k for k, v in SYMPTOM_MAP.items() if v in selected]
        print(f"✓ Prediction complete: {results[0]['disease']}")
        return render_template('results.html', results=results, selected_symptoms=selected_readable)
    
    except Exception as e:
        print("✗ PREDICTION ERROR:")
        traceback.print_exc()
        return render_template('index.html', symptoms=list(SYMPTOM_MAP.keys()), 
                               error="Something went wrong. Please try again.")

@app.route('/health')
def health():
    return {"status": "ok", "model_loaded": _model is not None}

if __name__ == '__main__':
    app.run(debug=True)
