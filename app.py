import streamlit as st
import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import gdown
import time

from symptoms import SYMPTOM_MAP, CHAT_KEYWORDS
from diseases_data import DISEASE_INFO

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="MediDiagnose AI — Intelligent Symptom Checker",
    page_icon="⚕️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    }

    /* Header */
    .med-header {
        text-align: center;
        padding: 2rem 0 1.5rem 0;
    }
    .med-logo {
        font-size: 3rem;
        margin-bottom: 0.5rem;
        animation: pulse 2s ease infinite;
        display: inline-block;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.1); }
    }
    .med-title {
        font-size: 2.4rem;
        font-weight: 700;
        letter-spacing: -1px;
        background: linear-gradient(135deg, #0ea5e9, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .med-title span {
        font-weight: 300;
        opacity: 0.9;
    }
    .med-tagline {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 0.5rem;
        font-weight: 300;
    }

    /* Intro card */
    .intro-card {
        background: rgba(30, 41, 59, 0.75);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border-left: 4px solid #0ea5e9;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }
    .intro-card p {
        color: #cbd5e1;
        font-size: 1.05rem;
        margin: 0;
        line-height: 1.6;
    }

    /* Tabs override */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.05);
        border-radius: 14px;
        padding: 14px 24px;
        border: 1px solid transparent;
        color: #94a3b8;
        font-weight: 500;
        font-size: 0.95rem;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255,255,255,0.1);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
        color: white !important;
        border-color: rgba(255,255,255,0.2) !important;
        box-shadow: 0 8px 30px rgba(14,165,233,0.3);
    }

    /* Inputs */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: rgba(0,0,0,0.25);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 14px;
        color: #f8fafc;
        font-size: 0.95rem;
        padding: 14px 18px;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #0ea5e9;
        box-shadow: 0 0 0 3px rgba(14,165,233,0.15);
    }

    /* Multiselect */
    .stMultiSelect > div > div {
        background: rgba(0,0,0,0.2);
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .stMultiSelect [data-baseweb="tag"] {
        background: linear-gradient(135deg, #0ea5e9, #6366f1);
        border-radius: 10px;
        color: white;
    }

    /* Analyze Button */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        border: none;
        border-radius: 16px;
        padding: 18px 32px;
        font-size: 1.15rem;
        font-weight: 700;
        width: 100%;
        letter-spacing: 1px;
        text-transform: uppercase;
        transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
        box-shadow: 0 10px 40px rgba(16,185,129,0.3);
        margin-top: 1rem;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 20px 60px rgba(16,185,129,0.4);
    }
    div.stButton > button:first-child:active {
        transform: translateY(-1px) scale(0.98);
    }

    /* Secondary Button */
    div.stButton > button[kind="secondary"]:first-child {
        background: rgba(255,255,255,0.08);
        color: #cbd5e1;
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 14px;
        font-weight: 600;
    }
    div.stButton > button[kind="secondary"]:first-child:hover {
        background: rgba(255,255,255,0.15);
    }

    /* Results */
    .results-title {
        text-align: center;
        font-size: 1.6rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 2rem 0 0.5rem 0;
    }
    .results-subtitle {
        text-align: center;
        color: #94a3b8;
        margin-bottom: 1.5rem;
        font-size: 0.95rem;
    }

    /* Result Cards */
    .result-card {
        background: rgba(30, 41, 59, 0.85);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        position: relative;
        overflow: hidden;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .result-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 60px rgba(0,0,0,0.5);
    }
    .rank-1 { border-left: 5px solid #fbbf24; background: linear-gradient(135deg, rgba(251,191,36,0.08), rgba(30,41,59,0.9)); }
    .rank-2 { border-left: 5px solid #94a3b8; background: linear-gradient(135deg, rgba(148,163,184,0.08), rgba(30,41,59,0.9)); }
    .rank-3 { border-left: 5px solid #cd7f32; background: linear-gradient(135deg, rgba(205,127,50,0.08), rgba(30,41,59,0.9)); }
    .rank-4, .rank-5 { border-left: 5px solid #0ea5e9; }

    .rank-badge {
        position: absolute;
        top: 1.2rem;
        right: 1.2rem;
        width: 44px;
        height: 44px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 1.1rem;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .badge-1 { background: linear-gradient(135deg, #fbbf24, #f59e0b); }
    .badge-2 { background: linear-gradient(135deg, #94a3b8, #64748b); }
    .badge-3 { background: linear-gradient(135deg, #cd7f32, #b45309); }
    .badge-4, .badge-5 { background: linear-gradient(135deg, #0ea5e9, #6366f1); }

    .result-headline {
        font-size: 1.15rem;
        font-weight: 600;
        color: #f8fafc;
        margin-bottom: 0.75rem;
        padding-right: 50px;
        line-height: 1.4;
    }
    .result-body {
        color: #cbd5e1;
        font-size: 0.92rem;
        line-height: 1.7;
        margin-bottom: 1rem;
    }
    .result-body p {
        margin: 0 0 0.5rem 0;
    }
    .learn-more {
        display: inline-block;
        padding: 8px 18px;
        background: rgba(14,165,233,0.1);
        border: 1px solid rgba(14,165,233,0.4);
        color: #0ea5e9;
        text-decoration: none;
        border-radius: 10px;
        font-weight: 500;
        font-size: 0.85rem;
        transition: all 0.3s ease;
    }
    .learn-more:hover {
        background: #0ea5e9;
        color: white;
    }

    /* Symptoms Summary */
    .symptoms-summary {
        background: rgba(30, 41, 59, 0.75);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.2rem;
        margin-bottom: 1.5rem;
        border-left: 4px solid #f59e0b;
    }
    .symptoms-summary h3 {
        color: #f8fafc;
        font-size: 1rem;
        margin-bottom: 0.75rem;
    }
    .symptom-tag {
        display: inline-block;
        padding: 6px 14px;
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid rgba(245, 158, 11, 0.3);
        color: #f59e0b;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 4px;
    }

    /* Disclaimer */
    .disclaimer {
        background: linear-gradient(135deg, rgba(239,68,68,0.08), rgba(30,41,59,0.9));
        border: 2px solid rgba(239,68,68,0.25);
        border-radius: 18px;
        padding: 1.5rem;
        margin-top: 2rem;
        display: flex;
        gap: 1rem;
        align-items: flex-start;
    }
    .disclaimer-icon {
        font-size: 2rem;
        flex-shrink: 0;
    }
    .disclaimer-text strong {
        display: block;
        color: #ef4444;
        margin-bottom: 0.4rem;
        font-size: 1.05rem;
    }
    .disclaimer-text p {
        color: #cbd5e1;
        font-size: 0.9rem;
        line-height: 1.6;
        margin: 0;
    }

    /* Footer */
    .med-footer {
        text-align: center;
        padding: 2rem 0;
        color: #64748b;
        font-size: 0.85rem;
    }

    /* Loading */
    .loading-box {
        text-align: center;
        padding: 3rem 1rem;
    }
    .spinner-ring {
        width: 70px;
        height: 70px;
        margin: 0 auto 1.5rem;
        border: 4px solid rgba(14,165,233,0.1);
        border-top-color: #0ea5e9;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    .loading-title {
        font-size: 1.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #0ea5e9, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .loading-quote {
        color: #0ea5e9;
        font-style: italic;
        font-size: 0.95rem;
        min-height: 1.5rem;
        margin-top: 0.8rem;
    }

    /* Action buttons row */
    .action-row {
        display: flex;
        gap: 12px;
        justify-content: center;
        margin: 1.5rem 0;
        flex-wrap: wrap;
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(0,0,0,0.2);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(255,255,255,0.15);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255,255,255,0.25);
    }
/* ==================== PAGE ENTRY ANIMATION ==================== */

@keyframes medPageEnter {
    from {
        opacity: 0;
        transform: translateY(18px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes medFadeIn {
    from {
        opacity: 0;
    }
    to {
        opacity: 1;
    }
}

/* Main page sections */
.med-header {
    animation: medPageEnter 0.8s ease-out both;
}

.intro-card {
    animation: medPageEnter 0.8s ease-out 0.12s both;
}

.stTabs {
    animation: medPageEnter 0.8s ease-out 0.22s both;
}

.symptoms-summary {
    animation: medPageEnter 0.8s ease-out 0.12s both;
}

.results-title {
    animation: medPageEnter 0.8s ease-out 0.18s both;
}

.results-subtitle {
    animation: medFadeIn 0.8s ease-out 0.28s both;
}

/* Disease result cards appear one after another */
.result-card {
    animation: medPageEnter 0.7s ease-out both;
}

.result-card:nth-of-type(1) {
    animation-delay: 0.30s;
}

.result-card:nth-of-type(2) {
    animation-delay: 0.42s;
}

.result-card:nth-of-type(3) {
    animation-delay: 0.54s;
}

.result-card:nth-of-type(4) {
    animation-delay: 0.66s;
}

.result-card:nth-of-type(5) {
    animation-delay: 0.78s;
}

/* Buttons and bottom sections */
.action-row {
    animation: medPageEnter 0.8s ease-out 0.85s both;
}

.disclaimer {
    animation: medPageEnter 0.8s ease-out 0.92s both;
}

.med-footer {
    animation: medFadeIn 1s ease-out 1s both;
}

/* Smooth appearance for text and controls */
.stButton,
.stTextInput,
.stTextArea,
.stMultiSelect {
    animation: medFadeIn 0.7s ease-out 0.25s both;
}

/* Respect users who disable motion */
@media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        animation: none !important;
        transition: none !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ==================== CONFIG ====================
MODEL_FILE_ID = "10IiRx52OXA36UJ6yYbMw8OTdN0E4MGcv"      # ← REPLACE WITH YOUR REAL DRIVE ID
BUNDLE_FILE_ID = "1jcY9npYVjtY4Vtc2PfJz7drSMWJHNgLJ"    # ← REPLACE WITH YOUR REAL DRIVE ID

MODEL_PATH = "/tmp/xgb_top200_model.pkl"
BUNDLE_PATH = "/tmp/xgb_top200_bundle.pkl"

# ==================== SESSION STATE ====================
if 'page' not in st.session_state:
    st.session_state.page = 'input'
if 'results' not in st.session_state:
    st.session_state.results = None
if 'selected_readable' not in st.session_state:
    st.session_state.selected_readable = []
if 'loading' not in st.session_state:
    st.session_state.loading = False

# ==================== LOAD MODEL ====================
@st.cache_resource(show_spinner=False)
def download_and_load_model():
    """Download model from Google Drive and load into memory. Cached for the session."""
    # Try to read from secrets first (more secure)
    try:
        m_id = st.secrets.get("MODEL_FILE_ID", MODEL_FILE_ID)
        b_id = st.secrets.get("BUNDLE_FILE_ID", BUNDLE_FILE_ID)
    except:
        m_id = MODEL_FILE_ID
        b_id = BUNDLE_FILE_ID

    if "YOUR_MODEL" in m_id or "YOUR_BUNDLE" in b_id:
        raise ValueError("Please replace MODEL_FILE_ID and BUNDLE_FILE_ID with your real Google Drive File IDs.")

    if not os.path.exists(MODEL_PATH):
        with st.spinner("⬇️ Downloading AI model from Google Drive (one-time)..."):
            gdown.download(f"https://drive.google.com/uc?id={m_id}", MODEL_PATH, quiet=True, fuzzy=True)

    if not os.path.exists(BUNDLE_PATH):
        with st.spinner("⬇️ Downloading bundle from Google Drive (one-time)..."):
            gdown.download(f"https://drive.google.com/uc?id={b_id}", BUNDLE_PATH, quiet=True, fuzzy=True)

    model = joblib.load(MODEL_PATH)
    bundle = joblib.load(BUNDLE_PATH)
    return model, bundle

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
    clean = disease_name.replace('_', ' ').title()
    return {
        "title": f"{clean}.",
        "lines": [
            f"{clean} is a medical condition that requires professional evaluation and diagnostic testing.",
            "Please consult a licensed healthcare provider for an accurate diagnosis and personalized treatment plan."
        ]
    }

# ==================== HEADER ====================
st.markdown("""
<div class="med-header">
    <div class="med-logo">⚕️</div>
    <h1 class="med-title">MediDiagnose <span>AI</span></h1>
    <p class="med-tagline">Advanced Symptom Analysis & Disease Prediction</p>
</div>
""", unsafe_allow_html=True)

# ==================== INPUT PAGE ====================
if st.session_state.page == 'input':
    st.markdown("""
    <div class="intro-card">
        <p>Select the symptoms you are experiencing to receive an AI-powered disease prediction powered by machine learning.</p>
    </div>
    """, unsafe_allow_html=True)

    # Load model in background (only once)
    try:
        _model, _bundle = download_and_load_model()
        model_ready = True
    except Exception as e:
        model_ready = False
        st.error(f"⚠️ Model Error: {e}")
        st.info("💡 Make sure you replaced the Drive File IDs in app.py and both files are shared as 'Anyone with the link'.")

    # Tabs
    tab1, tab2 = st.tabs(["☑️ Quick Select", "💬 Describe in Words"])

    selected_quick = []
    chat_text = ""

    with tab1:
        search = st.text_input("🔍 Search symptoms...", placeholder="Type to filter symptoms (e.g., fever, headache)", key="search_symptoms")
        all_symptoms = list(SYMPTOM_MAP.keys())
        if search:
            filtered = [s for s in all_symptoms if search.lower() in s.lower()]
        else:
            filtered = all_symptoms

        selected_quick = st.multiselect(
            "Select your symptoms:",
            options=filtered,
            default=[],
            key="symptom_multiselect",
            placeholder="Choose symptoms..."
        )

        if selected_quick:
            st.caption(f"**{len(selected_quick)}** symptom{'s' if len(selected_quick) != 1 else ''} selected")

    with tab2:
        st.markdown("<p style='color:#94a3b8; margin-bottom:0.5rem;'>🤖 Hello! I'm MediDiagnose AI. Please describe all the symptoms you are experiencing in your own words.</p>", unsafe_allow_html=True)
        chat_text = st.text_area(
            "Your description:",
            placeholder="Example: I have fever, headache, and cough for 3 days...",
            height=150,
            key="chat_textarea"
        )

    # Analyze Button
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        analyze_clicked = st.button("🔍 Analyze My Symptoms", use_container_width=True, disabled=not model_ready)

    if analyze_clicked:
        # Determine input method
        if selected_quick:
            selected_symptoms = [SYMPTOM_MAP.get(s, s) for s in selected_quick if s in SYMPTOM_MAP]
            selected_readable = selected_quick
        elif chat_text.strip():
            selected_symptoms = parse_chat_input(chat_text)
            selected_readable = [k for k, v in SYMPTOM_MAP.items() if v in selected_symptoms]
        else:
            st.error("⚠️ Please select or describe at least one symptom before analyzing.")
            st.stop()

        if not selected_symptoms:
            st.error("⚠️ No matching symptoms found. Try selecting from the list or describing differently.")
            st.stop()

        # Show loading animation
        loading_placeholder = st.empty()
        quotes = [
            "Analyzing symptom patterns across 200+ medical conditions...",
            "Cross-referencing with advanced medical databases...",
            "Evaluating probabilistic disease models...",
            "Consulting AI diagnostic neural networks...",
            "Generating personalized health insights...",
            "Comparing against clinical symptom profiles...",
            "Finalizing diagnostic possibilities..."
        ]

        for i, quote in enumerate(quotes[:4]):
            loading_placeholder.markdown(f"""
            <div class="loading-box">
                <div class="spinner-ring"></div>
                <div class="loading-title">MediDiagnose AI</div>
                <p style="color:#94a3b8; margin:0;">Analyzing Your Symptoms</p>
                <div class="loading-quote">{quote}</div>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(0.8)

        # Run prediction
        try:
            X_input = symptoms_to_df(selected_symptoms, _bundle)
            dmatrix = xgb.DMatrix(X_input)
            proba = _model.predict(dmatrix)[0]

            model_le = _bundle['model_le']
            name_le = _bundle['label_encoder']

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

            st.session_state.results = results
            st.session_state.selected_readable = selected_readable
            st.session_state.page = 'results'
            loading_placeholder.empty()
            st.rerun()

        except Exception as e:
            loading_placeholder.empty()
            st.error(f"❌ Analysis failed: {e}")
            st.stop()

# ==================== RESULTS PAGE ====================
else:
    # Selected Symptoms Summary
    if st.session_state.selected_readable:
        tags_html = "".join([f'<span class="symptom-tag">{s}</span>' for s in st.session_state.selected_readable])
        st.markdown(f"""
        <div class="symptoms-summary">
            <h3>📝 Symptoms You Reported</h3>
            <div>{tags_html}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="results-title">🔬 Diagnostic Possibilities</div>', unsafe_allow_html=True)
    st.markdown('<div class="results-subtitle">Here are the top conditions that match your symptoms</div>', unsafe_allow_html=True)

    for result in st.session_state.results:
        rank = result['rank']
        card_html = f"""
        <div class="result-card rank-{rank}">
            <div class="rank-badge badge-{rank}">{rank}</div>
            <div class="result-headline">{result['disease']}</div>
            <div class="result-body">
                <p>{result['lines'][0]}</p>
                <p>{result['lines'][1]}</p>
            </div>
            <a href="{result['search_url']}" target="_blank" class="learn-more">Learn more about {result['disease']} ↗</a>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        <div class="disclaimer-text">
            <strong>Medical Disclaimer</strong>
            <p>Please note that MediDiagnose AI provides predictions based on the selected symptoms and should not be considered a substitute for professional medical advice or diagnosis. Always consult a qualified healthcare provider for accurate diagnosis and treatment.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Action Buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("← Check New Symptoms", use_container_width=True, type="secondary"):
            st.session_state.page = 'input'
            st.session_state.results = None
            st.session_state.selected_readable = []
            st.rerun()
    with col3:
        st.markdown("""
        <a href="https://www.google.com/search?q=find+doctor+near+me" target="_blank" style="text-decoration:none;">
            <button style="width:100%; padding:12px; border-radius:14px; border:none; background:linear-gradient(135deg, #0ea5e9, #6366f1); color:white; font-weight:600; font-size:0.95rem; cursor:pointer; box-shadow:0 8px 30px rgba(14,165,233,0.3);">
                Find a Doctor Near Me
            </button>
        </a>
        """, unsafe_allow_html=True)

# ==================== FOOTER ====================
st.markdown("""
<div class="med-footer">
    <p>MediDiagnose AI © 2026 — Powered by Machine Learning</p>
    <p>Developed by Muhammad Sufian</p>
    <p style="font-size:0.75rem; margin-top:4px;">This tool is for educational purposes only. Not a substitute for professional medical advice.</p>
</div>
""", unsafe_allow_html=True)
