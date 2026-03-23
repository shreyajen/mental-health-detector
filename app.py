import streamlit as st
import pickle
import numpy as np

# ── Page config (MUST be first) ───────────────────────────
st.set_page_config(
    page_title="Mental Health Text Analyzer",
    page_icon="🧠",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
/* Hero banner */
.hero {
    background: linear-gradient(135deg, #7F77DD 0%, #D4537E 50%, #D85A30 100%);
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1.5rem;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.2);
    color: #fff;
    font-size: 12px;
    padding: 4px 14px;
    border-radius: 20px;
    margin-bottom: 1rem;
}
.hero h1 {
    color: #fff !important;
    font-size: 28px !important;
    font-weight: 700 !important;
    margin: 0 0 0.5rem !important;
}
.hero p {
    color: rgba(255,255,255,0.85);
    font-size: 14px;
    margin: 0;
}

/* Result pill */
.result-pill {
    display: inline-block;
    padding: 8px 22px;
    border-radius: 20px;
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 1.2rem;
}

/* Bar container */
.bar-wrap { margin-bottom: 10px; }
.bar-label-row {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    color: #888;
    margin-bottom: 4px;
}
.bar-track {
    height: 10px;
    background: #f0f0f0;
    border-radius: 5px;
    overflow: hidden;
}
.bar-fill {
    height: 100%;
    border-radius: 5px;
    transition: width 0.6s ease;
}

/* Cards */
.card {
    background: white;
    border-radius: 16px;
    border: 1px solid #f0f0f0;
    padding: 1.4rem;
    margin-bottom: 1.2rem;
}

/* Footer */
.footer {
    text-align: center;
    font-size: 12px;
    color: #aaa;
    padding: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()

# ── Hero banner ───────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">ML · NLP · Mental Health</div>
    <h1>🧠 Mental Health Text Analyzer</h1>
    <p>Detect mental health signals in text using machine learning trained on 52,000+ Reddit posts</p>
</div>
""", unsafe_allow_html=True)

# ── Warning ───────────────────────────────────────────────
st.warning("⚠️ This is an educational ML project — not a medical tool. Please seek professional help if needed.")

# ── Input ─────────────────────────────────────────────────
st.subheader("Enter text to analyze")
user_input = st.text_area(
    label="Paste or type any text below:",
    placeholder="e.g. I've been feeling really overwhelmed and can't sleep lately...",
    height=160
)

# ── Color map ─────────────────────────────────────────────
color_map = {
    "Normal":               ("linear-gradient(135deg,#1D9E75,#5DCAA5)", "#EAF3DE", "#085041"),
    "Depression":           ("linear-gradient(135deg,#D85A30,#F0997B)", "#FAECE7", "#712B13"),
    "Anxiety":              ("linear-gradient(135deg,#BA7517,#EF9F27)", "#FAEEDA", "#633806"),
    "Suicidal":             ("linear-gradient(135deg,#A32D2D,#E24B4A)", "#FCEBEB", "#501313"),
    "Stress":               ("linear-gradient(135deg,#BA7517,#EF9F27)", "#FAEEDA", "#633806"),
    "Bipolar":              ("linear-gradient(135deg,#534AB7,#AFA9EC)", "#EEEDFE", "#26215C"),
    "Personality disorder": ("linear-gradient(135deg,#534AB7,#AFA9EC)", "#EEEDFE", "#26215C"),
}

# ── Predict ───────────────────────────────────────────────
if st.button("Analyze text", type="primary"):
    if user_input.strip() == "":
        st.error("Please enter some text first.")
    else:
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]
        probabilities = model.predict_proba(input_vec)[0]
        classes = model.classes_

        bar_gradient, pill_bg, pill_text = color_map.get(
            prediction, ("linear-gradient(135deg,#888,#aaa)", "#f0f0f0", "#333")
        )

        st.divider()

        # ── Result pill ───────────────────────────────────
        st.markdown(f"""
        <div style="margin-bottom:0.5rem;font-size:13px;color:#888;">Detected signal</div>
        <div class="result-pill" style="background:{pill_bg};color:{pill_text};">
            ✓ {prediction}
        </div>
        """, unsafe_allow_html=True)

        # ── Confidence bars ───────────────────────────────
        st.markdown("<div style='font-size:15px;font-weight:600;margin:1rem 0 0.8rem;'>Confidence scores</div>",
                    unsafe_allow_html=True)

        sorted_probs = sorted(zip(classes, probabilities), key=lambda x: x[1], reverse=True)

        for label, prob in sorted_probs:
            bar_grad = color_map.get(label, ("linear-gradient(135deg,#888,#aaa)", "", ""))[0]
            pct = round(prob * 100, 1)
            st.markdown(f"""
            <div class="bar-wrap">
                <div class="bar-label-row">
                    <span style="color:#555;font-weight:500;">{label}</span>
                    <span style="font-weight:600;">{pct}%</span>
                </div>
                <div class="bar-track">
                    <div class="bar-fill" style="width:{pct}%;background:{bar_grad};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Model stats ───────────────────────────────────
        st.divider()
        col1, col2, col3 = st.columns(3)
        col1.metric("Model accuracy", "76%")
        col2.metric("Training samples", "42,144")
        col3.metric("Categories", "7")

# ── Footer ────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built with scikit-learn + Streamlit · Mental Health NLP Classifier
</div>
""", unsafe_allow_html=True)