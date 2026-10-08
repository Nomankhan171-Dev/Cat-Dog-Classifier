
from pathlib import Path
import joblib
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(page_title="Cat & Dog Classifier", page_icon="🐾", layout="centered")

CLASSES = ["Cat", "Dog"]
MODEL_PATH = Path("model/cat_dog_classifier.pkl")

st.markdown("""
<style>
.stApp{
    background:
        radial-gradient(circle at 10% 0%, rgba(37,99,235,.16), transparent 30%),
        radial-gradient(circle at 100% 12%, rgba(6,182,212,.10), transparent 25%),
        #07111f;
    color:#ffffff !important;
}
[data-testid="stHeader"]{background:transparent;}
.block-container{max-width:960px;padding-top:2rem;padding-bottom:3rem;}
.hero{
    border:1px solid #203653;
    background:linear-gradient(180deg,rgba(15,23,42,.96),rgba(8,17,31,.96));
    border-radius:24px;
    padding:28px;
    margin-bottom:20px;
    box-shadow:0 18px 55px rgba(0,0,0,.25);
}
.badges{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px;}
.badge{
    background:#10243a;
    border:1px solid #24496f;
    color:#22d3ee;
    padding:7px 12px;
    border-radius:9px;
    font-size:.76rem;
    font-weight:800;
    letter-spacing:.04em;
}
.hero h1{color:#ffffff !important;margin:0 0 8px;font-size:2.2rem;}
.hero p{color:#cbd5e1 !important;line-height:1.65;margin:0;font-size:1.02rem;}
.section-title{color:#ffffff !important;font-size:1.2rem;font-weight:800;margin:14px 0 8px;}
.info-card{
    background:#0d1728;
    border:1px solid #223652;
    border-radius:18px;
    padding:18px;
    margin:10px 0;
}
.metric-label{color:#9fb4cf !important;font-size:.82rem;}
.metric-value{color:#ffffff !important;font-size:1.55rem;font-weight:800;}
.small{color:#8fa8c7 !important;font-size:.82rem;}
[data-testid="stMarkdownContainer"] *{color:#ffffff !important;}
[data-testid="stFileUploader"]{
    background:#0d1728;
    border:1px dashed #35527d;
    border-radius:16px;
    padding:10px;
}
.stButton>button{
    width:100%;
    border:0;
    border-radius:12px;
    font-weight:800;
    background:linear-gradient(135deg,#2563eb,#06b6d4);
    color:white !important;
    padding:.7rem 1rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="badges">
    <span class="badge">MACHINE LEARNING</span>
    <span class="badge">STREAMLIT</span>
    <span class="badge">PYTHON</span>
  </div>
  <h1>Cat & Dog Classifier</h1>
  <p>
    Upload a pet image and classify it as a Cat or Dog.
    Built as a clean portfolio project using Python, Streamlit, and Machine Learning.
  </p>
</div>
""", unsafe_allow_html=True)

def extract_features(image):
    img = ImageOps.exif_transpose(image).convert("RGB").resize((224, 224))
    arr = np.asarray(img).astype(np.float32) / 255.0
    gray = arr.mean(axis=2)

    means = arr.mean(axis=(0,1))
    stds = arr.std(axis=(0,1))
    brightness = float(gray.mean())
    contrast = float(gray.std())
    saturation = float((arr.max(axis=2) - arr.min(axis=2)).mean())

    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()
    edge_strength = float(gx + gy)

    center = gray[56:168,56:168].mean()
    outer = np.concatenate([
        gray[:40,:].ravel(),
        gray[-40:,:].ravel(),
        gray[:, :40].ravel(),
        gray[:, -40:].ravel()
    ]).mean()

    return np.array([
        *means.tolist(),
        *stds.tolist(),
        brightness,
        contrast,
        saturation,
        edge_strength,
        float(center - outer)
    ], dtype=np.float32).reshape(1, -1)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None

def demo_predict(features):
    x = features.flatten()
    seed = int(abs(float(np.dot(x, np.arange(1, len(x)+1)))) * 100000) % 100000
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.2, 1.0, 2)
    raw += np.array([
        x[7] * 0.35 + abs(x[10]) * 0.20,
        x[8] * 0.35 + x[9] * 0.20
    ])
    return raw / raw.sum()

model = load_model()

if model is None:
    st.info("Demo mode is active. Add `model/cat_dog_classifier.pkl` to use a trained machine-learning model.")
else:
    st.success("Trained Cat & Dog model loaded successfully.")

st.markdown('<div class="section-title">Upload Pet Image</div>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Choose a JPG, JPEG or PNG image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded pet image", use_container_width=True)

    if st.button("Classify Image"):
        features = extract_features(image)

        if model is not None and hasattr(model, "predict_proba"):
            probs = model.predict_proba(features)[0]
            names = list(getattr(model, "classes_", CLASSES))
            pairs = sorted(zip(names, probs), key=lambda x: x[1], reverse=True)
            source = "Trained model"
        else:
            probs = demo_predict(features)
            pairs = sorted(zip(CLASSES, probs), key=lambda x: x[1], reverse=True)
            source = "Demo fallback"

        best, conf = pairs[0]

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f'<div class="info-card"><div class="metric-label">Predicted Class</div>'
                f'<div class="metric-value">{best}</div></div>',
                unsafe_allow_html=True
            )
        with c2:
            st.markdown(
                f'<div class="info-card"><div class="metric-label">Confidence</div>'
                f'<div class="metric-value">{conf*100:.1f}%</div></div>',
                unsafe_allow_html=True
            )

        st.markdown('<div class="section-title">Class Probabilities</div>', unsafe_allow_html=True)
        for name, prob in pairs:
            st.write(f"**{name}** — {prob*100:.1f}%")
            st.progress(float(prob))

        st.caption(f"Prediction source: {source}")

        if model is None:
            st.warning(
                "This is a portfolio demo prediction. Train the included model for meaningful Cat/Dog predictions."
            )
else:
    st.markdown(
        '<div class="info-card"><b>Supported classes</b><br>'
        '<span class="small">Cat • Dog</span></div>',
        unsafe_allow_html=True
    )

with st.expander("How this project works"):
    st.write(
        "The app extracts image features and sends them to a scikit-learn classifier. "
        "Use the included training script with labeled Cat and Dog images to create a trained model."
    )

st.caption("Cat & Dog Classifier • Python • Streamlit • Machine Learning")
