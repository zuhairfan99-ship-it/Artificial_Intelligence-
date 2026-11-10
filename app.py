"""Streamlit interface for the email spam classifier."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "spam_classifier.joblib"
DATA_PATH = ROOT / "data" / "emails.csv"

st.set_page_config(page_title="Email Spam Detector", page_icon="📧", layout="centered")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        from train_model import main
        main()
    return joblib.load(MODEL_PATH)

st.title("📧 Email Spam Detector")
st.write("Paste an email below to check whether it looks like spam or a legitimate message.")
st.caption("Educational machine-learning project using TF-IDF and Logistic Regression.")

try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load or train the model: {exc}")
    st.stop()

email_text = st.text_area(
    "Email text",
    height=220,
    placeholder="Example: Hello, your meeting is confirmed for tomorrow at 10 am..."
)

col1, col2 = st.columns([1, 1])
with col1:
    check = st.button("Check email", type="primary", use_container_width=True)
with col2:
    clear = st.button("Clear text", use_container_width=True)

if clear:
    st.rerun()

if check:
    if len(email_text.strip()) < 5:
        st.warning("Please enter a little more email text.")
    else:
        prediction = model.predict([email_text])[0]
        probabilities = model.predict_proba([email_text])[0]
        classes = list(model.classes_)
        confidence = float(probabilities[classes.index(prediction)])

        st.divider()
        if prediction == "spam":
            st.error("🚨 Prediction: SPAM")
            st.write("This email has patterns that the model associates with spam.")
        else:
            st.success("✅ Prediction: NOT SPAM")
            st.write("This email looks more like a legitimate message to the model.")
        st.metric("Model confidence", f"{confidence * 100:.1f}%")
        st.caption("Confidence is the model's estimated probability, not a guarantee. Check suspicious links and requests manually.")

with st.expander("Try an example"):
    st.write("**Spam example:**")
    st.code("Congratulations! You won a cash prize. Click here now to claim your reward.")
    st.write("**Not spam example:**")
    st.code("Hi, the project meeting is confirmed for Monday at 10 am. Please bring the report.")

with st.expander("About this project"):
    st.write("The model converts email text into TF-IDF numerical features and uses Logistic Regression to classify it as spam or ham (not spam).")
    st.write(f"Training dataset: {len(pd.read_csv(DATA_PATH))} sample emails.")
    st.warning("This starter dataset is small and educational. It is not suitable for filtering real inboxes without training and testing on a much larger, representative email dataset.")
