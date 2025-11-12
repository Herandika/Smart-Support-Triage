import streamlit as st
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predict import predict_one

st.set_page_config(page_title="Smart Support Triage", layout="centered")
st.title("🧠 Smart Support Triage & Auto-Reply")

text = st.text_area("Masukkan pesan pelanggan:", placeholder="Contoh: Saya tidak bisa login sejak semalam.")
if st.button("Prediksi"):
    if text.strip():
        out = predict_one(text)
        st.json(out)
        st.write("**Suggested Reply**")
        st.code(out["suggested_reply"])
