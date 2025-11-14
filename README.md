# 💬 Smart Support Triage & Auto-Reply

Mini-project untuk mensimulasikan **AI Support Triage** di perusahaan fintech/bank digital:

- **DistilBERT** untuk klasifikasi **intent** & **sentiment** pesan pelanggan  
- **Business rules + mapping** untuk mengubah label mentah jadi kategori bisnis (Login / Transfer / Billing / Card / Other)  
- **Auto-reply engine** dalam Bahasa Indonesia  
- **Prioritas tiket** (1–3) berdasarkan intent, sentiment, dan kata kunci kritikal  
- **Logging & drift monitoring** (input + embedding + output drift)  
- **LLM fallback dengan **Gemini** untuk merapikan balasan

Dibangun sebagai proyek mandiri.

---

## 🚀 Fitur Utama

- 🔍 **Intent & Sentiment Classification**
  - Fine-tuning DistilBERT untuk intent & sentiment
  - Output: label mentah + skor confidence

- 🧠 **Business Logic Layer**
  - Mapping label ke kategori bisnis:
    - `Login`, `Transfer`, `Billing`, `Card`, `Other`
  - Rule-based override memakai kata kunci Bahasa Indonesia
  - Priority score (1–3) berdasarkan intent, sentiment, dan keyword

- 💬 **Auto-Reply Engine**
  - Template balasan per kategori bisnis
  - Menyesuaikan tone berdasarkan sentiment
  - Opsional: **fallback ke Gemini** untuk merapikan balasan, tanpa mengubah kebijakan bisnis

- 📊 **Monitoring & Drift**
  - Logging setiap prediksi ke `logs/predictions.jsonl` (format JSON Lines)
  - Skrip `tools/drift_report.py`:
    - Text length distribution
    - Vocab shift
    - Embedding drift (cosine similarity)
    - Output drift (perubahan distribusi intent)

- 🖥 **Streamlit Dashboard**
  - Input pesan pelanggan
  - Tampilkan intent, sentiment, priority, auto-reply
  - Panel log recent predictions (tabel + stats)
  - Cocok untuk demo ke HR / stakeholder non-teknis

---

## 🧱 Arsitektur

```text
smart-support-triage/
  app/
    streamlit_app.py       # UI untuk CS / demo
  src/
    __init__.py
    predict.py             # core inference, business logic, logging, Gemini fallback
    train_baseline.py      # (opsional) baseline model
    train_transformer.py   # fine-tuning DistilBERT
  models/
    intent_distilbert/     # model intent hasil training
    sentiment_distilbert/  # model sentiment hasil training
  tools/
    drift_report.py        # generate monitoring/reports/text_drift_report.html
  monitoring/
    reports/
      text_drift_report.html
  logs/
    predictions.jsonl      # log real traffic (tidak di-commit)
  requirements.txt
  README.md
