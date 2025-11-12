\# Smart Support Triage \& Auto-Reply



\*\*Singkat:\*\* Klasifikasi intent + sentiment untuk triase tiket, hitung prioritas, dan hasilkan \*suggested reply\*. Ada \*\*API (FastAPI)\*\* dan \*\*UI demo (Streamlit)\*\*, siap diintegrasikan ke \*\*n8n/RPA\*\*.



\## Nilai Bisnis

\- SLA respon awal < 1 menit via auto-reply.

\- Eskalasi otomatis untuk kasus prioritas tinggi.

\- Konsistensi jawaban \& logging terstruktur.



\## Arsitektur

Data → Baseline (TF-IDF + LinearSVC / LogReg) → API FastAPI → Streamlit → (opsional) n8n/RPA → Monitoring (Evidently).



\## Hasil (Baseline)

\- \*\*Intent (banking77)\*\*: Macro-F1 ≈ \*\*0.89\*\*

\- \*\*Sentiment (tweet\_eval)\*\*: Macro-F1 ≈ \*\*0.56\*\*

> Catatan: angka baseline; bisa ditingkatkan dengan DistilBERT.



\## Cara Jalankan

```bash

\# 1) aktifkan venv, lalu:

uvicorn api.app:app --reload --port 8000      # API di :8000

\# tab lain:

streamlit run app/streamlit\_app.py            # UI di :8501

\# uji API:

curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\\"text\\":\\"tidak bisa login\\"}"



