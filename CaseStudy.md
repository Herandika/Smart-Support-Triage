\# Smart Support Triage \& Auto-Reply — 1 Page



\*\*Masalah\*\*  

Tim support menerima ratusan tiket/day. Perlu triase cepat \& konsisten.



\*\*Solusi\*\*  

Model intent + sentiment → priority score (1–3) + suggested reply. API FastAPI + UI Streamlit. Siap dihubungkan ke n8n/RPA.



\*\*Hasil (Baseline)\*\*

\- Intent Macro-F1 ≈ 0.89

\- Sentiment Macro-F1 ≈ 0.56

\- Simulasi: auto-reply < 1 menit; eskalasi otomatis untuk prioritas 3.



\*\*Arsitektur\*\*  

Data → TF-IDF model → API `/predict` → UI → n8n (Webhook → HTTP Request → Google Sheet/Email).



\*\*Nilai Bisnis\*\*  

\- Turunkan SLA respon awal.

\- Konsistensi jawaban tanpa beban agent.

\- Logging rapi untuk analitik.



\*\*Next\*\*  

Fine-tune DistilBERT, monitoring drift, feedback loop dari agent.



