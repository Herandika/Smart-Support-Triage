

---



\# 3) MODEL\_CARD.md (copy ke file MODEL\_CARD.md)

```markdown

\# Model Card — Smart Support Triage



\## Intended Use

\- Routing \& triase tiket: intent, sentiment, priority, suggested reply.



\## Data

\- Intent: PolyAI/banking77 (proxy domain).

\- Sentiment: tweet\_eval/sentiment.



\## Training

\- Baseline: TF-IDF + LinearSVC (intent), TF-IDF + LogisticRegression (sentiment).



\## Metrics

\- Intent Macro-F1 ≈ 0.89

\- Sentiment Macro-F1 ≈ 0.56



\## Limitations

\- Dataset umum (belum domain-adapted).

\- Bahasa: campuran; perlu fine-tune untuk korpus Indonesia internal.



\## Risks

\- Mis-classification pada kalimat ambigu.

\- Auto-reply harus ditinjau untuk kasus prioritas tinggi.



\## Recommendations

\- Tambah human-in-the-loop.

\- Few-shot domain adaptation + continual learning.



