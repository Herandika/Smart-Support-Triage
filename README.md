# Smart Support Triage & Auto-Reply (DistilBERT)

End-to-end AI untuk triage tiket + auto-reply:
- **Intent classification** (CLINC OOS, DistilBERT)
- **Sentiment analysis** (TweetEval, DistilBERT)
- **Streamlit UI** + `predict.py` API

## Hasil (v1)
| Model | Macro-F1 | Accuracy |
|------|----------|----------|
| Intent (CLINC OOS) | 0.9026902616578022 | 0.8714545454545455 |
| Sentiment (TweetEval) | 0.6979219533340298 | 0.7027841094106154|

## Cara jalan
```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
# atau:
python -c "from src.predict import predict_one; print(predict_one('tolong refund transaksi saya'))"
