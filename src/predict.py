# src/predict.py
from pathlib import Path
import joblib

# ===== Path aman (absolute) =====
# <repo_root>/models/...
REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = (REPO_ROOT / "models").resolve()

HF_INTENT_DIR = MODELS_DIR / "intent_distilbert"
HF_SENT_DIR   = MODELS_DIR / "sentiment_distilbert"

# ===== Coba load DistilBERT dulu =====
use_hf_intent = HF_INTENT_DIR.exists()
use_hf_sent   = HF_SENT_DIR.exists()

hf_intent = hf_sent = None
hf_intent_tok = hf_sent_tok = None
hf_intent_labels = hf_sent_labels = None

if use_hf_intent or use_hf_sent:
    # import hanya jika butuh
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

if use_hf_intent:
    hf_intent_tok = AutoTokenizer.from_pretrained(str(HF_INTENT_DIR))
    hf_intent = AutoModelForSequenceClassification.from_pretrained(str(HF_INTENT_DIR))
    hf_intent_labels = (HF_INTENT_DIR / "labels.txt").read_text(encoding="utf-8").splitlines()

if use_hf_sent:
    hf_sent_tok = AutoTokenizer.from_pretrained(str(HF_SENT_DIR))
    hf_sent = AutoModelForSequenceClassification.from_pretrained(str(HF_SENT_DIR))
    hf_sent_labels = (HF_SENT_DIR / "labels.txt").read_text(encoding="utf-8").splitlines()

# ===== Fallback: TF-IDF =====
intent_model_tfidf = intent_le = sent_model_tfidf = sent_le = None
if not use_hf_intent:
    intent_model_tfidf = joblib.load(MODELS_DIR / "intent_tfidf.joblib")
    intent_le          = joblib.load(MODELS_DIR / "intent_label_encoder.joblib")
if not use_hf_sent:
    sent_model_tfidf = joblib.load(MODELS_DIR / "sentiment_tfidf.joblib")
    sent_le          = joblib.load(MODELS_DIR / "sentiment_label_encoder.joblib")

# ===== Mapping label -> kategori bisnis =====
def _wild(pat: str, text: str) -> bool:
    pat = pat.lower(); text = text.lower()
    return text.startswith(pat[:-1]) if pat.endswith("*") else (pat == text)

# jika kamu masih pakai Banking77 bisa aktifkan ALIAS berbasis prefix
ALIAS = {
    "passcode_forgotten": "Login",
    "card_*":             "Card",
    "chargeback_*":       "Billing",
    "refund_*":           "Billing",
    "transfer_*":         "Transfer",
}
PRIORITY_INTENTS = {"Login", "Transfer", "Billing", "Card"}

def map_intent_category(raw_intent_name: str) -> str:
    """Gabungan: keyword CLINC + fallback pattern Banking77"""
    name = raw_intent_name.lower()

    # Keyword CLINC OOS (lebih umum)
    if any(k in name for k in ["card", "debit", "credit", "atm"]): return "Card"
    if any(k in name for k in ["refund", "charge", "chargeback", "fee", "billing", "payment_issue"]): return "Billing"
    if any(k in name for k in ["transfer", "send_money", "remittance", "wire"]): return "Transfer"
    if any(k in name for k in ["login", "password", "passcode", "signin", "reset"]): return "Login"

    # Fallback Banking77
    for pat, cat in ALIAS.items():
        if _wild(pat, name): return cat
    return "Other"

def apply_rule_overrides(text: str, raw_intent: str, mapped_cat: str) -> str:
    """Perbaiki misklasifikasi baseline dengan keyword dari teks pelanggan."""
    t = text.lower()
    if any(k in t for k in ["kartu", "card", "atm", "debit", "kredit", "credit"]):
        if any(p in t for p in ["belum datang","belum terima","belum sampai","pengiriman","kapan datang","kartu belum","card not arrived","card arrival","card delivery"]):
            return "Card"
    if any(k in t for k in ["transfer","tf","kirim uang","kirim saldo"]):
        if any(p in t for p in ["gagal","error","pending","tertahan","belum masuk","tidak masuk"]):
            return "Transfer"
    if any(k in t for k in ["refund","tagihan","biaya","fee","charge","chargeback","double charge"]):
        return "Billing"
    if any(k in t for k in ["login","masuk","password","kata sandi","passcode","lupa sandi","reset password"]):
        return "Login"
    return mapped_cat

def priority_score(intent_cat: str, sentiment: str, text: str) -> int:
    p = 1
    if sentiment == "neg": p += 1
    if any(k in text.lower() for k in ["refund","tidak bisa","gagal","error","terblokir","uang hilang","tidak dapat login","double charge"]):
        p += 1
    if intent_cat in PRIORITY_INTENTS: p += 1
    return min(p, 3)

def suggest_reply(intent_cat: str, sentiment: str) -> str:
    T = {
        "Login":   "Silakan coba reset password melalui menu 'Lupa Password'. Bila masih gagal, balas dengan detail waktu & pesan error.",
        "Billing": "Terkait tagihan/refund, mohon kirim nomor transaksi & bukti pembayaran. Kami bantu cek dalam 1x24 jam.",
        "Transfer":"Mohon info nominal & waktu transfer. Kami periksa statusnya.",
        "Card":    "Mohon info 4 digit terakhir kartu & kronologi singkat. Kami bantu verifikasi.",
        "Other":   "Terima kasih atas laporan Anda, sedang kami proses."
    }
    msg = T.get(intent_cat, T["Other"])
    if sentiment == "neg": msg += " Mohon maaf atas ketidaknyamanannya."
    return msg

# ===== Inference helpers =====
def _decode_sklearn(le, idx: int) -> str:
    return le.inverse_transform([idx])[0]

def _hf_predict(model, tok, labels, text):
    import torch
    with torch.no_grad():
        inputs = tok(text, return_tensors="pt", truncation=True)
        logits = model(**inputs).logits
        pred = int(logits.argmax(-1).item())
        return labels[pred]

# ===== Public API =====
def predict_one(text: str):
    # intent raw
    if use_hf_intent:
        raw_intent = _hf_predict(hf_intent, hf_intent_tok, hf_intent_labels, text)
        intent_backend = "distilbert"
    else:
        idx = intent_model_tfidf.predict([text])[0]
        raw_intent = _decode_sklearn(intent_le, idx)
        intent_backend = "tfidf"

    # sentiment raw
    if use_hf_sent:
        sentiment = _hf_predict(hf_sent, hf_sent_tok, hf_sent_labels, text)
        sent_backend = "distilbert"
    else:
        sidx = sent_model_tfidf.predict([text])[0]
        sentiment = _decode_sklearn(sent_le, sidx)
        sent_backend = "tfidf"

    # mapping + overrides
    intent_cat = map_intent_category(raw_intent)
    intent_cat = apply_rule_overrides(text, raw_intent, intent_cat)

    prio  = priority_score(intent_cat, sentiment, text)
    reply = suggest_reply(intent_cat, sentiment)

    return {
        "intent_raw": raw_intent,
        "intent": intent_cat,
        "sentiment": sentiment,
        "priority": prio,
        "suggested_reply": reply,
        "model_used": {"intent": intent_backend, "sentiment": sent_backend},
        "models_dir": str(MODELS_DIR)  # debug visibility
    }
