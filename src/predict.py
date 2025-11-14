"""
Smart Support Triage & Auto-Reply - Prediction Helper

Fitur:
- Load DistilBERT intent & sentiment dari ./models
- Mapping label raw -> kategori bisnis (Login / Transfer / Billing / Card / Other)
- Rule-based override pakai kata kunci teks (Indonesia + English)
- Hitung priority (1..3) dan suggested_reply (Bahasa Indonesia)
- Logging real prediction ke logs/predictions.jsonl (JSON Lines)
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# ========= PATH & GLOBALS =========

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

HF_INTENT_DIR = MODELS_DIR / "intent_distilbert"
HF_SENTIMENT_DIR = MODELS_DIR / "sentiment_distilbert"

LOG_PATH = PROJECT_ROOT / "logs" / "predictions.jsonl"

_INTENT_PIPE: Optional[object] = None
_SENTIMENT_PIPE: Optional[object] = None

# ========= MAPPING & RULES =========

# wildcard helper
def _wild(pat: str, text: str) -> bool:
    pat = pat.lower()
    text = text.lower()
    if pat.endswith("*"):
        return text.startswith(pat[:-1])
    return pat == text


# alias untuk beberapa label (CLINC / Banking77 style)
ALIAS = {
    "passcode_forgotten": "Login",
    "card_*": "Card",
    "chargeback_*": "Billing",
    "refund_*": "Billing",
    "transfer_*": "Transfer",
}

PRIORITY_INTENTS = {"Login", "Transfer", "Billing", "Card"}


def map_intent_category(raw_intent_name: str) -> str:
    """
    Map nama intent mentah (label model HF) -> kategori bisnis utama.
    Contoh kategori: Card, Billing, Transfer, Login, Other
    """
    name = raw_intent_name.lower()

    # CLINC-style keywords (lebih umum & fleksibel)
    if any(k in name for k in ["card", "debit", "credit", "atm"]):
        return "Card"
    if any(
        k in name
        for k in [
            "refund",
            "charge",
            "chargeback",
            "fee",
            "billing",
            "payment_issue",
            "payment",
        ]
    ):
        return "Billing"
    if any(k in name for k in ["transfer", "send_money", "remittance", "wire"]):
        return "Transfer"
    if any(k in name for k in ["login", "password", "passcode", "signin", "reset"]):
        return "Login"

    # Fallback alias (Banking77-like)
    for pat, cat in ALIAS.items():
        if _wild(pat, name):
            return cat

    return "Other"


def apply_rule_overrides(text: str, raw_intent: str, mapped_cat: str) -> str:
    """
    Perbaiki misklasifikasi dengan keyword dari teks user (Indonesia + Inggris).
    Contoh: teks mengandung 'kartu' & 'belum datang' -> paksa kategori Card.
    """
    t = text.lower()

    # Kasus kartu fisik
    if any(k in t for k in ["kartu", "card", "atm", "debit", "kredit", "credit"]):
        if any(
            p in t
            for p in [
                "belum datang",
                "belum terima",
                "belum sampai",
                "pengiriman",
                "kapan datang",
                "kartu belum",
                "card not arrived",
                "card arrival",
                "card delivery",
            ]
        ):
            return "Card"

    # Transfer error
    if any(k in t for k in ["transfer", "tf", "kirim uang", "kirim saldo", "send money"]):
        if any(p in t for p in ["gagal", "error", "pending", "tertahan", "belum masuk", "tidak masuk"]):
            return "Transfer"

    # Refund / billing
    if any(
        k in t
        for k in [
            "refund",
            "tagihan",
            "biaya",
            "fee",
            "charge",
            "chargeback",
            "double charge",
            "uang kembali",
        ]
    ):
        return "Billing"

    # Login / akses akun
    if any(
        k in t
        for k in [
            "login",
            "masuk",
            "akun",
            "password",
            "kata sandi",
            "passcode",
            "lupa sandi",
            "reset password",
            "gak bisa masuk",
            "tidak bisa login",
        ]
    ):
        return "Login"

    return mapped_cat


def priority_score(intent_cat: str, sentiment: str, text: str) -> int:
    """
    Skor prioritas 1..3 berdasarkan kategori intent, sentiment, dan kata kunci.
    1 = low, 2 = medium, 3 = high
    """
    p = 1
    t = text.lower()

    if sentiment.lower() in ["neg", "negative", "label_0"]:
        p += 1

    if any(
        k in t
        for k in [
            "refund",
            "tidak bisa",
            "gagal",
            "error",
            "terblokir",
            "uang hilang",
            "uang saya hilang",
            "tidak dapat login",
            "tidak bisa login",
            "double charge",
            "saldo berkurang",
        ]
    ):
        p += 1

    if intent_cat in PRIORITY_INTENTS:
        p += 1

    return min(p, 3)


def suggest_reply(intent_cat: str, sentiment: str) -> str:
    """
    Template balasan berdasarkan kategori intent & sentiment.
    """
    base = {
        "Login": (
            "Silakan coba reset password melalui menu 'Lupa Password'. "
            "Jika masih gagal, kirimkan tangkapan layar pesan error dan email/nomor HP yang terdaftar."
        ),
        "Billing": (
            "Terkait tagihan/refund, mohon kirim nomor transaksi & bukti pembayaran. "
            "Kami bantu cek dalam 1x24 jam hari kerja."
        ),
        "Transfer": (
            "Mohon info nominal, waktu, dan tujuan transfer. "
            "Jika saldo sudah berkurang namun belum masuk, kami akan bantu cek status transaksinya."
        ),
        "Card": (
            "Mohon info 4 digit terakhir kartu & kronologi singkat (misalnya kartu belum datang, rusak, atau terblokir). "
            "Kami bantu cek status kartu Anda."
        ),
        "Other": (
            "Terima kasih atas laporan Anda, saat ini kendala sedang kami proses. "
            "Jika ada detail tambahan, silakan kirimkan agar kami bisa membantu lebih tepat."
        ),
    }

    msg = base.get(intent_cat, base["Other"])

    # tone adjustment berdasarkan sentiment
    s = sentiment.lower()
    if s in ["neg", "negative", "label_0"]:
        msg += " Kami mohon maaf atas ketidaknyamanan yang Anda alami."
    elif s in ["pos", "positive", "label_2"]:
        msg += " Senang bisa membantu Anda 😊."

    return msg


# ========= INTERNAL HELPERS (MODEL & LOGGING) =========

def _get_device() -> int:
    return 0 if torch.cuda.is_available() else -1


def _load_pipelines():
    """
    Lazily load intent & sentiment pipelines dari folder models/.
    """
    global _INTENT_PIPE, _SENTIMENT_PIPE

    if _INTENT_PIPE is None:
        if not HF_INTENT_DIR.exists():
            raise FileNotFoundError(
                f"Intent model not found at {HF_INTENT_DIR}. "
                "Pastikan kamu sudah menjalankan training DistilBERT intent."
            )
        tok_i = AutoTokenizer.from_pretrained(HF_INTENT_DIR)
        mdl_i = AutoModelForSequenceClassification.from_pretrained(HF_INTENT_DIR)
        _INTENT_PIPE = pipeline(
            "text-classification",
            model=mdl_i,
            tokenizer=tok_i,
            device=_get_device(),
            truncation=True,
            max_length=128,
            batch_size=16,
        )

    if _SENTIMENT_PIPE is None:
        if not HF_SENTIMENT_DIR.exists():
            raise FileNotFoundError(
                f"Sentiment model not found at {HF_SENTIMENT_DIR}. "
                "Pastikan kamu sudah menjalankan training DistilBERT sentiment."
            )
        tok_s = AutoTokenizer.from_pretrained(HF_SENTIMENT_DIR)
        mdl_s = AutoModelForSequenceClassification.from_pretrained(HF_SENTIMENT_DIR)
        _SENTIMENT_PIPE = pipeline(
            "text-classification",
            model=mdl_s,
            tokenizer=tok_s,
            device=_get_device(),
            truncation=True,
            max_length=128,
            batch_size=16,
        )

    return _INTENT_PIPE, _SENTIMENT_PIPE


def _log_prediction(input_text: str, result: dict) -> None:
    """
    Append satu prediksi ke logs/predictions.jsonl.
    """
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "text": input_text,
        "intent_raw": result.get("intent_raw"),
        "intent": result.get("intent"),
        "sentiment": result.get("sentiment"),
        "priority": result.get("priority"),
        "suggested_reply": result.get("suggested_reply"),
        "model_used": result.get("model_used"),
        "models_dir": result.get("models_dir"),
        "intent_score": result.get("intent_score"),
        "sentiment_score": result.get("sentiment_score"),
    }

    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


# ========= PUBLIC API =========

def predict_one(text: str, log: bool = True) -> dict:
    """
    Prediksi satu pesan pelanggan.

    Flow:
    - DistilBERT intent → raw label (ex: "card_arrival", "refund_not_showing_up", dst.)
    - DistilBERT sentiment → raw label (ex: "negative"/"neutral"/"positive" atau LABEL_x)
    - Map raw intent ke kategori (Card/Billing/Transfer/Login/Other)
    - Apply rule override berbasis teks
    - Hitung priority & suggested_reply
    - Optional: logging ke logs/predictions.jsonl

    Return dict:
      {
        "intent_raw": "...",
        "intent": "Card" | "Billing" | ...,
        "sentiment": "...",
        "priority": 1..3,
        "suggested_reply": "...",
        "model_used": {"intent": "distilbert", "sentiment": "distilbert"},
        "models_dir": "...",
        "intent_score": float,
        "sentiment_score": float,
      }
    """
    intent_pipe, sent_pipe = _load_pipelines()

    intent_out = intent_pipe(text)[0]   # {'label': ..., 'score': ...}
    sent_out = sent_pipe(text)[0]

    raw_intent = intent_out["label"]
    intent_score = float(intent_out["score"])

    sentiment_label = sent_out["label"]
    sentiment_score = float(sent_out["score"])

    # map ke kategori
    intent_cat = map_intent_category(raw_intent)
    intent_cat = apply_rule_overrides(text, raw_intent, intent_cat)

    # hitung priority & reply
    prio = priority_score(intent_cat, sentiment_label, text)
    reply = suggest_reply(intent_cat, sentiment_label)

    result = {
        "intent_raw": raw_intent,
        "intent": intent_cat,
        "sentiment": sentiment_label,
        "priority": prio,
        "suggested_reply": reply,
        "model_used": {"intent": "distilbert", "sentiment": "distilbert"},
        "models_dir": str(MODELS_DIR),
        "intent_score": intent_score,
        "sentiment_score": sentiment_score,
    }

    if log:
        _log_prediction(text, result)

    return result


# ========= CLI TEST =========

if __name__ == "__main__":
    sample = "kartu saya belum datang, kapan ya?"
    out = predict_one(sample, log=False)
    print(out)
