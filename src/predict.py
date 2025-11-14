import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# ==== OPTIONAL: Gemini LLM fallback ====
try:
    import google.generativeai as genai

    _GEMINI_AVAILABLE = True
except Exception:  # library belum terinstall / gagal import
    genai = None
    _GEMINI_AVAILABLE = False

# -----------------------------------------------------------------------------#
# Path & global config
# -----------------------------------------------------------------------------#

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "predictions.jsonl"

# DistilBERT model dirs (hasil training kamu)
INTENT_MODEL_DIR = MODELS_DIR / "intent_distilbert"
SENTIMENT_MODEL_DIR = MODELS_DIR / "sentiment_distilbert"

# Device (GPU kalau ada)
_DEVICE = 0 if torch.cuda.is_available() else -1

# LLM fallback flags (environment-based)
USE_LLM_FALLBACK_ENV = os.getenv("LLM_FALLBACK", "false").lower() == "true"

# Default: Gemini 2.0 Flash (pakai ID penuh versi v1)
# Boleh dioverride: export GEMINI_MODEL="models/gemini-1.5-flash" dll.
_default_gemini_model = "models/gemini-2.0-flash"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", _default_gemini_model)

_GEMINI_MODEL_OBJ: Optional["genai.GenerativeModel"] = None

# -----------------------------------------------------------------------------#
# Lazy loading model pipelines
# -----------------------------------------------------------------------------#

_intent_pipe = None
_sentiment_pipe = None


def _load_intent_pipeline():
    global _intent_pipe
    if _intent_pipe is None:
        tokenizer = AutoTokenizer.from_pretrained(INTENT_MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(INTENT_MODEL_DIR)
        _intent_pipe = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=_DEVICE,
        )
    return _intent_pipe


def _load_sentiment_pipeline():
    global _sentiment_pipe
    if _sentiment_pipe is None:
        tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(
            SENTIMENT_MODEL_DIR
        )
        _sentiment_pipe = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=_DEVICE,
        )
    return _sentiment_pipe


def _load_pipelines():
    return _load_intent_pipeline(), _load_sentiment_pipeline()


# -----------------------------------------------------------------------------#
# Intent mapping & rules
# -----------------------------------------------------------------------------#

# Mapping kasar label mentah → kategori bisnis
INTENT_CATEGORY_MAP: Dict[str, str] = {
    # Card-related
    "card_arrival": "Card",
    "card_linking": "Card",
    "card_not_working": "Card",
    "lost_or_stolen_card": "Card",
    "card_payment_problem": "Card",
    # Billing / charges / refund
    "card_payment_fee_charged": "Billing",
    "extra_charge_on_statement": "Billing",
    "charged_back": "Billing",
    "balance_not_updated_after_bank_transfer": "Billing",
    "getting_spare_card": "Billing",
    "exchange_charge": "Billing",
    # Transfer / payment
    "bank_transfer_failed": "Transfer",
    "bank_transfer_not_received": "Transfer",
    "top_up_by_bank_transfer": "Transfer",
    "cash_withdrawal_not_received": "Transfer",
    # Login / auth
    "passcode_forgotten": "Login",
    "pending_card_payment": "Login",
    "pin_blocked": "Login",
}


def map_intent_category(raw_label: str) -> str:
    """
    Mapping dasar dari label mentah (banking77 / dataset kamu)
    ke kategori bisnis.
    """
    if raw_label in INTENT_CATEGORY_MAP:
        return INTENT_CATEGORY_MAP[raw_label]

    # fallback berdasarkan pattern sederhana
    lower = raw_label.lower()
    if "card" in lower:
        return "Card"
    if "transfer" in lower or "payment" in lower or "withdrawal" in lower:
        return "Transfer"
    if "fee" in lower or "charge" in lower or "refund" in lower or "balance" in lower:
        return "Billing"
    if "passcode" in lower or "pin" in lower or "login" in lower:
        return "Login"
    return "Other"


# Keyword-based rule overrides (bahasa Indonesia)
CARD_KEYWORDS = [
    "kartu",
    "debit",
    "atm",
    "fisik",
    "gesek",
    "chip",
    "magnetic",
]
TRANSFER_KEYWORDS = [
    "transfer",
    "tf ",
    "tf.",
    "tf,",
    "kirim uang",
    "kirim saldo",
    "saldo belum masuk",
    "top up",
    "topup",
    "isi saldo",
    "bayar",
    "pembayaran",
]
BILLING_KEYWORDS = [
    "refund",
    "dikembalikan",
    "uang kembali",
    "double charge",
    "tagihan",
    "penagihan",
    "biaya",
    "fee",
    "potongan",
    "saldo berkurang",
]
LOGIN_KEYWORDS = [
    "login",
    "masuk akun",
    "gak bisa masuk",
    "tidak bisa masuk",
    "password",
    "kata sandi",
    "sandi",
    "pin",
    "otp",
    "kode verifikasi",
    "verifikasi",
]


def apply_rule_overrides(
    text: str, raw_intent: str, mapped_category: str
) -> str:
    """
    Override kategori berdasarkan kata kunci bahasa Indonesia.
    Tujuannya untuk adaptasi ke bahasa user lokal.
    """
    t = text.lower()

    if any(k in t for k in CARD_KEYWORDS):
        return "Card"
    if any(k in t for k in TRANSFER_KEYWORDS):
        return "Transfer"
    if any(k in t for k in BILLING_KEYWORDS):
        return "Billing"
    if any(k in t for k in LOGIN_KEYWORDS):
        return "Login"

    # kalau tidak ada override → pakai mapping awal
    return mapped_category


# -----------------------------------------------------------------------------#
# Priority scoring
# -----------------------------------------------------------------------------#

BASE_PRIORITY = {
    "Card": 3,
    "Transfer": 3,
    "Billing": 2,
    "Login": 2,
    "Other": 1,
}

CRITICAL_KEYWORDS = [
    "hilang",
    "dicuri",
    "penipuan",
    "fraud",
    "disalahgunakan",
    "tidak bisa sama sekali",
    "gak bisa sama sekali",
    "saldo nol",
    "saldo habis",
]


def priority_score(intent_cat: str, sentiment_label: str, text: str) -> int:
    """
    Skor prioritas 1–3.
    Gabungan kategori bisnis, sentiment, dan keyword kritis.
    """
    prio = BASE_PRIORITY.get(intent_cat, 1)

    # sentiment negatif → naikkan
    if sentiment_label.lower().startswith("neg"):
        prio += 1

    # keyword sangat kritikal → naikkan
    t = text.lower()
    if any(k in t for k in CRITICAL_KEYWORDS):
        prio += 1

    return max(1, min(3, prio))


# -----------------------------------------------------------------------------#
# Auto-reply templates (Bahasa Indonesia)
# -----------------------------------------------------------------------------#

def suggest_reply(intent_cat: str, sentiment_label: str) -> str:
    """
    Template balasan dasar per kategori bisnis.
    Ini yang nanti boleh "dipoles" oleh Gemini.
    """
    sent = sentiment_label.lower()

    if intent_cat == "Card":
        return (
            "Terkait kendala kartu, mohon info 4 digit terakhir kartu dan kronologi singkat. "
            "Kami bantu cek dan proses perbaikan secepatnya."
        )

    if intent_cat == "Transfer":
        return (
            "Terkait transfer yang belum masuk, mohon kirimkan tanggal transaksi, nominal, "
            "serta tujuan rekening. Kami bantu cek status transaksi dalam 1x24 jam."
        )

    if intent_cat == "Billing":
        return (
            "Terkait tagihan/refund, mohon kirim nomor transaksi dan bukti pembayaran. "
            "Kami bantu cek detail transaksi Anda dalam 1x24 jam."
        )

    if intent_cat == "Login":
        return (
            "Terkait kendala login, silakan pastikan koneksi internet stabil dan coba kembali. "
            "Jika masih gagal, mohon kirimkan screenshot error untuk kami cek lebih lanjut."
        )

    # Other / fallback
    if sent.startswith("neg"):
        return (
            "Terima kasih atas laporannya, mohon maaf atas ketidaknyamanan yang Anda alami. "
            "Mohon detailkan kendala yang terjadi agar kami dapat membantu secara lebih tepat."
        )

    return (
        "Terima kasih atas pesan Anda. Mohon jelaskan detail kendala atau pertanyaan Anda "
        "agar kami dapat membantu dengan lebih tepat."
    )


# -----------------------------------------------------------------------------#
# Logging
# -----------------------------------------------------------------------------#

def _log_prediction(text: str, result: Dict):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "text": text,
        "intent": result.get("intent"),
        "intent_raw": result.get("intent_raw"),
        "intent_score": result.get("intent_score"),
        "sentiment": result.get("sentiment"),
        "sentiment_score": result.get("sentiment_score"),
        "priority": result.get("priority"),
        "suggested_reply": result.get("suggested_reply"),
        "model_used": result.get("model_used"),
    }
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


# -----------------------------------------------------------------------------#
# Gemini LLM fallback
# -----------------------------------------------------------------------------#

def _get_gemini_model() -> Optional["genai.GenerativeModel"]:
    """
    Inisialisasi lazy Gemini model.
    Return None kalau:
    - library belum terinstall
    - tidak ada GEMINI_API_KEY
    - terjadi error saat konfigurasi
    """
    global _GEMINI_MODEL_OBJ

    if not _GEMINI_AVAILABLE:
        return None

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    # Normalisasi nama model: pastikan pakai prefix "models/"
    model_name = (GEMINI_MODEL or "").strip() or _default_gemini_model
    if not model_name.startswith("models/"):
        model_name = "models/" + model_name

    if _GEMINI_MODEL_OBJ is None:
        try:
            genai.configure(api_key=api_key)
            _GEMINI_MODEL_OBJ = genai.GenerativeModel(model_name)
        except Exception:
            return None

    return _GEMINI_MODEL_OBJ


def generate_llm_reply_gemini(
    user_text: str,
    intent_cat: str,
    sentiment: str,
    base_reply: str,
) -> Optional[str]:
    """
    Gunakan Gemini untuk merapikan / memperkaya balasan.
    Tidak mengubah kebijakan bisnis (base_reply tetap jadi sumber kebenaran).
    Kalau gagal → return None (tidak memaksa).
    """
    model = _get_gemini_model()
    if model is None:
        return None

    prompt = f"""
Kamu adalah customer service bank digital di Indonesia.
Balas pesan nasabah dengan bahasa Indonesia yang sopan, singkat, dan jelas.
Selalu profesional dan empatik. Jangan mengubah kebijakan atau membuat janji baru.

Pesan nasabah:
\"\"\"{user_text}\"\"\"

Kategori intent (hasil model): {intent_cat}
Sentiment (hasil model): {sentiment}

Balasan rule-based yang sudah ada:
\"\"\"{base_reply}\"\"\"

Tolong buat balasan final dalam bahasa Indonesia berdasarkan balasan rule-based di atas.
- Pertahankan inti informasi & langkah yang sama.
- Perbaiki struktur kalimat dan tambah empati seperlunya.
- Maksimal 3 kalimat.
- Jangan menambahkan informasi atau promo baru.
"""

    try:
        resp = model.generate_content(prompt)
        text = resp.text.strip() if hasattr(resp, "text") else ""
        return text or None
    except Exception:
        return None


# -----------------------------------------------------------------------------#
# Public API
# -----------------------------------------------------------------------------#

def predict_one(
    text: str,
    log: bool = True,
    use_llm_fallback: Optional[bool] = None,
) -> Dict:
    """
    Prediksi satu pesan pelanggan.

    Flow:
    - DistilBERT intent & sentiment
    - Mapping + rules → intent_cat
    - Priority + base_reply
    - Opsional: Gemini fallback untuk merapikan balasan
      (aktif bila:
        - use_llm_fallback=True, atau
        - LLM_FALLBACK=true di environment)
    """
    if not isinstance(text, str) or not text.strip():
        result = {
            "intent_raw": None,
            "intent": "Other",
            "sentiment": "neu",
            "priority": 1,
            "suggested_reply": (
                "Tolong kirimkan detail pesan atau pertanyaan Anda agar kami dapat membantu."
            ),
            "model_used": {
                "intent": None,
                "sentiment": None,
                "llm_fallback": None,
            },
            "models_dir": str(MODELS_DIR),
            "intent_score": 0.0,
            "sentiment_score": 0.0,
        }
        if log:
            _log_prediction(text or "", result)
        return result

    intent_pipe, sent_pipe = _load_pipelines()

    # --- model predictions ---
    intent_out = intent_pipe(text)[0]  # {'label': ..., 'score': ...}
    sent_out = sent_pipe(text)[0]

    raw_intent = intent_out["label"]
    intent_score = float(intent_out["score"])

    sentiment_label = sent_out["label"]
    sentiment_score = float(sent_out["score"])

    # --- mapping & rules ---
    cat0 = map_intent_category(raw_intent)
    intent_cat = apply_rule_overrides(text, raw_intent, cat0)

    # --- priority & base reply ---
    prio = priority_score(intent_cat, sentiment_label, text)
    base_reply = suggest_reply(intent_cat, sentiment_label)

    # --- LLM fallback (Gemini) ---
    if use_llm_fallback is None:
        use_llm_fallback = USE_LLM_FALLBACK_ENV

    llm_used = False
    final_reply = base_reply

    low_confidence = intent_score < 0.7
    if use_llm_fallback and (intent_cat == "Other" or low_confidence):
        llm_reply = generate_llm_reply_gemini(
            user_text=text,
            intent_cat=intent_cat,
            sentiment=sentiment_label,
            base_reply=base_reply,
        )
        if llm_reply:
            final_reply = llm_reply
            llm_used = True

    result = {
        "intent_raw": raw_intent,
        "intent": intent_cat,
        "sentiment": sentiment_label,
        "priority": prio,
        "suggested_reply": final_reply,
        "model_used": {
            "intent": "distilbert",
            "sentiment": "distilbert",
            "llm_fallback": "gemini" if llm_used else None,
        },
        "models_dir": str(MODELS_DIR),
        "intent_score": intent_score,
        "sentiment_score": sentiment_score,
    }

    if log:
        _log_prediction(text, result)

    return result


if __name__ == "__main__":
    # quick manual test
    example = (
        "tolong refund transaksi saya kemarin, double charge dan saldo saya "
        "berkurang dua kali"
    )
    print(predict_one(example, log=False, use_llm_fallback=True))
