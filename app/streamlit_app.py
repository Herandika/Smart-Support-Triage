import json
from pathlib import Path
from datetime import datetime
import sys

import pandas as pd
import streamlit as st

# === Pastikan root project ada di sys.path ===
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import predict_one  # sekarang baru aman impor ini

LOG_PATH = PROJECT_ROOT / "logs" / "predictions.jsonl"

st.set_page_config(
    page_title="Smart Support Triage & Auto-Reply",
    page_icon="💬",
    layout="wide",
)


# ---------- Helpers ----------

def load_logs(limit: int = 50) -> pd.DataFrame:
    if not LOG_PATH.exists():
        return pd.DataFrame()
    rows = []
    with LOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp", ascending=False)
    return df.head(limit)


def kpi(label: str, value, help_text: str | None = None):
    # small helper for KPI tiles
    st.markdown(
        f"""
        <div style="padding:10px 14px;border-radius:12px;border:1px solid #eee;margin-bottom:8px;">
          <div style="font-size:11px;color:#777;text-transform:uppercase;letter-spacing:0.06em;">{label}</div>
          <div style="font-size:22px;font-weight:700;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if help_text:
        st.caption(help_text)


# ---------- Sidebar ----------

with st.sidebar:
    st.markdown("### 🔧 Pengaturan & Contoh Input")

    examples = {
        "Kartu belum datang": "kartu debit saya belum datang, padahal sudah daftar dari minggu lalu",
        "Masalah refund": "tolong refund transaksi saya kemarin, double charge dan saldo saya berkurang dua kali",
        "Gagal login": "aku dari tadi gak bisa login ke akun, selalu muncul error password salah padahal bener",
        "Transfer belum masuk": "saya sudah transfer ke teman tapi uangnya belum masuk juga sampai sekarang",
        "Feedback positif": "terima kasih ya, tadi CS nya sangat membantu masalah tagihan saya sampai beres",
    }

    example_key = st.selectbox(
        "Pilih contoh pesan (opsional):",
        options=["(kosong)"] + list(examples.keys()),
        index=0,
    )

    st.markdown("---")
    st.caption(
        "Setiap prediksi akan otomatis dicatat ke `logs/predictions.jsonl` "
        "dan bisa dianalisis di drift report."
    )


# ---------- Main layout ----------

st.markdown("## 💬 Smart Support Triage & Auto-Reply")
st.markdown(
    "Demo mini-project"
    " penentuan prioritas tiket, dan auto-reply Bahasa Indonesia."
)

col_input, col_meta = st.columns([1.6, 1.1])

with col_input:
    # initial text from example if chosen
    default_text = ""
    if example_key != "(kosong)":
        default_text = examples[example_key]

    user_text = st.text_area(
        "Masukkan pesan customer:",
        value=default_text,
        height=150,
        placeholder="Contoh: kartu saya belum datang, kapan ya?",
    )

    run_btn = st.button("🚀 Jalankan Prediksi", use_container_width=True)

    if run_btn:
        if not user_text.strip():
            st.warning("Tolong isi pesan dulu ya 🤏")
        else:
            with st.spinner("Menganalisis intent & sentiment..."):
                try:
                    result = predict_one(user_text)
                except Exception as e:
                    st.error(f"Gagal menjalankan prediksi: {e}")
                    result = None

            if result is not None:
                st.markdown("### ✅ Hasil Prediksi & Auto-Reply")

                # Suggested reply
                st.markdown("**Balasan yang disarankan:**")
                st.success(result.get("suggested_reply", "(tidak ada balasan)"))

                st.markdown("#### Detail Prediksi")
                col_top1, col_top2, col_top3 = st.columns(3)
                with col_top1:
                    kpi(
                        "Kategori intent",
                        result.get("intent", "-"),
                        help_text=f"Label mentah model: {result.get('intent_raw', '-')}",
                    )
                with col_top2:
                    kpi(
                        "Sentiment",
                        result.get("sentiment", "-"),
                        help_text="Berdasarkan DistilBERT sentiment.",
                    )
                with col_top3:
                    kpi(
                        "Prioritas tiket",
                        result.get("priority", "-"),
                        help_text="1 = rendah, 2 = sedang, 3 = tinggi.",
                    )

                with st.expander("🔍 Confidence & model info"):
                    st.write(
                        {
                            "intent_score": result.get("intent_score"),
                            "sentiment_score": result.get("sentiment_score"),
                            "model_used": result.get("model_used"),
                            "models_dir": result.get("models_dir"),
                        }
                    )

                with st.expander("📦 Payload komplet (JSON)"):
                    st.json(result)

    else:
        st.info(
            "Isi pesan pelanggan di atas atau pilih salah satu contoh di sidebar, "
            "lalu klik **🚀 Jalankan Prediksi**."
        )

with col_meta:
    st.markdown("### 📊 Aktivitas Terakhir (Logs)")

    df_logs = load_logs(limit=20)
    if df_logs.empty:
        st.caption("Belum ada logs. Jalankan beberapa prediksi dulu.")
    else:
        # KPI dari logs
        last_ts = df_logs["timestamp"].max()
        n_today = (df_logs["timestamp"].dt.date == datetime.utcnow().date()).sum()

        kpi("Total sample (tercatat)", len(df_logs), "Terbatas ke ~20 log terakhir.")
        kpi("Prediksi hari ini", int(n_today), "Berdasarkan timestamp UTC.")
        kpi(
            "Intent terbanyak (recent)",
            df_logs["intent"].value_counts().idxmax()
            if "intent" in df_logs.columns
            else "-",
        )

        with st.expander("📄 Tabel log terbaru"):
            # tampilkan subset kolom penting
            cols = [
                c
                for c in [
                    "timestamp",
                    "text",
                    "intent",
                    "sentiment",
                    "priority",
                    "intent_score",
                    "sentiment_score",
                ]
                if c in df_logs.columns
            ]
            st.dataframe(df_logs[cols], use_container_width=True, height=300)

        st.caption(
            "Log ini bisa dipakai sebagai *current data* untuk drift analysis via "
            "`tools/drift_report.py`."
        )

st.markdown("---")
st.caption(
    "Smart Support Triage & Auto-Reply · DistilBERT + rules Bahasa Indonesia · "
    "Logging & drift monitoring siap pakai."
)
