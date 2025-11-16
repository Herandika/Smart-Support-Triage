# ---- Base image ----
FROM python:3.11-slim

# Non-interactive & no cache
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ---- Workdir ----
WORKDIR /app

# ---- System deps (kalau butuh) ----
# (Bisa ditambah kalau nanti ada error lib, tapi mulai dari minimal dulu)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---- Install Python deps ----
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# ---- Copy project files ----
COPY . .

# ---- Default env for LLM fallback ----
# GEMINI_API_KEY akan di-set via "Secrets" di Hugging Face Spaces,
# jadi di sini kita cukup set flag & model name.
ENV LLM_FALLBACK=true
ENV GEMINI_MODEL=models/gemini-2.0-flash

# ---- Expose port (HF default 7860) ----
EXPOSE 7860

# ---- Run Streamlit app ----
# Kalau nama file beda, sesuaikan jalurnya
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0"]
