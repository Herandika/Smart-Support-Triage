from pathlib import Path
import base64
import io
import re
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import pairwise_distances
from sentence_transformers import SentenceTransformer
from jinja2 import Template

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


# ---------- utils ----------

def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen–Shannon divergence between two probability vectors."""
    m = 0.5 * (p + q)
    mask = (p > 0) & (q > 0)
    if not np.any(mask):
        return 0.0
    kl = lambda a, b: np.sum(a[mask] * np.log(a[mask] / b[mask]))
    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def fig_to_data_uri() -> str:
    """Convert current matplotlib figure to base64 data URI."""
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def tokenize(texts):
    """Very simple tokenizer for vocab drift."""
    return [re.findall(r"[a-z0-9]+", t.lower()) for t in texts]


# ---------- input drift ----------

def make_length_panel(ref_text: pd.Series, cur_text: pd.Series):
    ref_len = ref_text.str.len()
    cur_len = cur_text.str.len()
    all_len = pd.concat([ref_len, cur_len])
    bins = np.linspace(0, np.percentile(all_len, 99), 40)

    plt.figure(figsize=(7, 4))
    plt.hist(ref_len, bins=bins, alpha=0.6, label="ref")
    plt.hist(cur_len, bins=bins, alpha=0.6, label="cur")
    plt.xlabel("length (chars)")
    plt.ylabel("count")
    plt.legend()
    img = fig_to_data_uri()

    def stats(s: pd.Series):
        return {
            "mean": float(s.mean()),
            "median": float(s.median()),
            "p95": float(np.percentile(s, 95)),
        }

    return img, stats(ref_len), stats(cur_len)


def make_vocab_panel(ref_text: pd.Series, cur_text: pd.Series):
    vec = CountVectorizer(
        min_df=3,
        max_features=8000,
        tokenizer=lambda x: x,
        preprocessor=lambda x: x,
    )
    Xr = vec.fit_transform(tokenize(ref_text.tolist()))
    Xc = vec.transform(tokenize(cur_text.tolist()))

    r = (Xr.sum(axis=0) / Xr.sum()).A1
    c = (Xc.sum(axis=0) / Xc.sum()).A1
    vocab = np.array(vec.get_feature_names_out())

    js = js_divergence(r, c)
    delta = c - r
    order = np.argsort(np.abs(delta))[::-1][:20]

    plt.figure(figsize=(8, 4))
    idx = np.arange(len(order))
    plt.bar(idx, delta[order])
    plt.xticks(idx, vocab[order], rotation=45, ha="right")
    plt.ylabel("freq(cur) - freq(ref)")
    img = fig_to_data_uri()

    top_df = pd.DataFrame(
        {
            "token": vocab[order],
            "ref": r[order],
            "cur": c[order],
            "delta": delta[order],
        }
    ).round(6)

    return img, float(js), top_df


# ---------- embedding drift ----------

def make_embedding_panel(ref_text: pd.Series, cur_text: pd.Series):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    Er = model.encode(ref_text.tolist(), normalize_embeddings=True)
    Ec = model.encode(cur_text.tolist(), normalize_embeddings=True)

    mean_cos = 1 - pairwise_distances(
        [Er.mean(axis=0)], [Ec.mean(axis=0)], metric="cosine"
    )[0, 0]

    # Energy distance proxy
    d_rc = np.mean(pairwise_distances(Er, Ec))
    d_rr = np.mean(pairwise_distances(Er, Er))
    d_cc = np.mean(pairwise_distances(Ec, Ec))
    energy = float(d_rc - 0.5 * (d_rr + d_cc))

    return float(mean_cos), energy


# ---------- output drift (intent model) ----------

def label_distribution(labels):
    c = Counter(labels)
    keys = sorted(c.keys())
    probs = np.array([c[k] / sum(c.values()) for k in keys], dtype=float)
    return keys, probs


def make_output_panel(ref_text: pd.Series, cur_text: pd.Series):
    """
    Use intent DistilBERT model to compute:
    - JS divergence between label distributions
    - Delta mean confidence
    """
    model_dir = Path("models/intent_distilbert")
    if not model_dir.exists():
        print("[drift_report] models/intent_distilbert not found -> output drift = None")
        return None, None

    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        truncation=True,
        max_length=128,
        batch_size=16,
    )

    def run(texts):
        preds = clf(texts)
        labels = [p["label"] for p in preds]
        scores = [float(p["score"]) for p in preds]
        return labels, scores

    ref_labels, ref_scores = run(ref_text.tolist())
    cur_labels, cur_scores = run(cur_text.tolist())

    # JS div for label distribution
    kr, pr = label_distribution(ref_labels)
    kc, pc = label_distribution(cur_labels)
    all_labels = sorted(set(kr) | set(kc))

    def align(keys, probs):
        mapping = dict(zip(keys, probs))
        return np.array([mapping.get(k, 0.0) for k in all_labels], dtype=float)

    pr_all = align(kr, pr)
    pc_all = align(kc, pc)
    o_js = js_divergence(pr_all, pc_all)

    conf_shift = float(np.mean(cur_scores) - np.mean(ref_scores))

    return float(o_js), conf_shift


# ---------- main ----------

def main():
    # 1) Load reference & current slices
    ds_ref = load_dataset("clinc_oos", "plus")["train"]
    ds_cur = load_dataset("clinc_oos", "plus")["test"]

    ref = pd.Series(ds_ref["text"]).sample(1000, random_state=1).reset_index(drop=True)
    cur = pd.Series(ds_cur["text"]).sample(1000, random_state=2).reset_index(drop=True)

    # 2) Input drift
    len_img, ref_len_stats, cur_len_stats = make_length_panel(ref, cur)
    vocab_img, js_vocab, top_tokens = make_vocab_panel(ref, cur)

    # 3) Embedding drift
    mean_cos, emb_energy = make_embedding_panel(ref, cur)

    # 4) Output drift (intent model) – may return None if model not found
    o_js, conf_shift = make_output_panel(ref, cur)

    # 5) Render HTML
    template_path = Path("templates/drift_report.html.j2")
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    tmpl = Template(template_path.read_text(encoding="utf-8"))
    html = tmpl.render(
        n_ref=len(ref),
        n_cur=len(cur),
        len_img=len_img,
        tok_img=vocab_img,
        ref_len_stats=ref_len_stats,
        cur_len_stats=cur_len_stats,
        js_vocab=js_vocab,
        top_tokens=top_tokens.to_dict(orient="records"),
        mean_cos=mean_cos,
        emb_energy=emb_energy,
        o_js=o_js,
        conf_shift=conf_shift,
    )

    out = Path("monitoring/reports/text_drift_report.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"[drift_report] Saved: {out}")


if __name__ == "__main__":
    main()