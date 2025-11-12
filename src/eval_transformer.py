from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
import numpy as np, evaluate, json
from pathlib import Path

def eval_model(model_dir, dataset, text_col, label_col, label_names):
    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_dir)
    def tokf(b): return tok(b[text_col], truncation=True)
    rm = [c for c in dataset["train"].column_names if c not in [text_col, label_col]]
    ds_tok = dataset.map(tokf, batched=True, remove_columns=rm)
    if not isinstance(ds_tok["train"].features[label_col], ClassLabel):
        name2id = {n:i for i,n in enumerate(label_names)}
        ds_tok = ds_tok.map(lambda x: {"labels": name2id[x[label_col]]})
    else:
        ds_tok = ds_tok.rename_column(label_col, "labels")

    f1 = evaluate.load("f1"); acc = evaluate.load("accuracy")
    def metrics(p):
        preds = np.argmax(p.predictions, axis=-1)
        return {"macro_f1": f1.compute(predictions=preds, references=p.label_ids, average="macro")["f1"],
                "accuracy": acc.compute(predictions=preds, references=p.label_ids)["accuracy"]}
    tr = Trainer(model=mdl, tokenizer=tok, compute_metrics=metrics)
    res = tr.evaluate(ds_tok["test"] if "test" in ds_tok else ds_tok["validation"])
    return {k: float(v) for k,v in res.items() if k in ["eval_macro_f1","eval_accuracy","macro_f1","accuracy"]}

# Intent (CLINC OOS)
cl = load_dataset("clinc_oos", "plus")
intent_labels = cl["train"].features["intent"].names if isinstance(cl["train"].features["intent"], ClassLabel) else sorted(set(cl["train"]["intent"]))
intent_res = eval_model("models/intent_distilbert", cl, "text", "intent", intent_labels)

# Sentiment (tweet_eval)
tw = load_dataset("tweet_eval", "sentiment")
sent_res = eval_model("models/sentiment_distilbert", tw, "text", "label", ["neg","neu","pos"])

Path("metrics").mkdir(exist_ok=True)
Path("metrics/results.json").write_text(json.dumps({"intent": intent_res, "sentiment": sent_res}, indent=2))
print(open("metrics/results.json").read())
