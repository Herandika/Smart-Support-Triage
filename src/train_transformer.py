# src/train_transformer.py  (Fixed label handling for CLINC OOS + tweet_eval)
import argparse, numpy as np
from pathlib import Path
from datasets import load_dataset, ClassLabel
import evaluate
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, TrainingArguments, Trainer
)

def get_task_config(task: str):
    if task == "intent":
        ds = load_dataset("clinc_oos", "plus")   # works on datasets 3.x
        text_col = "text"
        label_col = "intent"
        # label names: if ClassLabel, pull .names; else unique strings
        if isinstance(ds["train"].features[label_col], ClassLabel):
            label_names = ds["train"].features[label_col].names
        else:
            label_names = sorted(list(set(ds["train"][label_col])))
        out_dir = Path("models/intent_distilbert")
    elif task == "sentiment":
        ds = load_dataset("tweet_eval", "sentiment")
        text_col = "text"; label_col = "label"
        label_names = ["neg","neu","pos"]  # 0,1,2
        out_dir = Path("models/sentiment_distilbert")
    else:
        raise ValueError("task must be 'intent' or 'sentiment'")
    out_dir.mkdir(parents=True, exist_ok=True)
    return ds, label_names, text_col, label_col, out_dir

def main(args):
    ds, label_names, text_col, label_col, out_dir = get_task_config(args.task)

    # subset cepat kalau CPU
    if args.small:
        for split in list(ds.keys()):
            ds[split] = ds[split].shuffle(seed=42).select(range(min(len(ds[split]), args.small)))
        print(f"[{args.task}] small sizes:", {k: len(v) for k,v in ds.items()})

    tok = AutoTokenizer.from_pretrained(args.model_name)
    def tokenize(batch):
        return tok(batch[text_col], truncation=True)
        return tokzr(batch[text_col], truncation=True, max_length=args.max_len)

    # tokenisasi; buang kolom selain text/label
    keep = [text_col, label_col]
    rm_cols = [c for c in ds["train"].column_names if c not in keep]
    tokenized = ds.map(tokenize, batched=True, remove_columns=rm_cols)

    # --- LABEL HANDLING (FIX) ---
    # Jika label sudah ClassLabel → sudah integer (0..N-1)
    # Jika label masih string → map ke integer berdasarkan label_names
    if not isinstance(tokenized["train"].features[label_col], ClassLabel):
        name2id = {n:i for i,n in enumerate(label_names)}
        def to_int(example):
            return {label_col: name2id[example[label_col]]}
        tokenized = tokenized.map(to_int)
    if label_col != "labels":
        tokenized = tokenized.rename_column(label_col, "labels")

    data_collator = DataCollatorWithPadding(tokenizer=tok)
    f1_metric = evaluate.load("f1")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"f1": f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]}

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_names),
        id2label={i:n for i,n in enumerate(label_names)},
        label2id={n:i for i,n in enumerate(label_names)},
    )

    training_args = TrainingArguments(
        output_dir=str(out_dir/"hf_outputs"),
        learning_rate=args.lr,
        gradient_accumulation_steps=args.grad_accum,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=not args.cpu,
        report_to="none",
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"] if "test" in tokenized else tokenized["validation"],
        tokenizer=tok,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    (out_dir/"labels.txt").write_text("\n".join(label_names), encoding="utf-8")
    print(f"[{args.task}] saved to {out_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["intent","sentiment"], required=True)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--model_name", default="distilbert-base-uncased")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--small", type=int, default=0)
    args = p.parse_args()
    main(args)
