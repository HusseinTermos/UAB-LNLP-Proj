import json
import os
from collections import Counter

import numpy as np

import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

from utilities import (
    load_processed_dataset,
    get_label_id,
    LongformerVerificationDataset,
    limit_df)


def add_label_ids(records, label_map):
    fixed_records = []

    for record in records.iloc:
        if record.get("label") is None:
            continue

        record = dict(record)
        record["label_id"] = get_label_id(record["label"], label_map)
        fixed_records.append(record)

    return pd.DataFrame(fixed_records)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    accuracy = float((preds == labels).mean())

    return {
        "accuracy": accuracy
    }

def train_longformer_from_config(cfg):

    splits = load_processed_dataset(cfg["data"]["processed_dataset_path"])

    train_cfg = cfg["training"]
    longformer_cfg = cfg["longformer"]

    label_map = longformer_cfg["label_map"]
    id_to_label = {v: k for k, v in label_map.items()}

    train_records = add_label_ids(splits["train"], label_map)
    val_records = add_label_ids(splits["validation"], label_map)
    test_records = add_label_ids(splits["test"], label_map)

    train_records = limit_df(train_records, cfg["data"]["max_examples_per_split"])
    val_records = limit_df(val_records, cfg["data"]["max_examples_per_split"])
    test_records = limit_df(test_records, cfg["data"]["max_examples_per_split"])
    
    print("Loaded processed records:")
    print(f"  train:      {len(train_records)}")
    print(f"  validation: {len(val_records)}")
    print(f"  test:       {len(test_records)}")

    print("Train label distribution:")
    print(Counter(str(r["label"]).strip().lower() for r in train_records.iloc))

    if len(train_records) == 0:
        raise ValueError("No training records found after adding label IDs.")

    tokenizer = AutoTokenizer.from_pretrained(longformer_cfg["model_name"])

    model = AutoModelForSequenceClassification.from_pretrained(
        longformer_cfg["model_name"],
        num_labels=len(label_map),
        id2label=id_to_label,
        label2id=label_map,
    )

    train_dataset = LongformerVerificationDataset(
        train_records,
        tokenizer,
        max_length=longformer_cfg["max_length"],
        include_labels=True,
    )

    val_dataset = LongformerVerificationDataset(
        val_records,
        tokenizer,
        max_length=longformer_cfg["max_length"],
        include_labels=True,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        per_device_eval_batch_size=train_cfg["batch_size"],
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg.get("save_steps", 500),
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if len(val_records) > 0 else None,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if len(val_records) > 0 else None,
    )

    trainer.train()

    os.makedirs(train_cfg["output_dir"], exist_ok=True)

    trainer.save_model(train_cfg["output_dir"])
    tokenizer.save_pretrained(train_cfg["output_dir"])

    label_map_path = os.path.join(train_cfg["output_dir"], "label_map.json")
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)

    print(f"Saved model to: {train_cfg['output_dir']}")
    print(f"Saved label map to: {label_map_path}")

    return {
        "trainer": trainer,
        "model": model,
        "tokenizer": tokenizer,
        "label_map": label_map,
        "train_records": train_records,
        "val_records": val_records,
        "test_records": test_records,
    }
