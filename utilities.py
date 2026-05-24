import os
import json
from pathlib import Path
import pandas as pd
import torch

from torch.utils.data import Dataset

from query_reformulation import Query_Reformulater
from rag import RAG

#TODO: move below 2 functions to RAG class as static
def build_rag_corpus_from_config(dataframes, cfg):
    evidence_column = "evidence"
    corpus_splits = ["train", "validation", "test"]

    texts = []
    seen = set()

    for split_name in corpus_splits:
        df = dataframes[split_name]

        if evidence_column not in df.columns:
            raise ValueError(
                f"Evidence column '{evidence_column}' not found in {split_name}. "
                f"Available columns: {list(df.columns)}"
            )

        for evidence in df[evidence_column].dropna().astype(str):
            evidence = evidence.strip()

            if not evidence:
                continue

            if evidence in seen:
                continue

            seen.add(evidence)
            texts.append(evidence)

    corpus = "\n\n".join(texts)

    if not corpus.strip():
        raise ValueError("RAG corpus is empty.")

    # logger.info(f"Built RAG corpus from {len(texts)} unique evidence texts.")

    return corpus


def build_rag_from_config(dataframes, cfg):
    corpus = build_rag_corpus_from_config(dataframes, cfg)

    rag_cfg = cfg["rag"]

    rag = RAG(
        document=corpus,
        chunk_size=rag_cfg["chunk_size"],
        chunk_overlap=rag_cfg["chunk_overlap"],
        embedding_model_name=rag_cfg["embedding_model_name"],
        cross_encoder_model_name=rag_cfg["embedding_model_name"],
        reset_collection=rag_cfg["reset_collection"]

    )

    return rag

def build_reformulater_from_config(cfg):
    reform_cfg = cfg["reformulation"]

    if not reform_cfg.get("enabled", True):
        return None

    return Query_Reformulater(
        model_name=reform_cfg["model_name"],
        model_load_mode=reform_cfg["model_load_mode"],
        HF_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        batch_size=reform_cfg["batch_size"],
        temperature=reform_cfg["temperature"],
        max_new_tokens=reform_cfg["max_new_tokens"],
        cache_from=str(Path(reform_cfg["cache_path"])),
        cache_to=str(Path(reform_cfg["cache_path"])),
    )



def load_pubhealth_from_config(cfg, splits=["test", "train", "validation"]):

    if isinstance(splits, str): splits = [splits]

    local_dir = Path(cfg["data"]["local_dir"])
    dataframes = {}
    for split in splits:
        dataframes[split] = pd.read_parquet(local_dir / split / "0000.parquet")

    for split in dataframes:
        dataframes[split] = dataframes[split].rename(
            columns={
                "text_1": "claim",
                "text_2": "evidence",
            }
        )

    return dataframes


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pubhealth(local_dir):
    return {
        "train": pd.read_parquet(os.path.join(local_dir, "train", "0000.parquet")),
        "validation": pd.read_parquet(os.path.join(local_dir, "validation", "0000.parquet")),
        "test": pd.read_parquet(os.path.join(local_dir, "test", "0000.parquet")),
    }

def load_processed_dataset(path):
    rows = []

    if not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    df = pd.DataFrame(rows)

    grouped = {
        label: group_df.reset_index(drop=True)
        for label, group_df in df.groupby("split")
    }

    return grouped


def write_jsonl(rows, path):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def limit_df(df, max_rows):
    if max_rows is None:
        return df

    return df.iloc[:max_rows].copy()


def format_rag_context(rag_results):
    chunks = []

    for i, result in enumerate(rag_results, 1):
        text = result.get("text", "")
        score = result.get("score", None)

        if score is not None:
            chunks.append(f"[{i}] score={score:.4f}\n{text}")
        else:
            chunks.append(f"[{i}]\n{text}")

    return "\n\n".join(chunks)


def build_longformer_input(example):
    return (
        "Query:\n"
        f"{example['reformulated_query']}\n\n"
        "Retrieved evidence:\n"
        f"{example['rag_context']}"
    )


def get_label_id(label_value, label_map):
    if isinstance(label_value, int):
        return label_value

    label_value = str(label_value).strip().lower()

    if label_value not in label_map:
        raise ValueError(
            f"Unknown label: {label_value}. "
            f"Known labels: {list(label_map.keys())}"
        )

    return label_map[label_value]


class LongformerVerificationDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=4096, include_labels=True):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_labels = include_labels

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples.iloc[idx]

        text = build_longformer_input(example)

        item = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
        )

        if self.include_labels:
            item["labels"] = int(example["label_id"])

        return item