# ============================================================
# Longformer Classifier for PUBHEALTH
# ============================================================

import os
import json
import hashlib
import warnings
import logging
import argparse
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import (
    LongformerTokenizerFast,
    LongformerModel,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import f1_score, classification_report

warnings.filterwarnings("ignore")


# ============================================================
# ARGS — override any default via command line
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Longformer PUBHEALTH classifier"
    )
    parser.add_argument(
        "--output_dir",     type=str, default="./outputs",
        help="Directory for checkpoints and logs"
    )
    parser.add_argument(
        "--reformulation_cache", type=str,
        default="./reformulation_cache.json",
        help="Path to query reformulation cache JSON"
    )
    parser.add_argument(
        "--rag_path",       type=str, default=None,
        help="Path to RAG passages JSON (optional)"
    )
    parser.add_argument(
        "--model_name",     type=str,
        default="allenai/longformer-base-4096"
    )
    parser.add_argument("--max_length",     type=int,   default=2048)
    parser.add_argument("--batch_size",     type=int,   default=4,
        help="Per-GPU batch size"
    )
    parser.add_argument("--epochs",         type=int,   default=4)
    parser.add_argument("--lr",             type=float, default=2e-5)
    parser.add_argument("--dropout",        type=float, default=0.1)
    parser.add_argument("--warmup_ratio",   type=float, default=0.1)
    parser.add_argument("--max_grad_norm",  type=float, default=1.0)
    parser.add_argument("--num_workers",    type=int,   default=4,
        help="DataLoader worker processes per GPU"
    )
    parser.add_argument("--seed",           type=int,   default=42)
    return parser.parse_args()


# ============================================================
# DISTRIBUTED SETUP
# ============================================================

def setup_distributed():
    """
    Initialise the process group when launched with torchrun.
    Falls back to single-GPU if not in a distributed context.
    """
    if "LOCAL_RANK" in os.environ:
        local_rank  = int(os.environ["LOCAL_RANK"])
        world_size  = int(os.environ.get("WORLD_SIZE", 1))
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return local_rank, world_size
    else:
        return 0, 1     # single GPU fallback


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(local_rank: int) -> bool:
    return local_rank == 0


def setup_logger(local_rank: int) -> logging.Logger:
    """Only rank-0 logs to stdout to avoid duplicate output."""
    logger = logging.getLogger("pubhealth")
    level  = logging.INFO if is_main_process(local_rank) else logging.WARNING
    logging.basicConfig(
        level=level,
        format="[%(asctime)s][rank-{:d}] %(message)s".format(local_rank),
    )
    return logger


# ============================================================
# CONSTANTS
# ============================================================

LABEL_MAP   = {"true": 0, "false": 1, "mixture": 2, "unproven": 3}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
NUM_CLASSES = 4

GITHUB_BASE = (
    "https://raw.githubusercontent.com/"
    "neemakot/Health-Fact-Checking/master/data/PUBHEALTH/"
)
SPLITS = {
    "train":      GITHUB_BASE + "train.tsv",
    "validation": GITHUB_BASE + "dev.tsv",
    "test":       GITHUB_BASE + "test.tsv",
}


# ============================================================
# !! RAG PLACEHOLDER !!
# Replace this function with your friend's RAG output.
#
# Expected format of the JSON file:
#   {
#     "<md5_hash_of_claim>": [
#         "passage text 1",
#         "passage text 2",
#         "passage text 3"
#     ],
#     ...
#   }
#
# Hash is computed as:
#   hashlib.md5(claim.strip().lower().encode()).hexdigest()
#
# Until the RAG file is available the model runs on
# claim + normalized claim only (valid degraded baseline).
# ============================================================

def load_rag_passages(path: Optional[str], logger) -> Dict[str, List[str]]:
    if path and os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
        logger.info(f"RAG passages loaded: {len(data)} entries from {path}")
        return data

    logger.warning(
        "RAG passages not found — running on claim + "
        "normalized claim only. Swap in real RAG output via --rag_path."
    )
    return {}


# ============================================================
# DATA LOADING
# ============================================================

def load_pubhealth(logger) -> dict:
    dataframes = {}
    for split, url in SPLITS.items():
        logger.info(f"Loading {split} from GitHub...")
        df = pd.read_csv(url, sep="\t", on_bad_lines="skip")
        df = df.dropna(subset=["claim"])
        df = df[df["claim"].str.strip() != ""]
        df = df[df["label"].isin(LABEL_MAP.keys())]
        df["label_id"] = df["label"].map(LABEL_MAP)
        logger.info(f"  {split}: {len(df)} rows")
        dataframes[split] = df
    return dataframes


def claim_hash(claim: str) -> str:
    return hashlib.md5(claim.strip().lower().encode()).hexdigest()


def load_reformulation_cache(path: str, logger) -> dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            cache = json.load(f)
        logger.info(f"Reformulation cache: {len(cache)} entries from {path}")
        return cache
    logger.warning(f"Reformulation cache not found at {path}. Using raw claims.")
    return {}


# ============================================================
# INPUT CONSTRUCTION
# ============================================================

def build_evidence_text(
    claim: str,
    reformulation: Optional[dict],
    passages: List[str],
) -> str:
    """
    Build the evidence side of the input pair.

    Structure (text_b passed to tokenizer):
      normalized_claim </s> passage_1 </s> passage_2 </s> passage_3

    </s> is Longformer's separator token.
    If no reformulation is available the raw claim is used instead.
    """
    normalized = (
        reformulation.get("normalized_claim", claim)
        if reformulation else claim
    )

    parts = [normalized]
    for p in passages[:3]:
        if p and p.strip():
            parts.append(p.strip())

    return " </s> ".join(parts)


# ============================================================
# DATASET
# ============================================================

class PubHealthDataset(Dataset):

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        reformulation_cache: dict,
        rag_passages: dict,
        max_length: int,
    ):
        self.df                  = df.reset_index(drop=True)
        self.tokenizer           = tokenizer
        self.reformulation_cache = reformulation_cache
        self.rag_passages        = rag_passages
        self.max_length          = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        claim = str(row["claim"])
        label = int(row["label_id"])

        key           = claim_hash(claim)
        reformulation = self.reformulation_cache.get(key)
        passages      = self.rag_passages.get(key, [])
        evidence_text = build_evidence_text(claim, reformulation, passages)

        encoding = self.tokenizer(
            claim,
            evidence_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,        # truncates evidence side first
            return_tensors="pt",
            return_attention_mask=True,
        )

        input_ids      = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # global attention:
        #   position 0  ([CLS]) always global
        #   claim tokens global so all evidence tokens attend to them
        global_attention_mask = torch.zeros_like(input_ids)
        global_attention_mask[0] = 1

        claim_tokens = self.tokenizer.tokenize(claim)
        claim_end    = min(len(claim_tokens) + 1, self.max_length - 1)
        global_attention_mask[1:claim_end] = 1

        return {
            "input_ids":             input_ids,
            "attention_mask":        attention_mask,
            "global_attention_mask": global_attention_mask,
            "label":                 torch.tensor(label, dtype=torch.long),
        }


# ============================================================
# MODEL
# ============================================================

class LongformerClassifier(nn.Module):

    def __init__(
        self,
        model_name:  str   = "allenai/longformer-base-4096",
        num_classes: int   = NUM_CLASSES,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.longformer = LongformerModel.from_pretrained(model_name)
        hidden          = self.longformer.config.hidden_size   # 768

        self.classifier = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_ids, attention_mask, global_attention_mask):
        outputs = self.longformer(
            input_ids             = input_ids,
            attention_mask        = attention_mask,
            global_attention_mask = global_attention_mask,
        )
        cls_vector = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_vector)


# ============================================================
# CLASS WEIGHTS
# ============================================================

def compute_class_weights(df: pd.DataFrame, device) -> torch.Tensor:
    counts  = df["label_id"].value_counts().sort_index()
    total   = len(df)
    weights = [
        total / (NUM_CLASSES * counts.get(i, 1))
        for i in range(NUM_CLASSES)
    ]
    return torch.tensor(weights, dtype=torch.float).to(device)


# ============================================================
# DATALOADER FACTORY
# ============================================================

def make_loader(
    dataset,
    batch_size:  int,
    shuffle:     bool,
    num_workers: int,
    distributed: bool,
    local_rank:  int,
) -> DataLoader:
    sampler = (
        DistributedSampler(dataset, shuffle=shuffle)
        if distributed else None
    )
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = (shuffle and sampler is None),
        sampler     = sampler,
        num_workers = num_workers,
        pin_memory  = True,         # faster CPU→GPU transfers on cluster
        persistent_workers = (num_workers > 0),
    )


# ============================================================
# TRAINING
# ============================================================

def train_epoch(
    model, loader, optimizer, scheduler,
    criterion, device, logger, epoch,
):
    model.train()
    total_loss = 0.0

    # required for DistributedSampler to shuffle differently each epoch
    if hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)

    for step, batch in enumerate(loader):
        input_ids             = batch["input_ids"].to(device, non_blocking=True)
        attention_mask        = batch["attention_mask"].to(device, non_blocking=True)
        global_attention_mask = batch["global_attention_mask"].to(device, non_blocking=True)
        labels                = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, global_attention_mask)
        loss   = criterion(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters() if not isinstance(model, DDP)
            else model.module.parameters(),
            max_norm=1.0,
        )

        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

        if (step + 1) % 50 == 0:
            logger.info(
                f"  step {step + 1}/{len(loader)} "
                f"| loss: {loss.item():.4f}"
            )

    return total_loss / len(loader)


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids             = batch["input_ids"].to(device, non_blocking=True)
            attention_mask        = batch["attention_mask"].to(device, non_blocking=True)
            global_attention_mask = batch["global_attention_mask"].to(device, non_blocking=True)
            labels                = batch["label"].to(device, non_blocking=True)

            logits = model(input_ids, attention_mask, global_attention_mask)
            loss   = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    # gather predictions across all GPUs in DDP
    if dist.is_initialized():
        world_size = dist.get_world_size()
        gathered_preds  = [None] * world_size
        gathered_labels = [None] * world_size
        dist.all_gather_object(gathered_preds,  all_preds)
        dist.all_gather_object(gathered_labels, all_labels)
        all_preds  = [p for sublist in gathered_preds  for p in sublist]
        all_labels = [l for sublist in gathered_labels for l in sublist]

    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    report   = classification_report(
        all_labels, all_preds,
        target_names=list(LABEL_MAP.keys()),
        digits=4,
    )

    return {
        "loss":     total_loss / len(loader),
        "macro_f1": macro_f1,
        "report":   report,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    # --- distributed init ---
    local_rank, world_size = setup_distributed()
    distributed = world_size > 1
    device      = torch.device(f"cuda:{local_rank}"
                               if torch.cuda.is_available()
                               else "cpu")

    logger = setup_logger(local_rank)
    logger.info(f"World size: {world_size} | Local rank: {local_rank} | Device: {device}")

    # --- reproducibility ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- output directory (rank-0 only) ---
    if is_main_process(local_rank):
        os.makedirs(args.output_dir, exist_ok=True)

    # --- data (all ranks load independently — no shared memory issues) ---
    dataframes          = load_pubhealth(logger)
    reformulation_cache = load_reformulation_cache(args.reformulation_cache, logger)
    rag_passages        = load_rag_passages(args.rag_path, logger)

    # --- tokenizer ---
    tokenizer = LongformerTokenizerFast.from_pretrained(args.model_name)

    # --- datasets ---
    train_dataset = PubHealthDataset(
        dataframes["train"], tokenizer,
        reformulation_cache, rag_passages, args.max_length,
    )
    val_dataset = PubHealthDataset(
        dataframes["validation"], tokenizer,
        reformulation_cache, rag_passages, args.max_length,
    )
    test_dataset = PubHealthDataset(
        dataframes["test"], tokenizer,
        reformulation_cache, rag_passages, args.max_length,
    )

    # --- dataloaders ---
    train_loader = make_loader(
        train_dataset, args.batch_size, shuffle=True,
        num_workers=args.num_workers, distributed=distributed,
        local_rank=local_rank,
    )
    val_loader = make_loader(
        val_dataset, args.batch_size, shuffle=False,
        num_workers=args.num_workers, distributed=distributed,
        local_rank=local_rank,
    )
    test_loader = make_loader(
        test_dataset, args.batch_size, shuffle=False,
        num_workers=args.num_workers, distributed=distributed,
        local_rank=local_rank,
    )

    # --- model ---
    model = LongformerClassifier(
        model_name  = args.model_name,
        num_classes = NUM_CLASSES,
        dropout     = args.dropout,
    ).to(device)

    if distributed:
        model = DDP(
            model,
            device_ids        = [local_rank],
            output_device     = local_rank,
            find_unused_parameters = False,
        )

    # --- loss ---
    weights   = compute_class_weights(dataframes["train"], device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # --- optimizer & scheduler ---
    # effective batch size scales with world_size
    effective_batch = args.batch_size * world_size
    logger.info(f"Effective batch size: {effective_batch}")

    optimizer    = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps  = warmup_steps,
        num_training_steps= total_steps,
    )

    # --- training loop ---
    best_macro_f1   = 0.0
    best_model_path = os.path.join(args.output_dir, "best_longformer.pt")

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n=== Epoch {epoch}/{args.epochs} ===")

        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, device, logger, epoch,
        )

        # evaluate on rank-0 only to avoid duplicate metrics
        if is_main_process(local_rank):
            val_metrics = evaluate(model, val_loader, criterion, device)
            logger.info(f"Train loss   : {train_loss:.4f}")
            logger.info(f"Val loss     : {val_metrics['loss']:.4f}")
            logger.info(f"Val Macro F1 : {val_metrics['macro_f1']:.4f}")
            print(val_metrics["report"])

            if val_metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = val_metrics["macro_f1"]
                # save underlying module weights (not DDP wrapper)
                state = (
                    model.module.state_dict()
                    if isinstance(model, DDP)
                    else model.state_dict()
                )
                torch.save(state, best_model_path)
                logger.info(
                    f"  ✓ Best model saved "
                    f"(Macro F1: {best_macro_f1:.4f}) → {best_model_path}"
                )

        # barrier: all ranks wait before next epoch
        if distributed:
            dist.barrier()

    # --- test evaluation (rank-0 only) ---
    if is_main_process(local_rank):
        logger.info("\n=== Test Evaluation (best checkpoint) ===")
        best_state = torch.load(best_model_path, map_location=device)
        if isinstance(model, DDP):
            model.module.load_state_dict(best_state)
        else:
            model.load_state_dict(best_state)

        test_metrics = evaluate(model, test_loader, criterion, device)
        logger.info(f"Test Macro F1: {test_metrics['macro_f1']:.4f}")
        print(test_metrics["report"])

        # save final report
        report_path = os.path.join(args.output_dir, "test_report.txt")
        with open(report_path, "w") as f:
            f.write(f"Test Macro F1: {test_metrics['macro_f1']:.4f}\n\n")
            f.write(test_metrics["report"])
        logger.info(f"Report saved to {report_path}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
