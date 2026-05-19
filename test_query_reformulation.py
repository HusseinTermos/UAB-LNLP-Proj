from query_reformulation import Query_Reformulater
import os

import pandas as pd
import logging
import warnings
from datasets import load_dataset, load_from_disk

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CACHE_PATH  = "/kaggle/working/reformulation_cache.json"
from query_reformulation_config import MODEL_NAME, CACHE_PATH, BATCH_SIZE, MAX_NEW_TOK, TEMPERATURE, LOCAL_PUBHEALTH_DIR, SPLITS, PROMPT_TEMPLATE

def load_pubhealth():
    dataframes = {
        "train": pd.read_parquet(LOCAL_PUBHEALTH_DIR / "train" / "0000.parquet"),
        "validation": pd.read_parquet(LOCAL_PUBHEALTH_DIR / "validation" / "0000.parquet"),
        "test": pd.read_parquet(LOCAL_PUBHEALTH_DIR / "test" / "0000.parquet"),
    }

    for split in dataframes:
        dataframes[split] = dataframes[split].rename(
            columns={"text_1": "claim", "text_2": "evidence"}
        )

    return dataframes

def main():

    # --- load data from GitHub ---
    dataframes = load_pubhealth()

    train_claims = dataframes["train"]["claim"].tolist()
    val_claims   = dataframes["validation"]["claim"].tolist()
    test_claims  = dataframes["test"]["claim"].tolist()

    logger.info(
        f"Claims — train: {len(train_claims)} | "
        f"val: {len(val_claims)} | "
        f"test: {len(test_claims)}"
    )
    reformulater = Query_Reformulater(HF_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
                                      cache_from=CACHE_PATH,
                                      cache_to=CACHE_PATH)
    # --- reformulate all splits ---
    all_results = {}
    for split_name, claims in [
        ("train",      train_claims[:3]),
        ("validation", val_claims[:3]),
        ("test",       test_claims[:3]),
    ]:
        logger.info(f"\n=== Reformulating {split_name} split ===")
        all_results[split_name] = reformulater.reformulate_batch(
            claims
        )

    logger.info(f"Cache saved to: {CACHE_PATH}")

    # --- inspect a few examples ---
    print("\n" + "=" * 60)
    print("SAMPLE REFORMULATIONS")
    print("=" * 60)

    for claim in train_claims[:3]:
        key    = Query_Reformulater.claim_hash(claim)
        result = reformulater.cache.get(key, {})
        print(f"\nOriginal claim:\n  {claim}")
        print(f"\nReformulated: {result}")

        print("-" * 60)

if __name__ == "__main__":
    main()

