# ============================================================
# Query Reformulation for PUBHEALTH Claims
# Local LLM (Mistral 7B Instruct)

import os
import json
import re
import hashlib
import warnings
import logging

import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME  = "mistralai/Mistral-7B-Instruct-v0.3"
CACHE_PATH  = "/kaggle/working/reformulation_cache.json"
BATCH_SIZE  = 4
MAX_NEW_TOK = 500
TEMPERATURE = 0.1

DATA_DIR = "/kaggle/input/pubhealthdataset"
SPLITS = {
    "train":      "train.tsv",
    "validation": "dev.tsv",
    "test":       "test.tsv",
}


# ============================================================
# PROMPT
# ============================================================

PROMPT_TEMPLATE = """<s>[INST] You are a biomedical information retrieval specialist.
Reformulate the health claim below into structured PubMed search queries.

Rules:
- normalized_claim: rewrite using precise clinical/scientific terminology.
  Remove source attributions ("scientists say", "studies show") and
  certainty framing ("definitely", "proven", "always"). Keep only the
  core factual assertion.
- pico.population: who the claim is about (e.g. "adults", "children",
  "patients with hypertension"). Use "general population" if unspecified.
- pico.intervention: the substance, action, or exposure being claimed
  (use chemical/clinical names where possible).
- pico.outcome: the claimed effect or result (use clinical terminology).
- queries: exactly 4 strings optimized for PubMed:
    1. Targets RCTs or meta-analyses (highest evidence tier)
    2. Targets the specific biological mechanism claimed
    3. Targets the population or a safety/risk study
    4. Broad query using BOTH lay terms and clinical equivalents
- bm25_keywords: space-separated list of the most important clinical
  and scientific terms (include both lay term and clinical equivalent
  where they differ, e.g. "heart attack myocardial infarction").

Respond with valid JSON only. No explanation. No preamble. No markdown.

CLAIM: {claim} [/INST]"""


# ============================================================
# DATA LOADING
# ============================================================

def load_pubhealth() -> dict:
    """
    Load PUBHEALTH train / validation / test splits from the local
    Kaggle dataset (/kaggle/input/pubhealthdataset/).
    Returns a dict of {split_name: pd.DataFrame}.
    """
    dataframes = {}

    for split_name, filename in SPLITS.items():
        path = os.path.join(DATA_DIR, filename)
        logger.info(f"Loading {split_name} from {path}...")
        df = pd.read_csv(path, sep="\t", on_bad_lines="skip")

        before = len(df)
        df = df.dropna(subset=["claim"])
        df = df[df["claim"].str.strip() != ""]
        after = len(df)

        if before != after:
            logger.warning(
                f"{split_name}: dropped {before - after} rows with missing claims"
            )

        logger.info(f"{split_name}: {len(df)} claims | columns: {list(df.columns)}")
        dataframes[split_name] = df

    return dataframes


# ============================================================
# MODEL LOADER
# ============================================================

def load_model(model_name: str = MODEL_NAME):
    """
    Load Mistral 7B in 4-bit NF4 quantization.
    Fits on a single Kaggle T4 (16GB) using ~4-5GB VRAM.
    """
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",  # required for batch generation
    )
    tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model in 4-bit (NF4)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()

    logger.info("Model loaded successfully.")
    return tokenizer, model


# ============================================================
# CACHE UTILITIES
# ============================================================

def load_cache(cache_path: str = CACHE_PATH) -> dict:
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict, cache_path: str = CACHE_PATH) -> None:
    dirpath = os.path.dirname(cache_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)


def claim_hash(claim: str) -> str:
    return hashlib.sha256(claim.strip().lower().encode()).hexdigest()


# ============================================================
# PARSING AND VALIDATION
# ============================================================

def parse_response(raw_text: str, claim: str) -> dict:
    """
    Attempt to extract valid JSON from the model output.
    Falls back to a minimal structure using the raw claim
    if parsing fails.
    """
    # direct parse
    try:
        return json.loads(raw_text.strip())
    except json.JSONDecodeError:
        pass

    # extract JSON block if model added surrounding text
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # fallback
    logger.warning(f"JSON parse failed. Using fallback for: {claim[:60]}")
    return {
        "normalized_claim": claim,
        "pico": {
            "population":   "general population",
            "intervention": claim,
            "outcome":      "",
        },
        "queries":       [claim],
        "bm25_keywords": " ".join(claim.split()),
    }


def validate(result: dict) -> bool:
    required = ["normalized_claim", "pico", "queries", "bm25_keywords"]
    if not all(k in result for k in required):
        return False
    if not isinstance(result["queries"], list) or len(result["queries"]) == 0:
        return False
    pico_keys = ["population", "intervention", "outcome"]
    if not all(k in result.get("pico", {}) for k in pico_keys):
        return False
    return True


# ============================================================
# BATCH REFORMULATION
# ============================================================

def reformulate_batch(
    claims: list,
    tokenizer,
    model,
    cache: dict,
    batch_size: int = BATCH_SIZE,
    cache_path: str = CACHE_PATH,
) -> list:
    """
    Reformulate a list of claims in batches.
    Skips claims already in cache.
    Saves cache to disk after every batch.
    """
    results  = [None] * len(claims)
    uncached = []  # (original_index, claim)

    for i, claim in enumerate(claims):
        key = claim_hash(claim)
        if key in cache:
            results[i] = cache[key]
        else:
            uncached.append((i, claim))

    logger.info(
        f"Total: {len(claims)} | "
        f"Cached: {len(claims) - len(uncached)} | "
        f"To process: {len(uncached)}"
    )

    for batch_start in range(0, len(uncached), batch_size):
        batch         = uncached[batch_start : batch_start + batch_size]
        indices, batch_claims = zip(*batch)

        prompts = [PROMPT_TEMPLATE.format(claim=c) for c in batch_claims]

        inputs = tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=768,
        ).to(next(model.parameters()).device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOK,
                temperature=TEMPERATURE,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]

        for j, (orig_idx, claim) in enumerate(zip(indices, batch_claims)):
            new_tokens = output_ids[j][input_len:]
            raw = tokenizer.decode(
                new_tokens, skip_special_tokens=True
            ).strip()

            result = parse_response(raw, claim)
            if not validate(result):
                logger.warning(f"Validation failed, using fallback for: {claim[:60]}")
                result = {
                    "normalized_claim": claim,
                    "pico": {
                        "population":   "general population",
                        "intervention": claim,
                        "outcome":      "",
                    },
                    "queries":       [claim],
                    "bm25_keywords": " ".join(claim.split()),
                }

            key               = claim_hash(claim)
            cache[key]        = result
            results[orig_idx] = result

        save_cache(cache, cache_path)
        processed = min(batch_start + batch_size, len(uncached))
        logger.info(
            f"Processed {processed}/{len(uncached)} uncached claims"
        )

    return results


# ============================================================
# MAIN
# ============================================================

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

    # --- load model ---
    tokenizer, model = load_model(MODEL_NAME)

    # --- load cache ---
    cache = load_cache(CACHE_PATH)
    logger.info(f"Cache loaded: {len(cache)} existing entries")

    # --- reformulate all splits ---
    all_results = {}
    for split_name, claims in [
        ("train",      train_claims),
        ("validation", val_claims),
        ("test",       test_claims),
    ]:
        logger.info(f"\n=== Reformulating {split_name} split ===")
        all_results[split_name] = reformulate_batch(
            claims,
            tokenizer,
            model,
            cache,
            batch_size=BATCH_SIZE,
            cache_path=CACHE_PATH,
        )

    logger.info(f"\nDone. Total cached entries: {len(cache)}")
    logger.info(f"Cache saved to: {CACHE_PATH}")

    # --- inspect a few examples ---
    print("\n" + "=" * 60)
    print("SAMPLE REFORMULATIONS")
    print("=" * 60)

    for claim in train_claims[:3]:
        key    = claim_hash(claim)
        result = cache.get(key, {})
        print(f"\nOriginal claim:\n  {claim}")
        print(f"Normalized:\n  {result.get('normalized_claim', 'N/A')}")
        pico = result.get("pico", {})
        print(f"PICO:")
        print(f"  Population:   {pico.get('population',   'N/A')}")
        print(f"  Intervention: {pico.get('intervention', 'N/A')}")
        print(f"  Outcome:      {pico.get('outcome',      'N/A')}")
        print(f"Queries:")
        for i, q in enumerate(result.get("queries", []), 1):
            print(f"  {i}. {q}")
        print(f"BM25 Keywords:\n  {result.get('bm25_keywords', 'N/A')}")
        print("-" * 60)


if __name__ == "__main__":
    main()
