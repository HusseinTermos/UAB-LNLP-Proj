import argparse
import json
import logging
from pathlib import Path

from utilities import build_rag_from_config, load_config, limit_df, load_pubhealth_from_config, build_reformulater_from_config


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parent


def format_rag_context(rag_results):
    chunks = []

    for i, result in enumerate(rag_results, start=1):
        text = str(result.get("text", "")).strip()
        score = result.get("score", None)

        if not text:
            continue

        if score is not None:
            chunks.append(f"[DOC {i} | score={float(score):.4f}]\n{text}")
        else:
            chunks.append(f"[DOC {i}]\n{text}")

    return "\n\n".join(chunks)



def build_dataset_from_config(cfg):
    dataframes = load_pubhealth_from_config(cfg)

    rag = build_rag_from_config(dataframes, cfg)
    reformulater = build_reformulater_from_config(cfg)

    data_cfg = cfg["data"]
    rag_cfg = cfg["rag"]
    reform_cfg = cfg["reformulation"]

    out_path = Path(data_cfg["processed_dataset_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    splits = ["train", "validation", "test"]
    max_examples_per_split = data_cfg["max_examples_per_split"]

    top_k = rag_cfg["top_k"]
    rag_method = rag_cfg["method"]
    query_source = rag_cfg["query_source"]

    with open(out_path, "w", encoding="utf-8") as f:
        for split_name in splits:
            df = dataframes[split_name]

            if max_examples_per_split is not None:
                df = limit_df(df, max_examples_per_split)

            records = df.to_dict("records")
            claims = [str(record["claim"]) for record in records]

            if reform_cfg.get("enabled", True):
                logger.info(f"Reformulating {len(claims)} claims from {split_name}...")
                reformulated_claims = reformulater.reformulate_batch(claims)
            else:
                reformulated_claims = claims

            logger.info(f"Running RAG for {split_name}...")

            for idx, record in enumerate(records):
                original_claim = str(record["claim"])
                reformulated_query = str(reformulated_claims[idx])

                if query_source == "original":
                    rag_query = original_claim
                elif query_source == "reformulated":
                    rag_query = reformulated_query
                else:
                    raise ValueError("rag.query_source must be 'original' or 'reformulated'.")

                rag_results = rag.query(
                    query=rag_query,
                    top_k=top_k,
                    method=rag_method,
                )

                rag_context = format_rag_context(rag_results)

                output_record = {
                    "split": split_name,
                    "example_id": record.get("id", f"{split_name}_{idx}"),
                    "claim": original_claim,
                    "reformulated_query": reformulated_query,
                    "rag_query": rag_query,
                    "rag_context": rag_context,
                    "rag_results": rag_results,
                    "label": record.get("label", None),
                }

                f.write(
                    json.dumps(
                        output_record,
                        ensure_ascii=False,
                        default=str,
                    )
                    + "\n"
                )

                if (idx + 1) % 100 == 0:
                    logger.info(f"{split_name}: processed {idx + 1}/{len(records)}")

    logger.info(f"Saved RAG dataset to: {out_path}")

    return str(out_path)


def build_dataset_from_config_path(config_path):
    cfg = load_config(config_path)
    return build_dataset_from_config(cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    build_dataset_from_config_path(args.config)


if __name__ == "__main__":
    main()