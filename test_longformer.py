import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utilities import (
    build_rag_from_config,
    format_rag_context,
    build_longformer_input,
    limit_df,
    load_pubhealth_from_config,
    build_reformulater_from_config
)


def build_rag_search_fn(cfg):
    """
    Connect your RAG setup here.

    Must return:
        rag_search_fn(query: str) -> list[dict]
    """

    dfs = load_pubhealth_from_config(cfg)
    rag = build_rag_from_config(dfs, cfg)

    rag_cfg = cfg["rag"]
    def search(query: str):
        return rag.query(
            query=query,
            top_k=rag_cfg["top_k"],
            method=rag_cfg["method"],
        )

    return search



def build_examples_for_inference(claims, cfg, rag_search_fn, reformulater=None):
    """
    Builds Longformer-ready inference examples.

    This does:
      claims -> reformulated queries -> RAG -> rag_context
    """

    if isinstance(claims, str):
        claims = [claims]

    claims = list(claims)

    if cfg["reformulation"]["enabled"]:
        if reformulater is None:
            raise ValueError("Reformulation enabled but reformulater is None.")

        reformulated_queries = reformulater.reformulate_batch(claims)
    else:
        reformulated_queries = claims

    examples = []

    for claim, reformulated_query in zip(claims, reformulated_queries):
        claim = str(claim)
        reformulated_query = str(reformulated_query)

        if cfg["rag"]["query_source"] == "original":
            rag_query = claim
        elif cfg["rag"]["query_source"] == "reformulated":
            rag_query = reformulated_query
        else:
            raise ValueError("rag.query_source must be 'original' or 'reformulated'.")

        rag_results = rag_search_fn(rag_query)

        rag_context = format_rag_context(rag_results)

        examples.append(
            {
                "claim": claim,
                "reformulated_query": reformulated_query,
                "rag_query": rag_query,
                "rag_results": rag_results,
                "rag_context": rag_context,
            }
        )

    return examples


def predict_claims_from_config(cfg, claims):
    """
    Batch inference.

    Args:
        cfg: Loaded config dictionary.
        claims: list[str] or a single claim string.
        model_dir: Optional trained model directory.
        batch_size: Optional inference batch size.

    Returns:
        list[dict]
    """

    if isinstance(claims, str):
        claims = [claims]

    claims = list(claims)

    if len(claims) == 0:
        return []

    model_dir = cfg["inference"]["model_dir"] or cfg["training"]["output_dir"]
    batch_size = cfg["inference"]["batch_size"]

    rag_search_fn = build_rag_search_fn(cfg) #TODO: IMP!!! THIS KEEPS REBUILDING RAG
    reformulater = build_reformulater_from_config(cfg)

    label_map = cfg["longformer"]["label_map"]
    id_to_label = {int(v): k for k, v in label_map.items()}

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    examples = build_examples_for_inference(
        claims=claims,
        cfg=cfg,
        rag_search_fn=rag_search_fn,
        reformulater=reformulater,
    )

    predictions = []

    for start in range(0, len(examples), batch_size):
        batch_examples = examples[start:start + batch_size]

        texts = [
            build_longformer_input(example)
            for example in batch_examples
        ]

        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg["longformer"]["max_length"],
        )

        if "longformer" in cfg["longformer"]["model_name"].lower():
            global_attention_mask = torch.zeros_like(inputs["input_ids"])
            global_attention_mask[:, 0] = 1
            inputs["global_attention_mask"] = global_attention_mask

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = model(**inputs)

        probs_batch = torch.softmax(outputs.logits, dim=-1)
        pred_ids = torch.argmax(probs_batch, dim=-1)

        for example, probs, pred_id_tensor in zip(batch_examples, probs_batch, pred_ids):
            pred_id = int(pred_id_tensor.item())

            predictions.append(
                {
                    "claim": example["claim"],
                    "reformulated_query": example["reformulated_query"],
                    "rag_query": example["rag_query"],
                    "predicted_label_id": pred_id,
                    "predicted_label": id_to_label.get(pred_id, str(pred_id)),
                    "probabilities": {
                        id_to_label.get(i, str(i)): float(probs[i].item())
                        for i in range(len(probs))
                    },
                    "rag_results": example["rag_results"],
                }
            )

    return predictions
def predict_pubhealth_split_from_config(cfg, splits="test"):
    """
    Loads PUBHEALTH using cfg, extracts claims from one split,
    then runs batch inference using predict_claims_from_config.

    Args:
        cfg: Loaded config dictionary.
        split_name: "train", "validation", or "test".

    Returns:
        list[dict]: predictions for that split.
    """
    if isinstance(splits, str): splits = [splits]

    dataframes = load_pubhealth_from_config(cfg, splits)

    claim_column = "claim"

    max_examples = cfg["inference"]["max_examples"]
    predictions = {}
    for split, df in dataframes.items():
        dataframes[split] = limit_df(df, max_examples)
        claims = dataframes[split][claim_column].astype(str).tolist()

        predictions[split] = predict_claims_from_config(
            cfg=cfg,
            claims=claims,
        )

    return predictions