import uuid

from utilities import load_config
from train_longformer import train_longformer_from_config
from test_longformer import predict_pubhealth_split_from_config
from build_rag_dataset import build_dataset_from_config

import json
from pathlib import Path
from datetime import datetime

CONFIGS_LOC = "configs"

def run_train_then_test(cfg: str):
    """
    Loads config, trains Longformer, then tests it.

    Assumes:
      - train_longformer_from_config(cfg) handles all training logic
      - test_longformer_from_config(cfg) handles all testing logic
    """

    # cfg = load_config(config_path)
    train_result = train_longformer_from_config(cfg)

    test_result = predict_pubhealth_split_from_config(cfg)

    return {
        "train_result": train_result,
        "test_result": test_result,
    }

if __name__ == "__main__":

    folder = Path(CONFIGS_LOC)  # change this
    i = 0
    for config_path in folder.glob("*.json"):
        config = load_config(config_path)

        run_id = str(uuid.uuid4())

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{timestamp}_{run_id}"

        model_dir = f"models/longformer_pubhealth_{run_id}"
        config["training"]["output_dir"] = model_dir

        print(f"\n--- Training run number {i} ---")
        run_train_then_test(config)
        new_config_path = Path("models") /f"longformer_pubhealth_{run_id}" / "config" / "config.json"
        new_config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(new_config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
        i += 1