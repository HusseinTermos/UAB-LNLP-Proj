from utilities import load_config
from train_longformer import train_longformer_from_config
from test_longformer import predict_pubhealth_split_from_config
from build_rag_dataset import build_dataset_from_config

def run_train_then_test(config_path: str):
    """
    Loads config, trains Longformer, then tests it.

    Assumes:
      - train_longformer_from_config(cfg) handles all training logic
      - test_longformer_from_config(cfg) handles all testing logic
    """

    cfg = load_config(config_path)
    train_result = train_longformer_from_config(cfg)

    test_result = predict_pubhealth_split_from_config(cfg)

    return {
        "train_result": train_result,
        "test_result": test_result,
    }

config_path = "config_example.json"
print(run_train_then_test(config_path))