"""Train/evaluate the HDAgentRec backbone through RecBole's native lifecycle."""
from __future__ import annotations

import argparse

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_trainer, init_logger, init_seed

from hdagentrec.model import HDAgentRec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="RecBole .inter dataset name/path stem")
    parser.add_argument("--config", default="configs/recbole_sasrec.yaml")
    args, extra = parser.parse_known_args()
    config = Config(model=HDAgentRec, dataset=args.dataset, config_file_list=[args.config], config_dict={})
    # RecBole command-style overrides retain its normal experiment workflow.
    for value in extra:
        if value.startswith("--") and "=" in value:
            key, setting = value[2:].split("=", 1)
            config[key] = setting
    init_seed(config["seed"], config["reproducibility"]); init_logger(config)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model = HDAgentRec(config, train_data.dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, saved=True, show_progress=config["show_progress"])
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config["show_progress"])
    print({"best_valid_score": best_valid_score, "best_valid_result": best_valid_result, "test_result": test_result})


if __name__ == "__main__": main()
