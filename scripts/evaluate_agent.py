"""Evaluate frozen-LLM reranking over future-safe SASRec top-M candidates."""
from __future__ import annotations
import argparse
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed
from hdagentrec.agent import DynamicUserAgent, TransformersLLMClient
from hdagentrec.cache import SQLiteCache
from hdagentrec.model import HDAgentRec
from hdagentrec.phase1 import Phase1Metrics, load_item_metadata, replay_history, rerank_prompt


def _checkpoint_state(path: str):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True); parser.add_argument("--config", default="configs/recbole_sasrec.yaml")
    parser.add_argument("--checkpoint", required=True, help="trusted checkpoint produced by scripts/run_recbole.py")
    parser.add_argument("--model", required=True, help="local frozen model directory, e.g. Qwen3-14B cache")
    parser.add_argument("--metadata", required=True, help="RecBole .item file")
    parser.add_argument("--cache", default="outputs/phase1_llm.sqlite")
    parser.add_argument("--max-queries", type=int, default=10, help="safe pilot default; use 0 for all queries")
    args = parser.parse_args()
    config = Config(model=HDAgentRec, dataset=args.dataset, config_file_list=[args.config]); init_seed(config["seed"], config["reproducibility"])
    dataset = create_dataset(config); train_data, _valid_data, test_data = data_preparation(config, dataset)
    model = HDAgentRec(config, train_data.dataset).to(config["device"]); model.load_state_dict(_checkpoint_state(args.checkpoint)); model.eval()
    raw_metadata = load_item_metadata(args.metadata)
    metadata = {index: raw_metadata.get(str(token), f"item_id: {token}") for index, token in enumerate(dataset.field2id_token[model.ITEM_ID])}
    agent = DynamicUserAgent(TransformersLLMClient(args.model, cache=SQLiteCache(args.cache))); metrics = Phase1Metrics()
    with torch.no_grad():
        for interaction, _history, _positive_u, positive_i in test_data:
            interaction = interaction.to(model.device); top_items, _top_scores, entropy = model.candidate_topk(interaction, config["agent_candidate_m"])
            histories = interaction[model.ITEM_ID + config["LIST_SUFFIX"]].cpu().tolist(); lengths = interaction[config["ITEM_LIST_LENGTH_FIELD"]].cpu().tolist()
            for target, history, length, items, uncertainty in zip(positive_i.cpu().tolist(), histories, lengths, top_items.cpu().tolist(), entropy.cpu().tolist()):
                if args.max_queries and metrics.queries >= args.max_queries:
                    print({**metrics.report(), "cost": agent.cost_metrics}); return
                history = [int(item) for item in history[-int(length):] if item]
                state = replay_history(agent, history, metadata)
                candidates = [{"candidate_id": int(item), "metadata": metadata.get(int(item), f"item_id: {item}")} for item in items]
                recent = [{"item_id": item, "metadata": metadata.get(item, f"item_id: {item}")} for item in history[-config.get("recent_k", 10):]]
                reranked = agent.rerank([int(item) for item in items], rerank_prompt(state, recent, candidates, float(uncertainty)))
                metrics.add([int(item) for item in items], reranked, int(target))
    print({**metrics.report(), "cost": agent.cost_metrics})


if __name__ == "__main__": main()
