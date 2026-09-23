"""Evaluate frozen-LLM reranking over future-safe SASRec top-M candidates."""
from __future__ import annotations
import argparse
import json
import numpy as np
import torch
from pathlib import Path
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed
from hdagentrec.agent import DynamicUserAgent, TransformersLLMClient
from hdagentrec.cache import SQLiteCache
from hdagentrec.model import HDAgentRec
from hdagentrec.phase1 import Phase1Metrics, load_item_metadata, replay_history, rerank_prompt
from hdagentrec.trainer import AgentCFCandidatePool


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
    parser.add_argument("--only-candidate-hits", action="store_true", help="evaluate reranking only where SASRec already retrieved the target")
    parser.add_argument("--agentcf-10", action="store_true", help="use AgentCF's 9-negative plus held-out-positive candidate protocol")
    parser.add_argument("--candidate-file", help="AgentCF .random file; defaults to config data_path/agentcf_candidate_file")
    parser.add_argument("--trace-output", help="optional JSONL per-query rank trace")
    args = parser.parse_args()
    config = Config(model=HDAgentRec, dataset=args.dataset, config_file_list=[args.config]); init_seed(config["seed"], config["reproducibility"])
    dataset = create_dataset(config); train_data, _valid_data, test_data = data_preparation(config, dataset)
    model = HDAgentRec(config, train_data.dataset).to(config["device"]); model.load_state_dict(_checkpoint_state(args.checkpoint)); model.eval()
    raw_metadata = load_item_metadata(args.metadata)
    metadata = {index: raw_metadata.get(str(token), f"item_id: {token}") for index, token in enumerate(dataset.field2id_token[model.ITEM_ID])}
    agent = DynamicUserAgent(TransformersLLMClient(args.model, cache=SQLiteCache(args.cache))); metrics = Phase1Metrics(candidate_size=10 if args.agentcf_10 else config["agent_candidate_m"]); traces = []
    pool, rng = None, np.random.RandomState(config["seed"])
    if args.agentcf_10:
        candidate_file = args.candidate_file or str(__import__("pathlib").Path(config["data_path"]) / config["agentcf_candidate_file"])
        pool = AgentCFCandidatePool.from_file(candidate_file, dataset.field2token_id[model.USER_ID], dataset.field2token_id[model.ITEM_ID])
    with torch.no_grad():
        for interaction, _history, _positive_u, positive_i in test_data:
            interaction = interaction.to(model.device)
            histories = interaction[model.ITEM_ID + config["LIST_SUFFIX"]].cpu().tolist(); lengths = interaction[config["ITEM_LIST_LENGTH_FIELD"]].cpu().tolist()
            targets = positive_i.cpu().tolist()
            if pool is None:
                top_items, _top_scores, entropy = model.candidate_topk(interaction, config["agent_candidate_m"])
                candidate_lists, backbone_lists, uncertainties = top_items.cpu().tolist(), top_items.cpu().tolist(), entropy.cpu().tolist()
            else:
                all_scores = model.full_sort_predict(interaction).view(len(interaction), model.n_items)
                candidate_lists, backbone_lists, uncertainties = [], [], []
                for row_scores, user_id, target in zip(all_scores, interaction[model.USER_ID].cpu().tolist(), targets):
                    candidates = pool.with_positive(user_id, int(target), budget=10); rng.shuffle(candidates)
                    tensor = torch.tensor(candidates, device=model.device)
                    scores = row_scores[tensor]
                    ranked = tensor[scores.argsort(descending=True)].cpu().tolist()
                    probabilities = torch.softmax(scores, dim=0)
                    candidate_lists.append(candidates); backbone_lists.append(ranked)
                    uncertainties.append(float(-(probabilities * probabilities.clamp_min(1e-12).log()).sum().item()))
            for target, history, length, items, backbone_ranking, uncertainty in zip(targets, histories, lengths, candidate_lists, backbone_lists, uncertainties):
                if args.max_queries and metrics.queries >= args.max_queries:
                    continue
                if args.only_candidate_hits and int(target) not in backbone_ranking:
                    continue
                history = [int(item) for item in history[-int(length):] if item]
                state = replay_history(agent, history, metadata)
                candidates = [{"candidate_id": int(item), "metadata": metadata.get(int(item), f"item_id: {item}")} for item in items]
                recent_k = config["recent_k"]
                recent = [{"item_id": item, "metadata": metadata.get(item, f"item_id: {item}")} for item in history[-recent_k:]]
                reranked = agent.rerank([int(item) for item in items], rerank_prompt(state, recent, candidates, float(uncertainty)))
                metrics.add([int(item) for item in backbone_ranking], reranked, int(target))
                traces.append({"query_index": metrics.queries, "target": int(target), "backbone_rank": [int(item) for item in backbone_ranking].index(int(target)) + 1, "agent_rank": reranked.index(int(target)) + 1, "candidate_ids": [int(item) for item in items]})
    if args.trace_output:
        Path(args.trace_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.trace_output).write_text("".join(json.dumps(row) + "\n" for row in traces), encoding="utf-8")
    print({**metrics.report(), "cost": agent.cost_metrics})


if __name__ == "__main__": main()
