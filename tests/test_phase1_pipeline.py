from pathlib import Path
from hdagentrec.agent import DynamicUserAgent, LLMClient
from hdagentrec.phase1 import Phase1Metrics, load_item_metadata, replay_history, rerank_prompt
from hdagentrec.schemas import UserState
from hdagentrec.agent import _first_json_object


class Stub(LLMClient):
    def _generate(self, prompt):
        if "new_interaction" in prompt:
            return ('{"interaction_type":"stable_preference","confidence":0.8,"reason":"history","concept":"rock","update_long_term":true,"update_short_term":true}', 3)
        return ('{"ranking":[2,1]}', 2)


def test_history_replay_uses_only_given_prefix():
    state = replay_history(DynamicUserAgent(Stub("stub")), [3, 4], {3: "old", 4: "recent"})
    assert [entry.item_id for entry in state.evidence] == [3, 4]


def test_metrics_report_candidate_coverage_and_rerank_gain():
    result = Phase1Metrics(); result.add([1, 2], [2, 1], 2)
    report = result.report(); assert report["candidate_coverage@20"] == 1.0
    assert report["agent"]["NDCG@5"] > report["backbone"]["NDCG@5"]


def test_metadata_loader_and_prompt_are_json_bounded(tmp_path: Path):
    item_file = tmp_path / "items.item"; item_file.write_text("item_id:token\ttitle:token_seq\n7\tExample\n", encoding="utf-8")
    assert "Example" in load_item_metadata(item_file)["7"]
    assert "candidate_id" in rerank_prompt(UserState(), [], [{"candidate_id": 1, "metadata": "x"}], 0.2)


def test_json_parser_accepts_a_model_output_label():
    assert _first_json_object('OUTPUT_JSON:\n{"ranking":[2,1]}') == '{"ranking":[2,1]}'
