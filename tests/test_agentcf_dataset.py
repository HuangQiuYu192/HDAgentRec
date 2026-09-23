from pathlib import Path

from hdagentrec.trainer import AgentCFCandidatePool


ROOT = Path(__file__).resolve().parents[1] / "dataset" / "CDs-100-user-dense"


def test_agentcf_presplit_files_are_present_and_sequential():
    for split in ("train", "valid", "test"):
        lines = (ROOT / f"CDs-100-user-dense.{split}.inter").read_text(encoding="utf-8").splitlines()
        assert lines[0] == "user_id:token\titem_id_list:token_seq\titem_id:token"
        assert len(lines) > 100


def test_agentcf_candidate_protocol_removes_then_adds_positive(tmp_path):
    path = tmp_path / "candidates.random"
    path.write_text("u\ta b p c\n", encoding="utf-8")
    pool = AgentCFCandidatePool.from_file(path, {"u": 1}, {"a": 2, "b": 3, "p": 4, "c": 5})
    assert pool.with_positive(1, 4, budget=3) == [2, 3, 4]
