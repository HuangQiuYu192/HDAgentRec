# HDAgentRec

Hindsight-Guided Dynamic User Agents for Sequential Recommendation.

## Phase 1

This first research prototype follows **AgentCF's RecBole-native structure**:
`HDAgentRec` is a `SequentialRecommender` that owns a per-user agent registry;
the model delegates differentiable next-item training to RecBole SASRec, and an
agent-evaluation bridge invokes a frozen local LLM only after top-M retrieval.
This preserves RecBole's data, temporal evaluation and metric lifecycle while
replacing AgentCF's free-text mutable user memory with a typed, deterministic
dynamic state. Oracle/hindsight training is deliberately deferred to Phase 2.

```bash
pip install -r requirements.txt
pytest -q
# Put a RecBole-format Beauty.inter under dataset/Beauty/.
python scripts/run_recbole.py --dataset Beauty --config configs/recbole_sasrec.yaml
```

The RecBole `.inter` file must have `user_id`, `item_id`, and `timestamp` fields.
Point `llm.model_name` in `configs/amazon_beauty.yaml` to the pre-downloaded
Qwen3-14B directory. The local Transformers client loads the model lazily and
all validated JSON responses are cached in SQLite.

On the A40 host, the dedicated `hdagentrec` environment and the cached model can
be checked without any network access:

```bash
source /home/hqy/miniconda3/etc/profile.d/conda.sh
CUDA_VISIBLE_DEVICES=0 conda run -n hdagentrec python scripts/smoke_qwen_gpu0.py \
  --model /home/hqy/.cache/huggingface/hub/models--Qwen--Qwen3-14B/snapshots/40c069824f4251a91eefaf281ebe4c544efd3e18
```

## AgentCF comparison dataset

For the initial apples-to-apples comparison, use the included public AgentCF
benchmark `dataset/CDs-100-user-dense/`. Its original pre-split sequence files,
CD metadata, and `CDs-100-user-dense.random` candidate file are preserved
unchanged. The config below mirrors AgentCF's temporal leave-valid-and-test
protocol and 20-item candidate budget:

```bash
python scripts/run_recbole.py --dataset CDs-100-user-dense \
  --config configs/agentcf_cds_100_user_dense.yaml
```

This evaluates the SASRec candidate generator with standard full-sort RecBole
metrics. `AgentCFCandidatePool` consumes the provided `.random` file and forms
the same 19 sampled negatives plus held-out positive protocol as AgentCF. It
never uses test labels to retrieve additional candidates.
