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
