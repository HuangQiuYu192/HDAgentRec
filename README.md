# HDAgentRec

Hindsight-Guided Dynamic User Agents for Sequential Recommendation.

## Phase 1

This first research prototype is **RecBole-native**, like the reproducible
backbone layer in AgentCF-style systems. It subclasses RecBole's maintained
`SASRec`, so data handling, strict leave-two-out temporal evaluation, trainer,
full-sort ranking and metrics stay within RecBole. HDAgentRec contributes the
structured dynamic user state, provider-independent LLM client, SQLite response
cache, and a top-20 candidate reranking boundary. Oracle/hindsight training is
deliberately deferred to Phase 2.

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
