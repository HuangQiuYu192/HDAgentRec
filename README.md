# HDAgentRec

Hindsight-Guided Dynamic User Agents for Sequential Recommendation.

## Phase 1

This first research prototype contains a PyTorch SASRec backbone, strict temporal
data splitting, structured dynamic user state, a provider-independent LLM client,
SQLite response caching, and top-20 candidate reranking. Oracle/hindsight training
is deliberately deferred to Phase 2.

```bash
pip install -r requirements.txt
pytest -q
python scripts/train_backbone.py --interactions data/Beauty/interactions.csv --device cuda:0
```

The interactions CSV must have `user_id`, `item_id`, and `timestamp` columns.
Point `llm.model_name` in `configs/amazon_beauty.yaml` to the pre-downloaded
Qwen3-14B directory. The local Transformers client loads the model lazily and
all validated JSON responses are cached in SQLite.
