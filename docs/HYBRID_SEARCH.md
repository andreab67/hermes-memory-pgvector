# Hybrid Search & Relevance Tuning in Hexus

Hexus supports advanced hybrid search over both core memory entries (`memory_entries` table) and conversation history (`conversations` table). This combines the semantic understanding of vector similarity search with the precise keyword matching of Postgres Full-Text Search, along with dynamic relevance boosting based on age (temporal decay) and retrieval frequency (recall boost).

---

## 1. Hybrid Search Model

The hybrid search architecture blends semantic vector distance and full-text keyword matching (using a double-CTE query):

`Combined Score = (w_vector × S_vector) + (w_text × S_text)`

* **Vector Similarity (`S_vector`)**: Calculated as `1 - cosine_distance(embedding, query)`.
* **Text Similarity (`S_text`)**: Uses Postgres `ts_rank` with English full-text indexing (`to_tsvector` and `websearch_to_tsquery`).
* **Weights**: You can balance the contribution of both using `vector_weight` (`w_vector`) and `text_weight` (`w_text`). The default weights are `0.7` and `0.3` respectively.

---

## 2. Relevance Adjustments

Hexus applies post-retrieval mathematical scoring adjustments to reflect memory freshness and historical utility:

### Temporal Decay
Memory relevance degrades over time (newer items are prioritized). Decay is modeled exponentially based on age:

`Score_decayed = Score × 2^(-Age / HalfLife)`

* **Half-Life (`decay_half_life_days`):** Specifies the time period (in days) after which a memory's score is halved.
* Setting `decay_half_life_days = 0.0` (default) disables temporal decay.

### Recall Boosting
Frequently recalled memories or conversation turns receive a logarithmic score boost to prioritize topics that are frequently requested:

`Score_boosted = Score × (1.0 + w_boost × ln(1 + recall_count))`

* **Recall Count:** The database increments `recall_count` in the item's JSONB `metadata` every time it is included in search results.
* **Boost Weight (`recall_boost_weight`):** Controls the scale of the boost.
* Setting `recall_boost_weight = 0.0` (default) disables recall boosting.

---

## 3. Configuration & Usage

### FastMCP Tool Usage
When using the standalone MCP server, you can supply these parameters directly in the tool arguments for `memory_hybrid_search` or `memory_hybrid_recall_turns`:

```json
{
  "name": "memory_hybrid_recall_turns",
  "arguments": {
    "query": "Postgres connection pool optimization",
    "top_k": 5,
    "vector_weight": 0.6,
    "text_weight": 0.4,
    "decay_half_life_days": 14.0,
    "recall_boost_weight": 0.15
  }
}
```

### Hermes Agent Plugin Configuration
For the Hermes runtime agent, specify these default values in your `$HERMES_HOME/config.yaml` file under `plugins.hexus`:

```yaml
plugins:
  hexus:
    dsn: "dbname=hermes_memory user=hermes host=/var/run/postgresql"
    # Hybrid search parameters
    vector_weight: 0.7
    text_weight: 0.3
    decay_half_life_days: 7.0
    recall_boost_weight: 0.1
```
