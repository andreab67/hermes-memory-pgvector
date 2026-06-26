# Benchmarks

Performance baselines for hexus. These are **reference numbers** — your results will vary based on hardware, Postgres tuning, and workload.

## Hardware Reference

| Component | Specification |
|-----------|---------------|
| CPU | Intel Core i7 (4 cores / 8 threads, ~3.0-3.5 GHz base) |
| RAM | 16 GB DDR4 |
| Storage | NVMe SSD |
| GPU | None (CPU-only inference) |
| OS | Linux (containerized) |

> **Note:** These numbers are from a single-socket i7 with 16 GB RAM. Scale expectations accordingly — a modern 8-core i7/AMD Ryzen 7 will be ~1.5-2× faster on embed throughput; a 32 GB+ machine with tuned Postgres will handle larger corpora with lower latency.

---

## Embedder: LocalBertEmbedder (MiniLM-L6-v2, 384-dim)

The local sentence-transformers model runs entirely in-process. No network calls, no external service dependency.

### Single-Embed Latency

| Runs | Mean | Stddev | Min | Max |
|------|------|--------|-----|-----|
| 10 | 7.4 ms | 0.2 ms | 7.2 ms | 8.0 ms |

*Text: ~120 chars. First call after model load includes lazy initialization overhead (~500 ms cold). Subsequent calls are consistent.*

### Batch Throughput

| Batch Size | Total Time | Per Item | Items/sec |
|------------|------------|----------|-----------|
| 1 | 6.2 ms | 6.19 ms | 161 |
| 8 | 9.0 ms | 1.13 ms | 887 |
| 32 | 21.5 ms | 0.67 ms | 1,486 |
| 128 | 75.2 ms | 0.59 ms | 1,702 |

**Key insight:** Batch > 32 saturates the CPU; diminishing returns past batch=64. For ingestion pipelines, batch 32-64 is the sweet spot.

### Model Load (Cold Start)

| Metric | Value |
|--------|-------|
| Model size (disk) | ~88 MB (MiniLM-L6-v2) |
| RAM resident (loaded) | ~500 MB |
| First `ensure_loaded()` | ~500-800 ms |
| Subsequent calls | < 1 ms |

The `embed_eager_load: true` plugin option pre-loads at startup, shifting the 500 ms cost to init.

---

## MemoryStore: Postgres + hexus (HNSW)

Test corpus: 1,000 documents, 384-dim vectors, HNSW index (m=16, ef_construction=64).

### Ingestion

| Operation | Time | Throughput |
|-----------|------|------------|
| Embed 1,000 docs | 3.26 s | 307 docs/sec |
| Insert 1,000 rows | 2.27 s | 440 inserts/sec |

*Inserts are single-row `INSERT ... ON CONFLICT`. Bulk `COPY` + batched embeds would be 5-10× faster for initial loads.*

### Recall Latency (p50 / p95)

| Top-K | Mean | p95 | Results |
|-------|------|-----|---------|
| 5 | 2.0 ms | 3.7 ms | 5 |
| 10 | 1.9 ms | 2.1 ms | 10 |
| 50 | 2.2 ms | 2.5 ms | 50 |
| 100 | 2.6 ms | 3.7 ms | 100 |

*20 runs each. Sub-millisecond mean at K=5-10; p95 stays under 4 ms even at K=100.*

### Multi-Agent Isolation

| Agent Scope | Rows Returned |
|-------------|---------------|
| Agent A (owner) | 10 |
| Agent B (other) | 1 |
| Cross-agent (`agent_identity=None`) | 11 |

`agent_identity` scoping is enforced at the SQL level — zero leakage between agents.

---

## Resource Usage (Container)

| Component | Estimate |
|-----------|----------|
| MiniLM-L6-v2 (loaded) | ~500 MB RSS |
| 1,000 vectors in hexus | ~3 MB (vectors + HNSW graph) |
| Postgres 16 (idle) | ~50 MB |
| Postgres (under load) | ~100-150 MB |
| **Total (container)** | **~650-750 MB** |

No GPU, no CUDA libraries. Runs on any x86_64 Linux with Docker.

---

## Scaling Projections

| Corpus Size | Expected Recall (K=10) | RAM (vectors + HNSW) | Ingest Time (batched) |
|-------------|------------------------|----------------------|----------------------|
| 10 K | ~3 ms | ~30 MB | ~1 min |
| 100 K | ~5 ms | ~300 MB | ~10 min |
| 1 M | ~10-15 ms | ~3 GB | ~2 hrs |
| 10 M | ~25-40 ms | ~30 GB | ~20 hrs |

*Projections assume HNSW (m=16, ef_construction=64), tuned `work_mem`/`maintenance_work_mem`, and batched `COPY` for ingest. At >1M rows, consider `ef_search` tuning and partial indexes per `agent_identity`.*

---

## Running the Benchmark

### Prerequisites

- Docker + Docker Compose
- `hexus:test` image built (`docker compose -f docker/compose.yml --profile test build`)

### One-Liner

```bash
# Start Postgres + hexus
docker compose -f docker/compose.yml --profile dev up -d pg

# Run benchmark (mounts this file into the container)
cat benchmarks/bench.py | docker run --rm --entrypoint python \
  --network hexus_default \
  -e PG_TEST_DSN="dbname=hermes_test user=postgres password=postgres host=hexus-pg" \
  hexus:test
```

### With Custom Corpus Size

Edit `bench.py` and change `n_docs = 1000` to your desired size.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PG_TEST_DSN` | (required) | Postgres connection string |
| `HEXUS_EMBED_EAGER_LOAD` | `0` | Set `1` to pre-load model at startup |

---

## Interpreting Results

| Metric | Good | Investigate If |
|--------|------|----------------|
| Single embed | < 15 ms | CPU throttling, thermal limits |
| Batch 32 throughput | > 1,000 items/sec | Other processes stealing CPU |
| Recall top-10 | < 5 ms (p95) | Missing HNSW index, `work_mem` too low |
| Insert throughput | > 200 rows/sec | `autocommit` overhead, network latency |
| Cross-agent leakage | 0 rows | Bug in `agent_identity` filtering |

For production tuning, see Postgres `hexus` docs on:
- `hnsw.ef_search` (query-time accuracy/speed tradeoff)
- `work_mem` / `maintenance_work_mem` (index build + query)
- `max_parallel_workers_per_gather` (parallel index scans)
