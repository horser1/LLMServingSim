# ME2 Cluster Configs

These configs compare homogeneous, context-specialized, P/D-isolated,
decode-biased, and hybrid serving plans for `meta-llama/Llama-3.1-8B` on
RTXPRO6000 instances.

## Pool roles

Pool `admission` is token-class admission. Instance limits are scheduler
capacity knobs.

| Role | Admission | Instance limits |
| --- | --- | --- |
| `short` | `max_total_toks = 4096` | `max_num_seqs = 256`, `max_num_batched_tokens = 2048` |
| `long` / `general_long` | `max_total_toks = 32768` | `max_num_seqs = 32`, `max_num_batched_tokens = 2048` |
| `prefill` | B3: `max_total_toks = 32768`; B5: `8192 <= input_toks <= 15999` | `max_num_seqs = 32`, `max_num_batched_tokens = 2048`, `long_prefill_token_threshold = 2048` |
| `decode` | Same P/D path as prefill | `max_num_seqs = 256`, `max_num_batched_tokens = 256` |
| `decode_biased` | `input_toks <= 4096`, `output_toks >= 8192` | `max_num_seqs = 128`, `max_num_batched_tokens = 512` |

The concurrency settings use the standard vLLM capacity relationship:
`max_num_seqs ~= floor(KV_token_budget / admitted_context_window)`. For
Llama-3.1-8B bf16, one KV token is 128 KiB per instance. On a 96 GiB
RTXPRO6000, after model weights this leaves about 663k KV tokens, enough for
about 162 x 4K sequences or 20 x 32K sequences. The configs keep round,
profiler-friendly limits while preserving the short-vs-long capacity gap.

`max_num_batched_tokens` controls per-step token work. Short and long colocated
pools use the profiled 2048-token sweep bound. Prefill roles also use 2048 and
cap each long prefill chunk at 2048. Decode roles use smaller token budgets so
many one-token decode steps can batch without letting large prefill chunks
dominate that pool.

## Arrival rate

`workloads/me2/workload_me2_01_mixed.jsonl` is already in the useful pressure
range:

| Slice | Requests | Effective arrival rate |
| --- | ---: | ---: |
| Full file | 608 | 9.97 req/s |
| Script default, `--num-reqs 120` | 120 | 9.44 req/s |

For this config family, use about 8-12 req/s to expose the short/long and
prefill/decode differences. Below about 2 req/s the system is often idle, so
pool specialization is mostly hidden by arrival gaps. Above about 15 req/s,
queueing dominates and is useful as a stress test, but it is less clean for
showing the role-specific tradeoff.
