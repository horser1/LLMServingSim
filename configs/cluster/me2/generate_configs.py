#!/usr/bin/env python3
"""Generate all me2 cluster config files for Motivation Experiment 1.

Baselines:
  B1 – Homogeneous long-context colocated (y1 only)
  B2 – Context-only (y2 + y3, short:long = 35:65)
  B3 – P/D-only (y4 only, ratios: 1P:3D, 2P:2D, 3P:1D)
  B4 – Decode-biased-only (y1 + y5)
  B5 – Coexisting (y2 + y3 + y4 + y5)

GPU budgets: 8, 10, 12, 14, 16, 18, 20
"""

import json
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_NAME = "meta-llama/Llama-3.1-8B"
HARDWARE = "RTXPRO6000"
NPU_MEM = {"mem_size": 96, "mem_bw": 1597, "mem_latency": 0}
CPU_MEM = {"mem_size": 512, "mem_bw": 256, "mem_latency": 0}
LINK_BW = 16
LINK_LATENCY = 20000

GPU_BUDGETS = [8, 10, 12, 14, 16, 18, 20]

SHORT_ADMISSION = {"max_total_toks": 4096}
LONG_ADMISSION = {"max_total_toks": 32768}
PREFILL_HEAVY_ADMISSION = {"min_input_toks": 8192, "max_input_toks": 15999}
DECODE_BIASED_ADMISSION = {
    "max_input_toks": 4096,
    "min_output_toks": 8192,
}

INSTANCE_LIMITS = {
    # Approximate constant KV-token budget per instance:
    # short 4K window -> high concurrency, long 32K window -> low concurrency.
    "short": {
        "max_num_seqs": 256,
        "max_num_batched_tokens": 2048,
    },
    "long": {
        "max_num_seqs": 32,
        "max_num_batched_tokens": 2048,
    },
    "prefill": {
        "max_num_seqs": 32,
        "max_num_batched_tokens": 2048,
        "long_prefill_token_threshold": 2048,
        "enable_chunked_prefill": True,
    },
    "decode": {
        "max_num_seqs": 256,
        "max_num_batched_tokens": 256,
        "enable_chunked_prefill": True,
    },
    "decode_biased": {
        "max_num_seqs": 128,
        "max_num_batched_tokens": 512,
        "enable_chunked_prefill": True,
    },
}


def make_instance(pd_type=None, tp_size=1, role=None):
    inst = {
        "model_name": MODEL_NAME,
        "hardware": HARDWARE,
        "npu_mem": dict(NPU_MEM),
        "pd_type": pd_type,
        "tp_size": tp_size,
    }
    if role is not None:
        inst.update(INSTANCE_LIMITS[role])
    return inst


def make_instances(count, pd_type=None, role=None):
    return [make_instance(pd_type=pd_type, role=role) for _ in range(count)]


def make_node(instances):
    return {
        "num_instances": len(instances),
        "cpu_mem": dict(CPU_MEM),
        "instances": instances,
    }


def make_config(pools, instances):
    return {
        "num_nodes": 1,
        "link_bw": LINK_BW,
        "link_latency": LINK_LATENCY,
        "pools": pools,
        "nodes": [make_node(instances)],
    }


def save_config(filename, config):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"  ✓ {filename}")


# =============================================================================
# B1: Homogeneous long-context colocated
# =============================================================================
def gen_b1():
    print("\n[B1] Homogeneous (y1 only)")
    for n in GPU_BUDGETS:
        instances = make_instances(n, pd_type=None, role="long")
        pools = [
            {
                "id": "general_long",
                "mode": "agg",
                "instances": list(range(n)),
                "admission": LONG_ADMISSION,
                "fallback": [],
            }
        ]
        config = make_config(pools, instances)
        save_config(f"me2_b1_{n}gpu.json", config)


# =============================================================================
# B2: Context-only (y2 short + y3 long, 35:65)
# =============================================================================
def gen_b2():
    print("\n[B2] Context-only (y2 + y3, short:long = 35:65)")
    for n in GPU_BUDGETS:
        n_short = max(1, int(round(n * 0.35)))
        n_long = n - n_short

        instances = []
        instances += make_instances(n_short, pd_type=None, role="short")
        instances += make_instances(n_long, pd_type=None, role="long")

        short_ids = list(range(0, n_short))
        long_ids = list(range(n_short, n))

        pools = [
            {
                "id": "short",
                "mode": "agg",
                "instances": short_ids,
                "admission": SHORT_ADMISSION,
                "fallback": ["long"],
            },
            {
                "id": "long",
                "mode": "agg",
                "instances": long_ids,
                "admission": LONG_ADMISSION,
                "fallback": [],
            },
        ]
        config = make_config(pools, instances)
        save_config(f"me2_b2_{n}gpu.json", config)


# =============================================================================
# B3: P/D-only (y4, ratios: 1P:3D, 2P:2D, 3P:1D)
# =============================================================================
def gen_b3():
    print("\n[B3] P/D-only (y4 only)")

    ratios = [
        ("1p3d", 1, 3),   # 1P:3D → prefill=25%, decode=75%
        ("2p2d", 1, 1),   # 2P:2D → prefill=50%, decode=50%
        ("3p1d", 3, 1),   # 3P:1D → prefill=75%, decode=25%
    ]

    for label, p_ratio, d_ratio in ratios:
        denom = p_ratio + d_ratio
        if p_ratio == d_ratio:
            # 1:1 works for all budgets
            budgets = GPU_BUDGETS
        else:
            # Only budgets divisible by 4 (1+3=4, 3+1=4)
            budgets = [b for b in GPU_BUDGETS if b % denom == 0]

        for n in budgets:
            n_prefill = int(round(n * p_ratio / denom))
            n_decode = n - n_prefill

            instances = []
            instances += make_instances(n_prefill, pd_type="prefill", role="prefill")
            instances += make_instances(n_decode, pd_type="decode", role="decode")

            prefill_ids = list(range(0, n_prefill))
            decode_ids = list(range(n_prefill, n))

            pools = [
                {
                    "id": "pd_path",
                    "mode": "pd",
                    "prefill_instances": prefill_ids,
                    "decode_instances": decode_ids,
                    "admission": LONG_ADMISSION,
                    "fallback": [],
                }
            ]
            config = make_config(pools, instances)
            save_config(f"me2_b3_{n}gpu_{label}.json", config)


# =============================================================================
# B4: Decode-biased-only (y1 general + y5 decode_biased)
# =============================================================================
def gen_b4():
    print("\n[B4] Decode-biased-only (y1 + y5)")
    for n in GPU_BUDGETS:
        n_decode_bias = max(1, int(round(n * 0.30)))
        n_general = n - n_decode_bias

        instances = []
        instances += make_instances(n_decode_bias, pd_type=None, role="decode_biased")
        instances += make_instances(n_general, pd_type=None, role="long")

        decode_bias_ids = list(range(0, n_decode_bias))
        general_ids = list(range(n_decode_bias, n))

        pools = [
            {
                "id": "decode_biased",
                "mode": "agg",
                "instances": decode_bias_ids,
                "admission": DECODE_BIASED_ADMISSION,
                "fallback": ["general_long"],
            },
            {
                "id": "general_long",
                "mode": "agg",
                "instances": general_ids,
                "admission": LONG_ADMISSION,
                "fallback": [],
            },
        ]
        config = make_config(pools, instances)
        save_config(f"me2_b4_{n}gpu.json", config)


# =============================================================================
# B5: Coexisting (y2 + y3 + y4 + y5)
# =============================================================================
def gen_b5():
    print("\n[B5] Coexisting (y2 + y3 + y4 + y5)")

    # GPU allocation ratios (sum = 100%)
    RATIO_SHORT = 0.30
    RATIO_LONG = 0.25
    RATIO_PREFILL = 0.10
    RATIO_DECODE = 0.10
    RATIO_DECODE_BIASED = 0.25

    for n in GPU_BUDGETS:
        # Round each, with minimum 1 per pool (prefill+decode minimum 2 total)
        allocations = {
            "short":       max(1, int(round(n * RATIO_SHORT))),
            "decode_bias": max(1, int(round(n * RATIO_DECODE_BIASED))),
            "prefill":     max(1, int(round(n * RATIO_PREFILL))),
            "decode":      max(1, int(round(n * RATIO_DECODE))),
            "long":        max(1, int(round(n * RATIO_LONG))),
        }

        total = sum(allocations.values())
        # Adjust long pool to make total = n
        diff = n - total
        allocations["long"] += diff

        # Safety: ensure no negative
        if allocations["long"] < 1:
            # steal from the largest pool
            largest = max(allocations, key=lambda k: allocations[k] if k != "long" else -1)
            while allocations["long"] < 1:
                if allocations[largest] > 1:
                    allocations[largest] -= 1
                    allocations["long"] += 1
                else:
                    # find next largest
                    del allocations[largest]
                    largest = max(allocations, key=allocations.get)

        n_short = allocations["short"]
        n_decode_bias = allocations["decode_bias"]
        n_prefill = allocations["prefill"]
        n_decode = allocations["decode"]
        n_long = allocations["long"]

        # Build instances in order: short, decode_bias, prefill, decode, long
        instances = []
        idx = 0

        short_ids = list(range(idx, idx + n_short))
        instances += make_instances(n_short, pd_type=None, role="short")
        idx += n_short

        decode_bias_ids = list(range(idx, idx + n_decode_bias))
        instances += make_instances(n_decode_bias, pd_type=None, role="decode_biased")
        idx += n_decode_bias

        prefill_ids = list(range(idx, idx + n_prefill))
        instances += make_instances(n_prefill, pd_type="prefill", role="prefill")
        idx += n_prefill

        decode_ids = list(range(idx, idx + n_decode))
        instances += make_instances(n_decode, pd_type="decode", role="decode")
        idx += n_decode

        long_ids = list(range(idx, idx + n_long))
        instances += make_instances(n_long, pd_type=None, role="long")

        # Pool order: specific → general (first match wins)
        pools = [
            {
                "id": "short",
                "mode": "agg",
                "instances": short_ids,
                "admission": SHORT_ADMISSION,
                "fallback": ["long"],
            },
            {
                "id": "decode_biased",
                "mode": "agg",
                "instances": decode_bias_ids,
                "admission": DECODE_BIASED_ADMISSION,
                "fallback": ["long"],
            },
            {
                "id": "pd_path",
                "mode": "pd",
                "prefill_instances": prefill_ids,
                "decode_instances": decode_ids,
                "admission": PREFILL_HEAVY_ADMISSION,
                "fallback": ["long"],
            },
            {
                "id": "long",
                "mode": "agg",
                "instances": long_ids,
                "admission": LONG_ADMISSION,
                "fallback": [],
            },
        ]
        config = make_config(pools, instances)
        save_config(f"me2_b5_{n}gpu.json", config)

        # Print allocation for verification
        print(f"    N={n}: short={n_short}, decode_bias={n_decode_bias}, "
              f"prefill={n_prefill}, decode={n_decode}, long={n_long}")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print(f"Generating configs into: {OUTPUT_DIR}")
    gen_b1()
    gen_b2()
    gen_b3()
    gen_b4()
    gen_b5()

    # Count generated files
    files = sorted(f for f in os.listdir(OUTPUT_DIR)
                   if f.startswith("me2_b") and f.endswith(".json"))
    print(f"\n{'='*60}")
    print(f"Total config files generated: {len(files)}")
    for f in files:
        print(f"  {f}")
