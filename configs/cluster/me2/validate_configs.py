#!/usr/bin/env python3
"""Validate all me2 config files against the agreed specification."""
import json
import os
import sys

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DIR)

from generate_configs import (  # noqa: E402
    DECODE_BIASED_ADMISSION,
    INSTANCE_LIMITS,
    LONG_ADMISSION,
    PREFILL_HEAVY_ADMISSION,
    SHORT_ADMISSION,
)

GPU_BUDGETS = [8, 10, 12, 14, 16, 18, 20]
MODEL = "meta-llama/Llama-3.1-8B"
HARDWARE = "RTXPRO6000"
NPU_MEM = {"mem_size": 96, "mem_bw": 1597, "mem_latency": 0}
CPU_MEM = {"mem_size": 512, "mem_bw": 256, "mem_latency": 0}

errors = []
checked_files = []

def err(msg):
    errors.append(msg)
    print(f"  ❌ {msg}")

def config_exists(path):
    return os.path.exists(os.path.join(DIR, path))

def load(path):
    with open(os.path.join(DIR, path)) as f:
        cfg = json.load(f)
    checked_files.append(path)
    return cfg

def check_common(cfg, name, expected_n):
    """Check fields common to all configs."""
    if cfg["num_nodes"] != 1:
        err(f"{name}: num_nodes should be 1, got {cfg['num_nodes']}")
    if cfg["link_bw"] != 16:
        err(f"{name}: link_bw should be 16, got {cfg['link_bw']}")
    if cfg["link_latency"] != 20000:
        err(f"{name}: link_latency should be 20000, got {cfg['link_latency']}")
    if len(cfg["nodes"]) != 1:
        err(f"{name}: expected 1 node, got {len(cfg['nodes'])}")

    node = cfg["nodes"][0]
    if node["cpu_mem"] != CPU_MEM:
        err(f"{name}: cpu_mem mismatch")

    instances = node["instances"]
    if len(instances) != expected_n:
        err(f"{name}: expected {expected_n} instances, got {len(instances)}")
    if len(instances) != node["num_instances"]:
        err(f"{name}: num_instances mismatch")

    for i, inst in enumerate(instances):
        if inst["model_name"] != MODEL:
            err(f"{name}: instance[{i}] model_name should be {MODEL}")
        if inst["hardware"] != HARDWARE:
            err(f"{name}: instance[{i}] hardware should be {HARDWARE}")
        if inst["npu_mem"] != NPU_MEM:
            err(f"{name}: instance[{i}] npu_mem mismatch")
        if inst["tp_size"] != 1:
            err(f"{name}: instance[{i}] tp_size should be 1")

    # Check pool instance coverage
    all_pool_insts = set()
    for pool in cfg["pools"]:
        if pool["mode"] == "agg":
            for iid in pool["instances"]:
                if iid in all_pool_insts:
                    err(f"{name}: instance {iid} appears in multiple pools")
                all_pool_insts.add(iid)
        elif pool["mode"] == "pd":
            for iid in pool["prefill_instances"]:
                if iid in all_pool_insts:
                    err(f"{name}: prefill instance {iid} appears in multiple pools")
                all_pool_insts.add(iid)
            for iid in pool["decode_instances"]:
                if iid in all_pool_insts:
                    err(f"{name}: decode instance {iid} appears in multiple pools")
                all_pool_insts.add(iid)

    expected_set = set(range(expected_n))
    missing = expected_set - all_pool_insts
    extra = all_pool_insts - expected_set
    if missing:
        err(f"{name}: instances not in any pool: {sorted(missing)}")
    if extra:
        err(f"{name}: pool references non-existent instances: {sorted(extra)}")

    return node, instances


def check_pd_types(instances, prefill_ids, decode_ids, name):
    """Verify pd_type matches pool assignment."""
    for iid in prefill_ids:
        if instances[iid]["pd_type"] != "prefill":
            err(f"{name}: instance[{iid}] should be pd_type='prefill', got '{instances[iid]['pd_type']}'")
    for iid in decode_ids:
        if instances[iid]["pd_type"] != "decode":
            err(f"{name}: instance[{iid}] should be pd_type='decode', got '{instances[iid]['pd_type']}'")
    agg_ids = set(range(len(instances))) - set(prefill_ids) - set(decode_ids)
    for iid in agg_ids:
        if instances[iid]["pd_type"] is not None:
            err(f"{name}: instance[{iid}] should be pd_type=null, got '{instances[iid]['pd_type']}'")


def check_fallback_refs(pools, name):
    """Verify all fallback targets reference existing pools."""
    pool_ids = {p["id"] for p in pools}
    for pool in pools:
        for target in pool.get("fallback", []):
            if target not in pool_ids:
                err(f"{name}: pool '{pool['id']}' fallback target '{target}' does not exist")


def check_instance_limits(instances, instance_ids, role, name):
    """Verify per-instance scheduler limits for a pool role."""
    expected = INSTANCE_LIMITS[role]
    for iid in instance_ids:
        inst = instances[iid]
        for key, value in expected.items():
            if inst.get(key) != value:
                err(f"{name}: instance[{iid}] {key} should be {value}, got {inst.get(key)}")


# =============================================================================
# B1 validation
# =============================================================================
def validate_b1():
    print("\n=== B1: Homogeneous ===")
    for n in GPU_BUDGETS:
        fname = f"me2_b1_{n}gpu.json"
        if not config_exists(fname):
            continue
        cfg = load(fname)
        node, instances = check_common(cfg, fname, n)

        pools = cfg["pools"]
        if len(pools) != 1:
            err(f"{fname}: B1 should have 1 pool, got {len(pools)}")
            continue

        pool = pools[0]
        if pool["id"] != "general_long":
            err(f"{fname}: pool id should be 'general_long'")
        if pool["mode"] != "agg":
            err(f"{fname}: pool mode should be 'agg'")
        if pool["admission"] != LONG_ADMISSION:
            err(f"{fname}: B1 admission should be {LONG_ADMISSION}, got {pool['admission']}")
        if pool["fallback"] != []:
            err(f"{fname}: B1 should have no fallback")
        if pool["instances"] != list(range(n)):
            err(f"{fname}: pool instances should be [0..{n-1}]")

        check_pd_types(instances, [], [], fname)
        check_instance_limits(instances, pool["instances"], "long", fname)
        check_fallback_refs(pools, fname)

    if not any(e.startswith("me2_b1") for e in errors):
        print("  ✅ All B1 configs valid")


# =============================================================================
# B2 validation
# =============================================================================
def validate_b2():
    print("\n=== B2: Context-only ===")
    for n in GPU_BUDGETS:
        fname = f"me2_b2_{n}gpu.json"
        if not config_exists(fname):
            continue
        cfg = load(fname)
        node, instances = check_common(cfg, fname, n)

        pools = cfg["pools"]
        if len(pools) != 2:
            err(f"{fname}: B2 should have 2 pools, got {len(pools)}")
            continue

        short_pool = next((p for p in pools if p["id"] == "short"), None)
        long_pool = next((p for p in pools if p["id"] == "long"), None)
        if not short_pool:
            err(f"{fname}: missing 'short' pool")
        if not long_pool:
            err(f"{fname}: missing 'long' pool")
        if not short_pool or not long_pool:
            continue

        # Pool mode
        if short_pool["mode"] != "agg":
            err(f"{fname}: short pool should be agg")
        if long_pool["mode"] != "agg":
            err(f"{fname}: long pool should be agg")

        # Admission
        if short_pool["admission"] != SHORT_ADMISSION:
            err(f"{fname}: short pool admission should be {SHORT_ADMISSION}, got {short_pool['admission']}")
        if long_pool["admission"] != LONG_ADMISSION:
            err(f"{fname}: long pool admission should be {LONG_ADMISSION}, got {long_pool['admission']}")

        # Fallback
        if short_pool["fallback"] != ["long"]:
            err(f"{fname}: short fallback should be ['long']")
        if long_pool["fallback"] != []:
            err(f"{fname}: long should have no fallback")

        # Instance counts (35:65)
        n_short = len(short_pool["instances"])
        n_long = len(long_pool["instances"])
        if n_short + n_long != n:
            err(f"{fname}: short({n_short})+long({n_long}) != {n}")
        expected_ratio = 0.35
        actual_ratio = n_short / n
        if abs(actual_ratio - expected_ratio) > 0.15:
            err(f"{fname}: short ratio {actual_ratio:.2f} too far from {expected_ratio}")

        # Instance pd_type (all null)
        check_pd_types(instances, [], [], fname)
        check_instance_limits(instances, short_pool["instances"], "short", fname)
        check_instance_limits(instances, long_pool["instances"], "long", fname)

        # Pool instances are contiguous
        if short_pool["instances"] != list(range(0, n_short)):
            err(f"{fname}: short instances should be [0..{n_short-1}]")
        if long_pool["instances"] != list(range(n_short, n)):
            err(f"{fname}: long instances should be [{n_short}..{n-1}]")

        check_fallback_refs(pools, fname)

    if not any(e.startswith("me2_b2") for e in errors):
        print("  ✅ All B2 configs valid")


# =============================================================================
# B3 validation
# =============================================================================
def validate_b3():
    print("\n=== B3: P/D-only ===")
    ratios = {
        "1p3d": (1, 3),
        "2p2d": (1, 1),
        "3p1d": (3, 1),
    }

    for label, (p_ratio, d_ratio) in ratios.items():
        denom = p_ratio + d_ratio
        budgets = GPU_BUDGETS if p_ratio == d_ratio else [b for b in GPU_BUDGETS if b % denom == 0]

        for n in budgets:
            fname = f"me2_b3_{n}gpu_{label}.json"
            if not config_exists(fname):
                continue
            cfg = load(fname)
            node, instances = check_common(cfg, fname, n)

            pools = cfg["pools"]
            if len(pools) != 1:
                err(f"{fname}: B3 should have 1 pool, got {len(pools)}")
                continue

            pool = pools[0]
            if pool["id"] != "pd_path":
                err(f"{fname}: pool id should be 'pd_path'")
            if pool["mode"] != "pd":
                err(f"{fname}: pool mode should be 'pd'")
            if pool["admission"] != LONG_ADMISSION:
                err(f"{fname}: B3 admission should be {LONG_ADMISSION}, got {pool['admission']}")
            if pool["fallback"] != []:
                err(f"{fname}: B3 should have no fallback")

            n_prefill = len(pool["prefill_instances"])
            n_decode = len(pool["decode_instances"])
            if n_prefill + n_decode != n:
                err(f"{fname}: prefill({n_prefill})+decode({n_decode}) != {n}")

            # Check P:D ratio
            expected_n_prefill = round(n * p_ratio / denom)
            expected_n_decode = n - expected_n_prefill
            if n_prefill != expected_n_prefill:
                err(f"{fname}: prefill count {n_prefill}, expected ~{expected_n_prefill} (ratio {p_ratio}:{d_ratio})")
            if n_decode != expected_n_decode:
                err(f"{fname}: decode count {n_decode}, expected ~{expected_n_decode}")

            # Instance pd_type
            prefill_ids = pool["prefill_instances"]
            decode_ids = pool["decode_instances"]
            if prefill_ids != list(range(0, n_prefill)):
                err(f"{fname}: prefill instances should be [0..{n_prefill-1}]")
            if decode_ids != list(range(n_prefill, n)):
                err(f"{fname}: decode instances should be [{n_prefill}..{n-1}]")

            check_pd_types(instances, prefill_ids, decode_ids, fname)
            check_instance_limits(instances, prefill_ids, "prefill", fname)
            check_instance_limits(instances, decode_ids, "decode", fname)
            check_fallback_refs(pools, fname)

    if not any(e.startswith("me2_b3") for e in errors):
        print("  ✅ All B3 configs valid")


# =============================================================================
# B4 validation
# =============================================================================
def validate_b4():
    print("\n=== B4: Decode-biased-only ===")
    for n in GPU_BUDGETS:
        fname = f"me2_b4_{n}gpu.json"
        if not config_exists(fname):
            continue
        cfg = load(fname)
        node, instances = check_common(cfg, fname, n)

        pools = cfg["pools"]
        if len(pools) != 2:
            err(f"{fname}: B4 should have 2 pools, got {len(pools)}")
            continue

        db_pool = next((p for p in pools if p["id"] == "decode_biased"), None)
        gl_pool = next((p for p in pools if p["id"] == "general_long"), None)
        if not db_pool or not gl_pool:
            err(f"{fname}: missing required pool")
            continue

        # Pool order: decode_biased first
        if pools[0]["id"] != "decode_biased":
            err(f"{fname}: decode_biased must be first pool (for C4 matching)")
        if pools[1]["id"] != "general_long":
            err(f"{fname}: general_long must be second pool")

        # Admission
        if db_pool["admission"] != DECODE_BIASED_ADMISSION:
            err(f"{fname}: decode_biased admission should be {DECODE_BIASED_ADMISSION}, got {db_pool['admission']}")
        if gl_pool["admission"] != LONG_ADMISSION:
            err(f"{fname}: general_long admission should be {LONG_ADMISSION}, got {gl_pool['admission']}")

        # Fallback
        if db_pool["fallback"] != ["general_long"]:
            err(f"{fname}: decode_biased fallback should be ['general_long']")
        if gl_pool["fallback"] != []:
            err(f"{fname}: general_long should have no fallback")

        # GPU allocation: decode_biased ~30%
        n_db = len(db_pool["instances"])
        n_gl = len(gl_pool["instances"])
        if n_db + n_gl != n:
            err(f"{fname}: decode_biased({n_db})+general_long({n_gl}) != {n}")
        if abs(n_db / n - 0.30) > 0.15:
            err(f"{fname}: decode_biased ratio {n_db/n:.2f} too far from 0.30")

        check_pd_types(instances, [], [], fname)
        check_instance_limits(instances, db_pool["instances"], "decode_biased", fname)
        check_instance_limits(instances, gl_pool["instances"], "long", fname)
        check_fallback_refs(pools, fname)

    if not any(e.startswith("me2_b4") for e in errors):
        print("  ✅ All B4 configs valid")


# =============================================================================
# B5 validation
# =============================================================================
def validate_b5():
    print("\n=== B5: Coexisting ===")
    for n in GPU_BUDGETS:
        fname = f"me2_b5_{n}gpu.json"
        if not config_exists(fname):
            continue
        cfg = load(fname)
        node, instances = check_common(cfg, fname, n)

        pools = cfg["pools"]
        if len(pools) != 4:
            err(f"{fname}: B5 should have 4 pools, got {len(pools)}")
            continue

        # Pool order must be: short, decode_biased, pd_path, long
        expected_order = ["short", "decode_biased", "pd_path", "long"]
        actual_order = [p["id"] for p in pools]
        if actual_order != expected_order:
            err(f"{fname}: pool order should be {expected_order}, got {actual_order}")
            continue

        short = pools[0]
        decode_bias = pools[1]
        pd_path = pools[2]
        long_pool = pools[3]

        # ---- short pool ----
        if short["mode"] != "agg":
            err(f"{fname}: short pool should be agg")
        if short["admission"] != SHORT_ADMISSION:
            err(f"{fname}: short admission should be {SHORT_ADMISSION}, got {short['admission']}")
        if short["fallback"] != ["long"]:
            err(f"{fname}: short fallback should be ['long']")

        # ---- decode_biased pool ----
        if decode_bias["mode"] != "agg":
            err(f"{fname}: decode_biased pool should be agg")
        if decode_bias["admission"] != DECODE_BIASED_ADMISSION:
            err(f"{fname}: decode_biased admission should be {DECODE_BIASED_ADMISSION}, got {decode_bias['admission']}")
        if decode_bias["fallback"] != ["long"]:
            err(f"{fname}: decode_biased fallback should be ['long']")

        # ---- pd_path pool ----
        if pd_path["mode"] != "pd":
            err(f"{fname}: pd_path pool should be pd")
        expected_pd_admission = PREFILL_HEAVY_ADMISSION
        if pd_path["admission"] != expected_pd_admission:
            err(f"{fname}: pd_path admission should be {expected_pd_admission}, got {pd_path['admission']}")
        if pd_path["fallback"] != ["long"]:
            err(f"{fname}: pd_path fallback should be ['long']")

        # ---- long pool ----
        if long_pool["mode"] != "agg":
            err(f"{fname}: long pool should be agg")
        if long_pool["admission"] != LONG_ADMISSION:
            err(f"{fname}: long admission should be {LONG_ADMISSION}, got {long_pool['admission']}")
        if long_pool["fallback"] != []:
            err(f"{fname}: long should have no fallback")

        # Instance counts
        n_short = len(short["instances"])
        n_db = len(decode_bias["instances"])
        n_prefill = len(pd_path["prefill_instances"])
        n_decode = len(pd_path["decode_instances"])
        n_long = len(long_pool["instances"])
        total = n_short + n_db + n_prefill + n_decode + n_long
        if total != n:
            err(f"{fname}: short({n_short})+db({n_db})+prefill({n_prefill})+decode({n_decode})+long({n_long}) = {total} != {n}")

        # Verify pd_type
        prefill_ids = pd_path["prefill_instances"]
        decode_ids = pd_path["decode_instances"]
        check_pd_types(instances, prefill_ids, decode_ids, fname)
        check_instance_limits(instances, short["instances"], "short", fname)
        check_instance_limits(instances, decode_bias["instances"], "decode_biased", fname)
        check_instance_limits(instances, prefill_ids, "prefill", fname)
        check_instance_limits(instances, decode_ids, "decode", fname)
        check_instance_limits(instances, long_pool["instances"], "long", fname)

        # Verify instance indices are contiguous and in order
        start = 0
        if short["instances"] != list(range(start, start + n_short)):
            err(f"{fname}: short instances not contiguous from {start}")
        start += n_short
        if decode_bias["instances"] != list(range(start, start + n_db)):
            err(f"{fname}: decode_biased instances not contiguous from {start}")
        start += n_db
        if prefill_ids != list(range(start, start + n_prefill)):
            err(f"{fname}: prefill instances not contiguous from {start}")
        start += n_prefill
        if decode_ids != list(range(start, start + n_decode)):
            err(f"{fname}: decode instances not contiguous from {start}")
        start += n_decode
        if long_pool["instances"] != list(range(start, start + n_long)):
            err(f"{fname}: long instances not contiguous from {start}")

        check_fallback_refs(pools, fname)

        # Print allocation for review
        print(f"  {fname}: short={n_short}, db={n_db}, pref={n_prefill}, dec={n_decode}, long={n_long}")

    if not any(e.startswith("me2_b5") for e in errors):
        print("  ✅ All B5 configs valid")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    validate_b1()
    validate_b2()
    validate_b3()
    validate_b4()
    validate_b5()

    print(f"\n{'='*60}")
    if errors:
        print(f"❌ {len(errors)} ERRORS FOUND:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    else:
        print(f"✅ ALL {len(checked_files)} CONFIG FILES VALID — no errors found")
