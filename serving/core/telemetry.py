"""Low-overhead phase accounting for a simulator run.

The simulator deliberately keeps this module independent from scheduling and
ASTRA-Sim.  It is therefore safe to leave enabled in ``exact`` mode: it only
observes wall-clock work performed by the Python process.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
import json
import os
from time import perf_counter_ns


class SimulationTelemetry:
    """Aggregate timings and counters without changing simulation state."""

    def __init__(self):
        self.phase_ns = Counter()
        self.phase_calls = Counter()
        self.counters = Counter()
        self.samples = defaultdict(list)

    @contextmanager
    def phase(self, name):
        start = perf_counter_ns()
        try:
            yield
        finally:
            self.phase_ns[name] += perf_counter_ns() - start
            self.phase_calls[name] += 1

    def count(self, name, value=1):
        self.counters[name] += value

    def sample(self, name, value):
        self.samples[name].append(value)

    def as_dict(self):
        phases = {
            name: {
                "calls": self.phase_calls[name],
                "seconds": self.phase_ns[name] / 1_000_000_000,
            }
            for name in sorted(self.phase_ns)
        }
        sample_summary = {
            name: {
                "count": len(values),
                "total": sum(values),
                "min": min(values),
                "max": max(values),
            }
            for name, values in sorted(self.samples.items()) if values
        }
        return {
            "phases": phases,
            "counters": dict(sorted(self.counters.items())),
            "samples": sample_summary,
        }

    def write_json(self, path):
        path = os.path.abspath(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = f"{path}.tmp.{os.getpid()}"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.as_dict(), f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, path)
