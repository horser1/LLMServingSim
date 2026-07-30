"""Chakra graph generation backed by exact, content-addressed artifacts."""

import os

from .artifact_provider import GraphArtifactProvider
from .logger import get_logger
from .run_paths import input_path


logger = get_logger("GraphGenerator")
_artifact_provider = None


def configure_graph_artifacts(chakra_root, cache_dir=None, enable_cache=True,
                              converter_mode="inprocess", telemetry=None,
                              max_cache_bytes=None):
    """Configure the run-scoped provider used by ``generate_graph``.

    Keeping this explicit makes the legacy subprocess converter a useful
    parity fallback while ``inprocess`` is the exact default.
    """
    global _artifact_provider
    _artifact_provider = GraphArtifactProvider(
        chakra_root, cache_dir=cache_dir, enable_cache=enable_cache,
        converter_mode=converter_mode, telemetry=telemetry,
        max_bytes=max_cache_bytes)
    return _artifact_provider


def _provider():
    global _artifact_provider
    if _artifact_provider is None:
        cwd = os.getcwd()
        _artifact_provider = GraphArtifactProvider(
            os.path.join(cwd, "extern", "graph_frontend", "chakra"),
            cache_dir=os.path.join(cwd, ".llmservingsim-cache", "graphs"))
    return _artifact_provider


def generate_graph(batch, hardware, num_npus, node_id=0, instance_id=0,
                   npu_offset=0, enable_local_offloading=False, event=False,
                   workload_name=None, inputs_root=None, cleanup_trace=True,
                   descriptor=None):
    """Materialize an exact ET workload from a descriptor or legacy trace.

    No cache hit changes the workload submitted to ASTRA-Sim: it only reuses
    bytes previously generated from the identical canonical trace.
    """
    cwd = os.getcwd()
    if inputs_root is None:
        inputs_root = os.path.join(cwd, "inputs")
    if event:
        file_name = "event_handler"
    else:
        file_name = f"{hardware}/{batch.model}/instance{instance_id}_batch{batch.batch_id}"
    output_name = workload_name if workload_name else file_name

    trace_path = input_path(inputs_root, "trace", f"{file_name}.txt")
    output_path = input_path(inputs_root, "workload", output_name, "llm")
    if descriptor is not None:
        trace_bytes = descriptor.trace_bytes
    else:
        with open(trace_path, "rb") as f:
            trace_bytes = f.read()

    provider = _provider()
    telemetry = provider.telemetry
    if telemetry is not None:
        with telemetry.phase("chakra_conversion"):
            hit = provider.provide(trace_bytes, output_path, num_npus, npu_offset,
                                   enable_local_offloading)
        telemetry.count("graph_cache_hit" if hit else "graph_cache_miss")
        telemetry.sample("graph_trace_bytes", len(trace_bytes))
        telemetry.sample("graph_nodes", descriptor.node_count if descriptor is not None
                         else max(0, len(trace_bytes.decode("utf-8").splitlines()) - 3))
        et_bytes = sum(
            os.path.getsize(f"{output_path}.{rank}.et")
            for rank in range(npu_offset, npu_offset + num_npus))
        telemetry.sample("graph_et_bytes", et_bytes)
    else:
        hit = provider.provide(trace_bytes, output_path, num_npus, npu_offset,
                               enable_local_offloading)

    logger.debug("Graph artifact %s for %s", "cache hit" if hit else "cache miss", output_path,
                 extra={"node_id": node_id, "instance_id": instance_id})
    if cleanup_trace:
        try:
            os.remove(trace_path)
        except FileNotFoundError:
            pass
    return output_path
